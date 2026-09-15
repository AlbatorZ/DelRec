"""Dedicated Triton path for the hybrid recurrent delay.

A hybrid ``synaptic_recdel`` has ``recurrent_delays[i, o] = base[o] + offset[i, o]``
with ``base`` (per source ``o``) the only learned part and ``offset`` a frozen
integer in ``[0, hybrid_max_synaptic_delay]``. Numerically it is an ordinary
per-synapse delay, so:

  * forward  - the proven spike-sparse event-driven scatter
    (``_ed_scatter_compact_kernel``), same launch as ``EventDrivenSyn``.
  * recurrence backward - the event-driven reverse-BPTT (``_ed_bwd_kernel``),
    same launch as ``EventDrivenSyn``.
  * delay gradient - a fused Triton reduction (``_hybrid_dd_kernel``) using the
    closed-form derivative of the triangular delay kernel, instead of building the
    dense ``(N, N, L)`` soft mask and differentiating it with autograd. Offsets
    are frozen, so only ``d`` -> grad is needed and ``torch.autograd`` never sees
    an ``(N, N, L)`` intermediate. ``@triton.autotune`` sizes this kernel's tile
    to ``(N, L)``.

Autotune note: a kernel's tiling depends on the tensors it processes - here
``(B, N, L)``. ``num_samples`` and ``epochs`` are outer training-loop counts and
never reach a kernel; batch size ``B`` is the shape that matters and drives ``L``
and the reused scatter/BPTT launch heuristic below.

Exact - forward bit-identical, gradients within float tolerance - against
``multi_step_forward_v2`` in the supported regime (``decay_input=False``,
``use_sig_p=False``). ``synaptic_recdel.forward`` checks eligibility before calling
this path; unsupported settings use the event-driven / v2 path. Forward and
backward kernel errors propagate to the caller.
"""

import math
import os

import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover - Triton is an optional dependency
    _HAVE_TRITON = False

if _HAVE_TRITON:
    from .synaptic_eventdriven import (
        _ed_scatter_compact_kernel, _ed_bwd_kernel, _syn_consts, _surrogate_ids,
    )
    from .delays import build_syn_mask


def hybrid_kernel_usable(layer, x_seq):
    """Cheap gate for the dedicated path in ``synaptic_recdel.forward``."""
    return (
        _HAVE_TRITON
        and x_seq.is_cuda
        and not os.environ.get("DELREC_DISABLE_HYBRID_KERNEL")
        and not getattr(layer, "use_sig_p", False)
        and not getattr(layer.neuron_module, "decay_input", False)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reused-kernel launch heuristic (scales with the kernel-visible dims)
# ──────────────────────────────────────────────────────────────────────────────

def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _lane_warps(block_n):
    # Both reused kernels do a block-wide reduction/compaction over the BLOCK_N
    # lanes; num_warps * 32 must track BLOCK_N (idle lanes corrupt the prefix sum
    # in the scatter kernel). Fixed by N, not free to tune - same rule as
    # EventDrivenSyn. Only the fused delay-gradient kernel below is autotuned.
    return max(1, min(16, block_n // 32))


# ──────────────────────────────────────────────────────────────────────────────
# Fused delay-gradient reduction (closed form, autotuned)
# ──────────────────────────────────────────────────────────────────────────────

if _HAVE_TRITON:

    @triton.autotune(
        configs=[triton.Config({"BLOCK_O": bo}, num_warps=nw)
                 for bo in (16, 32, 64, 128) for nw in (1, 2, 4, 8)],
        key=["N", "LM"],   # the kernel-visible shape knobs; batch B drives Cstack's L
        warmup=3, rep=10,
    )
    @triton.jit
    def _hybrid_dd_kernel(Ct_ptr, Wt_ptr, dt_ptr, dd_ptr,
                          s, inv_s2,
                          N: tl.constexpr, LM: tl.constexpr, BLOCK_O: tl.constexpr):
        """dd_t[o, i] = W[i,o] * sum_{l=1..LM-1} Cstack[i,o,l] * d/dd clamp(s-|l-1-d[i,o]|,0)/s^2

        d/dd of the triangular kernel is sign(l-1-d)/s^2 inside its support and 0
        outside - exactly what autograd gives through build_syn_mask (clamp/abs).
        Ct is Cstack permuted to (o, i, l); Wt, dt are (o, i) = W.t(), d.t().
        """
        pid = tl.program_id(0)
        o = pid * BLOCK_O + tl.arange(0, BLOCK_O)
        om = o < N
        for i in range(0, N):
            w = tl.load(Wt_ptr + o * N + i, mask=om, other=0.0)
            dv = tl.load(dt_ptr + o * N + i, mask=om, other=0.0)
            row = o * (N * LM) + i * LM
            tap = tl.zeros((BLOCK_O,), dtype=tl.float32)
            for l in range(1, LM):
                a = l - 1.0 - dv
                g = tl.where(a > 0.0, inv_s2, tl.where(a < 0.0, -inv_s2, 0.0))
                g = tl.where(tl.abs(a) <= s, g, 0.0)
                c = tl.load(Ct_ptr + row + l, mask=om, other=0.0)
                tap += g * c
            tl.store(dd_ptr + o * N + i, w * tap, mask=om)


def _delay_grad(Cstack, W, d_eff, sigma, Lm):
    """(N_in, N_out) delay gradient from the lag-correlation stack, fused."""
    N = W.shape[0]
    s = 1.0 + float(sigma)
    Ct = Cstack.permute(1, 0, 2).contiguous()          # (o, i, l)
    Wt = W.t().contiguous()                            # (o, i)
    dt = d_eff.t().contiguous()                        # (o, i)
    dd_t = torch.empty(N, N, device=W.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_O"]),)
    _hybrid_dd_kernel[grid](Ct, Wt, dt, dd_t, s, 1.0 / (s * s), N=N, LM=Lm)
    return dd_t.t().contiguous()                       # (i, o)


# ──────────────────────────────────────────────────────────────────────────────
# autograd.Function
# ──────────────────────────────────────────────────────────────────────────────

class HybridSynDelay(torch.autograd.Function):
    """Event-driven forward + BPTT recurrence backward + fused closed-form delay
    gradient. ``d`` is the materialized ``(N_in, N_out)`` delay; its grad is
    collapsed to the ``(N_out,)`` learned base by the parametrization outside."""

    @staticmethod
    def forward(ctx, x, W, d, bias, drop, layer):
        assert not layer.use_sig_p, "hybrid kernel: use_sig_p not supported"
        assert not getattr(layer.neuron_module, "decay_input", False), \
            "hybrid kernel: decay_input not supported"
        device, dtype = x.device, torch.float32
        T, B, N = x.shape

        d_eff = d.detach().round() if layer.round_delays else d.detach()
        s = 1.0 + float(layer.sigma)
        L = int(math.ceil(1.0 + float(d_eff.max()) + s)) + 1
        MAXR = int(math.ceil(s)) + 1

        Wt = W.detach().t().contiguous()
        dt = d_eff.t().contiguous()
        s_vec = torch.full((N,), s, device=device, dtype=dtype)
        decay, charge, v_th, v_reset_val, soft = _syn_consts(layer)
        use_bias = bias is not None
        bias_k = bias.detach() if use_bias else torch.zeros(N, device=device, dtype=dtype)

        buf = torch.zeros(B, L, N, device=device, dtype=dtype)
        y = torch.empty(T, B, N, device=device, dtype=dtype)
        u = torch.empty(T, B, N, device=device, dtype=dtype)
        fired = torch.empty(B, N, device=device, dtype=torch.int32)
        BLOCK_N = _next_pow2(N)
        _ed_scatter_compact_kernel[(B,)](
            x.detach().contiguous(), Wt, dt, s_vec, bias_k, drop.detach(), y, buf, fired, u,
            decay, charge, v_th, v_reset_val, T, B,
            N=N, L=L, MAXR=MAXR, BLOCK_N=BLOCK_N, USE_BIAS=use_bias,
            STORE_U=True, SOFT=soft, num_warps=_lane_warps(BLOCK_N))

        ctx.save_for_backward(W.detach(), d_eff, drop.detach(), u, y)
        ctx.layer = layer
        ctx.shapes = (T, B, N, L, MAXR, decay, v_th, v_reset_val, soft, use_bias)
        return y

    @staticmethod
    def backward(ctx, gy):
        W, d_eff, drop, u, y = ctx.saved_tensors
        layer = ctx.layer
        T, B, N, L, MAXR, decay, v_th, v_reset_val, SOFT, use_bias = ctx.shapes
        device, dtype = gy.device, torch.float32
        DETACH = bool(layer.config.detach_reset)
        SURR, gamma, alpha = _surrogate_ids(layer)
        s_vec = torch.full((N,), 1.0 + float(layer.sigma), device=device, dtype=dtype)

        gr = torch.zeros(T, B, N, device=device, dtype=dtype)
        dx = torch.empty(T, B, N, device=device, dtype=dtype)
        BLOCK_N = _next_pow2(N)
        _ed_bwd_kernel[(B,)](
            gy.detach().contiguous(), u, drop, W.contiguous(), d_eff.contiguous(),
            s_vec, gr, dx,
            decay, v_th, v_reset_val, gamma, alpha, T, B,
            N=N, L=L, MAXR=MAXR, BLOCK_N=BLOCK_N, SOFT=SOFT, DETACH=DETACH, SURR=SURR,
            num_warps=_lane_warps(BLOCK_N), num_stages=1)

        dbias = gr.sum(dim=(0, 1)) if use_bias else None

        # C_l[i,o] = sum_{t>=l, b} gr[t,b,i] * y[t-l,b,o] - one GEMM per lag.
        # mask3 here is delay-kernel *values* (no autograd graph: d_eff is detached).
        mask3, Lm = build_syn_mask(d_eff, layer.sigma, None, False)
        Cstack = torch.zeros(N, N, Lm, device=device, dtype=dtype)
        for l in range(1, min(Lm, T)):
            M = T - l
            Cstack[:, :, l] = gr[l:T].reshape(M * B, N).t() @ y[0:M].reshape(M * B, N)

        need_W = ctx.needs_input_grad[1]
        need_d = ctx.needs_input_grad[2]
        # dW[i,o] = sum_l mask[i,o,l] C_l[i,o];  dd via the fused closed-form kernel.
        dW = (mask3 * Cstack).sum(-1) if need_W else None
        dd = _delay_grad(Cstack, W, d_eff, layer.sigma, Lm) if need_d else None

        # signature: x, W, d, bias, drop, layer
        return dx, dW, dd, dbias, None, None


_ANNOUNCED = set()


def hybrid_trainable_forward(layer, x_seq):
    """Autograd-enabled hybrid-delay forward for a hybrid ``synaptic_recdel``.
    Grads wire to ``recurrent_weights``, the ``(N_out,)`` learned delay base
    (through the parametrization on ``recurrent_delays``), ``recurrent_bias`` and
    ``x``. Raises on an unsupported regime or any Triton failure."""
    if not _HAVE_TRITON:
        raise RuntimeError("Triton unavailable")
    T, B, N = x_seq.shape
    if N not in _ANNOUNCED:
        _ANNOUNCED.add(N)
        print(f"[delrec] hybrid recurrent delay: fused Triton kernel active "
              f"(N={N}, T={T}, B={B})", flush=True)
    if layer.training and getattr(layer.config, "recurrent_dropout_rate", 0.0) > 0:
        drop = layer.dropout(torch.ones(B, N, device=x_seq.device, dtype=x_seq.dtype))
    else:
        drop = torch.ones(B, N, device=x_seq.device, dtype=x_seq.dtype)
    bias = layer.recurrent_bias if layer.use_rec_bias else None
    y = HybridSynDelay.apply(x_seq, layer.recurrent_weights, layer.recurrent_delays,
                             bias, drop, layer)
    layer.x_seq = x_seq
    layer.rec_now_seq = None
    if layer.store_v_seq:
        layer.v_seq = None
    return y
