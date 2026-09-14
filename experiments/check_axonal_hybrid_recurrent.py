"""Compare hybrid vs axonal recurrent outputs and gradients.

Run: .venv/bin/python experiments/check_axonal_hybrid_recurrent.py
     .venv/bin/python experiments/check_axonal_hybrid_recurrent.py --hybrid-max-delay 0
     .venv/bin/python experiments/check_axonal_hybrid_recurrent.py --hybrid-max-delay 4
All settings and data are defined here; no experiment config or dataset is used.
Biases are disabled. "Projection i" means feedforward mapping i: with one hidden
layer, projection 0 maps input -> hidden and projection 1 maps hidden -> output.
It denotes a learned channel mapping, not necessarily a mathematical projector.
The two implementations use different floating-point operation orders, so the
check asserts numerical closeness only for zero offsets. Positive maximum
offsets enable inspection mode: differences are reported without requiring
equivalence. Structural, finite-value and frozen-parameter checks remain active.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from spikingjelly.activation_based import functional, neuron, surrogate

from delrec.delay_layers import axonal_recdel
from delrec.networks import (
    SNN_recurrent_delays,
    SNN_recurrent_hybrid_delays,
    dcls_module,
    learned_delay_parameter,
    spike_registrator,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-max-delay", "--hybrid-max-synaptic-delay",
                        dest="hybrid_max_delay", type=int, default=None,
                        help="Override the hardcoded maximum offset; 0 enables equivalence assertions.")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = SimpleNamespace(
        dataset="MEM", input_size=16, hidden_layers=[64], output_size=4,
        bias=False, use_batch_norm=False, feedforward_dropout_rate=0.0,
        no_recurrence_in_last_layer=False, recurrent_dropout_rate=0.0,
        init_rec_weights="orthogonal", rec_delay_init_gain=0.5,
        use_rec_bias=False, init_rec_delay="uniform", init_recdel_offset=0.25,
        max_rec_delay=3.0, use_sig_p=False, sigma_init=0.0, round_delays=False,
        neuron_module=neuron.LIFNode, tau=2.0, decay_input=False,
        v_reset=0.0, v_threshold=0.7, detach_reset=False,
        surrogate_function=surrogate.ATan(alpha=2.0), step_mode="m",
        backend="torch", store_v_seq=False,
        init_ff_weights="default",
        # Used only for bookkeeping in the shared hybrid mixin; no FF delays.
        max_feedforward_delay=5,
        hybrid_max_synaptic_delay=0, hybrid_delay_seed=123,
    )
    if args.hybrid_max_delay is not None:
        config.hybrid_max_synaptic_delay = args.hybrid_max_delay
    if config.hybrid_max_synaptic_delay < 0:
        parser.error("The maximum hybrid offset must be nonnegative")
    check_equivalence = config.hybrid_max_synaptic_delay == 0
    axonal = SNN_recurrent_delays(deepcopy(config))
    hybrid = SNN_recurrent_hybrid_delays(deepcopy(config))
    for model in (axonal, hybrid):
        assert not any(isinstance(m, dcls_module) for m in model.modules())
        model.train()
        functional.reset_net(model)
        model.zero_grad(set_to_none=True)

    ax_linears = [m for m in axonal.layers if isinstance(m, torch.nn.Linear)]
    hy_linears = [m for m in hybrid.layers if isinstance(m, torch.nn.Linear)]
    ax_recurrent = [m for m in axonal.layers if isinstance(m, axonal_recdel)]
    hy_recurrent = [m for m in hybrid.layers if isinstance(m, axonal_recdel)]
    assert len(ax_recurrent) == len(hy_recurrent) == len(config.hidden_layers)

    pairs = []
    initial_offsets = []
    with torch.no_grad():
        for i, (a, h) in enumerate(zip(ax_linears, hy_linears, strict=True)):
            # Moderate positive input weights create spikes and silent steps,
            # while leaving the final output projection signed.
            if i == 0:
                a.weight.uniform_(0.02, 0.12)
            h.weight.copy_(a.weight)
            pairs.extend([
                (f"projection {i} weights", a.weight, h.weight),
            ])
        for i, (a, h) in enumerate(zip(ax_recurrent, hy_recurrent, strict=True)):
            # Both use their own PyTorch v2 implementation: axonal filtering
            # before mixing vs dense synaptic filtering with shared base delays.
            a.forward_version = h.forward_version = "v2"
            h.recurrent_weights.copy_(a.recurrent_weights)
            base = learned_delay_parameter(h, "recurrent_delays")
            base.copy_(a.recurrent_delays)
            offsets = h.parametrizations.recurrent_delays[0].offsets
            if check_equivalence:
                assert torch.count_nonzero(offsets) == 0
            assert not offsets.requires_grad
            initial_offsets.append(offsets.clone())
            pairs.extend([
                (f"recurrent {i} weights", a.recurrent_weights, h.recurrent_weights),
                (f"recurrent {i} axonal delays", a.recurrent_delays, base),
            ])

    x = torch.rand(16, 2, config.input_size)  # (time, batch, input neurons)
    xa = x.clone().requires_grad_(True)
    xh = x.clone().requires_grad_(True)
    ya, yh = axonal(xa), hybrid(xh)
    assert ya.shape == yh.shape == (16, 2, config.output_size)
    target = torch.randn_like(ya)
    loss_a = (ya - target).square().mean()
    loss_h = (yh - target).square().mean()

    failures = []

    def compare(label, a, h, *, gradient=False):
        assert a is not None and h is not None, f"Missing tensor/gradient: {label}"
        a, h = a.detach().reshape(-1), h.detach().reshape(-1)
        assert a.shape == h.shape, label
        assert torch.isfinite(a).all() and torch.isfinite(h).all(), label
        if gradient and check_equivalence:
            assert a.abs().max() > 0 and h.abs().max() > 0, f"Vacuous zero-gradient check: {label}"
        absolute_diff = (a - h).abs()
        error = absolute_diff.max().item()
        # Elementwise symmetric relative difference. Mean magnitudes avoid
        # cancellation for signed gradients; equal zeros have zero difference.
        denominator = (a.abs() + h.abs()) / 2
        normalized_diff = absolute_diff / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator))
        print(
            f"{label:34s} exact={str(torch.equal(a, h)):5s}  max_abs_diff={error:.3e}"
            f"  max_normalized_diff={normalized_diff.max().item():.3e}"
            f"  mean_normalized_diff={normalized_diff.mean().item():.3e}"
        )
        if check_equivalence:
            try:
                torch.testing.assert_close(a, h, rtol=1e-4, atol=1e-6)
            except AssertionError as exc:
                failures.append(f"{label}: {exc}")

    topology = " -> ".join(map(str, [config.input_size] + config.hidden_layers + [config.output_size]))
    print(f"CPU float32 | topology {topology} | offsets in [0, {config.hybrid_max_synaptic_delay}]")
    print("Equivalence assertions ON (rtol=1e-4, atol=1e-6)." if check_equivalence
          else "Inspection mode: equivalence assertions OFF; differences are expected.")
    print("Normalized diff: |a-h| / ((|a|+|h|)/2), elementwise; both zero -> 0.")
    print("\nForward (before any optimizer update):")
    compare("output sequence", ya, yh)
    compare("dummy MSE loss", loss_a, loss_h)

    loss_a.backward()
    loss_h.backward()
    print("\nBackward:")
    compare("input gradient", xa.grad, xh.grad, gradient=True)
    for label, a, h in pairs:
        # Compare learned base vectors, not the hybrid computed (N, N) matrix.
        compare(label + " gradient", a.grad, h.grad, gradient=True)
    for module, initial in zip(hy_recurrent, initial_offsets, strict=True):
        offsets = module.parametrizations.recurrent_delays[0].offsets
        assert not offsets.requires_grad and offsets.grad is None
        assert torch.equal(offsets, initial), "Fixed offsets changed"
    if failures:
        raise AssertionError("\n".join(failures))
    if check_equivalence:
        print("\nPASS: outputs, loss, and all learned-parameter/input gradients are close.")
    else:
        print("\nInspection complete: output, loss, and gradient differences reported; closeness not asserted.")
    print("Fixed offsets are unchanged and have no gradients. No optimizer update was performed.")


if __name__ == "__main__":
    main()
