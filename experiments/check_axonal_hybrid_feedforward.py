"""Compare hybrid vs axonal feedforward outputs and gradients.

Run: .venv/bin/python experiments/check_axonal_hybrid_feedforward.py
     .venv/bin/python experiments/check_axonal_hybrid_feedforward.py --hybrid-max-delay 0
     .venv/bin/python experiments/check_axonal_hybrid_feedforward.py --hybrid-max-delay 4
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
    SNN_axonal_feedforward_delays,
    SNN_hybrid_feedforward_delays,
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
        no_delay_in_first_layer=False, no_delay_in_last_layer=False,
        neuron_module=neuron.LIFNode, tau=2.0, decay_input=False,
        v_reset=0.0, v_threshold=0.7, detach_reset=False,
        surrogate_function=surrogate.ATan(alpha=2.0), step_mode="m",
        backend="torch", store_v_seq=False,
        init_ff_weights="default", init_dcls_weights="default",
        DCLSversion="gauss", kernel_count=1, max_feedforward_delay=5,
        left_padding=4, right_padding=0, init_pos_a=-1.5, init_pos_b=1.5,
        hybrid_max_synaptic_delay=0, hybrid_delay_seed=123,
    )
    if args.hybrid_max_delay is not None:
        config.hybrid_max_synaptic_delay = args.hybrid_max_delay
    if config.hybrid_max_synaptic_delay < 0:
        parser.error("The maximum hybrid offset must be nonnegative")
    check_equivalence = config.hybrid_max_synaptic_delay == 0
    axonal = SNN_axonal_feedforward_delays(deepcopy(config))
    hybrid = SNN_hybrid_feedforward_delays(deepcopy(config))
    for model in (axonal, hybrid):
        assert not any(isinstance(m, axonal_recdel) for m in model.modules())
        model.train()
        functional.reset_net(model)
        model.zero_grad(set_to_none=True)

    ax_filters = [m for m in axonal.layers if isinstance(m, dcls_module)]
    ax_linears = [m for m in axonal.layers if isinstance(m, torch.nn.Linear)]
    hy_filters = [m for m in hybrid.layers if isinstance(m, dcls_module)]

    # Map corresponding leaf parameters explicitly: the hybrid stores the
    # effective dense positions as a computed tensor, not as a learned matrix.
    pairs = []
    initial_offsets = []
    with torch.no_grad():
        for i, (delay, linear, dense) in enumerate(zip(ax_filters, ax_linears, hy_filters, strict=True)):
            # Positive hidden weights produce a nontrivial train with both
            # spikes and silent time steps; the output projection stays signed.
            if i == 0:
                linear.weight.uniform_(0.2, 0.6)
            dense.weight.copy_(linear.weight.unsqueeze(-1))
            base = learned_delay_parameter(dense, "P")
            base.copy_(delay.P[:, :, 0, :].unsqueeze(1))
            for module in (delay, dense):
                # Same fixed Gaussian width, as with scheduled widths in MEM.
                module.SIG.fill_(0.8)
                module.SIG.requires_grad_(False)
            offsets = dense.parametrizations.P[0].offsets
            if check_equivalence:
                assert torch.count_nonzero(offsets) == 0
            assert not offsets.requires_grad
            initial_offsets.append(offsets.clone())
            pairs.extend([
                (f"projection {i} weights", linear.weight, dense.weight),
                (f"projection {i} axonal positions", delay.P, base),
            ])

    # Confirm the comparison covers every learned parameter in both networks.
    assert {id(a) for _, a, _ in pairs} == {id(p) for p in axonal.parameters() if p.requires_grad}
    assert {id(h) for _, _, h in pairs} == {id(p) for p in hybrid.parameters() if p.requires_grad}

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
        # Weight shapes: (out, in) vs (out, in, 1).
        # Position shapes: (z1, in, 1, 1) vs (1, 1, in, 1).
        # Flattening aligns the same logical parameters with kernel_count=1.
        compare(label + " gradient", a.grad, h.grad, gradient=True)
    for module, initial in zip(hy_filters, initial_offsets, strict=True):
        offsets = module.parametrizations.P[0].offsets
        assert not offsets.requires_grad and offsets.grad is None
        assert torch.equal(offsets, initial), "Fixed offsets changed"
    for module in ax_filters:
        assert not module.weight.requires_grad and module.weight.grad is None
    if failures:
        raise AssertionError("\n".join(failures))
    if check_equivalence:
        print("\nPASS: outputs, loss, and all learned-parameter/input gradients are close.")
    else:
        print("\nInspection complete: output, loss, and gradient differences reported; closeness not asserted.")
    print("Fixed offsets are unchanged and have no gradients. No optimizer update was performed.")


if __name__ == "__main__":
    main()
