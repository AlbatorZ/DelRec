"""Delay-parametrization x pathway memorization experiment (3 x 2).

All measurements use training data. Six models, each a delay parametrization
(axonal / synaptic / hybrid) on a pathway (feedforward only / recurrent only):

  axonal    one learned delay per source neuron, shared by its connections
  synaptic  one learned delay per connection
  hybrid    learned axonal delay + a fixed random per-synapse integer offset

All six are seeded from ONE global reference: ``generate_matched_parameters``
builds one RNG-consistent set of feedforward weights/biases, feedforward axonal
delays, recurrent weights/biases and recurrent axonal delays, which are then
injected into every model. Within each pathway group the axonal and synaptic
models therefore start from identical initial outputs (asserted); the hybrids
keep their own fixed random offsets, so they start from different effective
delays by design.

Run: .venv/bin/python experiments/compare_mem_ziyad.py
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from train_mem import ROOT, Config, networks, run, torch, plt
from delrec.delay_layers import axonal_recdel
from delrec.networks import dcls_module, learned_delay_parameter
from delrec.training.mem import set_epoch
from delrec.utils import reset_states, seed_everything


# (legend label, output-subdirectory slug, network class), grouped by pathway and
# ordered axonal / synaptic / hybrid within each group. matched_models returns the
# built (config, model) pairs in this exact order.
MODELS = (
    ('Feedforward axonal',   'ff_axonal',      'SNN_axonal_feedforward_delays'),
    ('Feedforward synaptic', 'ff_synaptic',    'SNN_synaptic_feedforward_delays'),
    ('Feedforward hybrid',   'ff_hybrid',      'SNN_hybrid_feedforward_delays'),
    ('Recurrent axonal',     'rec_axonal',     'SNN_axonal_recurrent_delays'),
    ('Recurrent synaptic',   'rec_synaptic',   'SNN_synaptic_recurrent_delays'),
    ('Recurrent hybrid',     'rec_hybrid',     'SNN_recurrent_hybrid_delays'),
)

# axonal / synaptic init outputs must agree within each pathway group.
MATCHED_GROUPS = (
    ('feedforward', 'Feedforward axonal', 'Feedforward synaptic'),
    ('recurrent',   'Recurrent axonal',   'Recurrent synaptic'),
)

PARAM_COLORS = {'axonal': 'tab:blue', 'synaptic': 'tab:orange', 'hybrid': 'tab:green'}
# Pathway -> line style, so the six curves stay distinct (colour encodes the
# parametrization, style encodes the pathway).
PATHWAY_STYLES = {'Feedforward': '-', 'Recurrent': '--'}


def generate_matched_parameters(config):
    """Build one RNG-consistent set of feedforward + recurrent weights/delays.

    This is the shared reference every matched model is injected from: for each
    feedforward projection (one per hidden layer plus the output), a depthwise
    unit-weight delay filter (one learned axonal delay per source neuron, ``P``
    uniform-initialized and clamped like ``SNN_axonal_feedforward_delays``) followed
    by a plain ``nn.Linear`` projection; for each hidden layer, an ``axonal_recdel``
    recurrent layer (self-initializing its own weights/bias/delays). Mirrors what a
    both-pathways axonal model would build, without needing that network class.

    ``ff_w`` / ``ff_b`` / ``ff_delay`` (shape (in_channels, kernel_count)) are one
    entry per feedforward projection, in depth order; ``rec_*`` are one entry per
    recurrent layer (every hidden layer, never the output).
    """
    if config.kernel_count != 1:
        raise ValueError('matched_models requires kernel_count=1 (one delay per axon/synapse).')

    canon = {k: [] for k in ('ff_w', 'ff_b', 'ff_delay', 'rec_w', 'rec_b', 'rec_delay', 'rec_p_spread')}
    dim = config.input_size
    for idx, out_dim in enumerate(list(config.hidden_layers) + [config.output_size]):
        delay_config = deepcopy(config)
        delay_config.bias = False
        delay = dcls_module(delay_config, in_channels=dim, out_channels=dim, groups=dim)
        torch.nn.init.constant_(delay.weight, 1.0)
        torch.nn.init.uniform_(delay.P, a=config.init_pos_a, b=config.init_pos_b)
        delay.clamp_parameters()
        if config.DCLSversion == 'gauss':
            torch.nn.init.constant_(delay.SIG, config.siginit)
        canon['ff_delay'].append(delay.P.detach()[0, :, 0, :].clone())

        proj = torch.nn.Linear(dim, out_dim, bias=config.bias)
        canon['ff_w'].append(proj.weight.detach().clone())
        canon['ff_b'].append(proj.bias.detach().clone() if proj.bias is not None else None)

        if idx < len(config.hidden_layers):        # a recurrent layer follows every hidden layer
            rec = axonal_recdel(config, out_dim, config.neuron_module)
            canon['rec_w'].append(rec.recurrent_weights.detach().clone())
            canon['rec_b'].append(rec.recurrent_bias.detach().clone() if rec.use_rec_bias else None)
            canon['rec_delay'].append(rec.recurrent_delays.detach().clone())
            canon['rec_p_spread'].append(rec.p_spread.detach().clone() if rec.use_sig_p else None)
        dim = out_dim
    return canon


def _inject(model, canon):
    """Overwrite a model's shared components with the reference tensors."""
    fi = ri = 0
    with torch.no_grad():
        for m in model.layers:
            if isinstance(m, dcls_module) and m.weight.shape[1] == 1:
                # Depthwise delay filter: per-source delay, unit weights left frozen.
                # The projection's Linear follows and advances `fi`.
                learned_delay_parameter(m, 'P').copy_(
                    canon['ff_delay'][fi].view(1, m.in_channels, 1, m.kernel_count))
            elif isinstance(m, dcls_module):
                # Dense feedforward projection: DCLS weight carries the Linear
                # weight; every target shares the one per-source delay.
                m.weight.copy_(canon['ff_w'][fi].unsqueeze(-1).expand(-1, -1, m.kernel_count))
                if m.bias is not None and canon['ff_b'][fi] is not None:
                    m.bias.copy_(canon['ff_b'][fi])
                leaf = learned_delay_parameter(m, 'P')                # (1, O, in, k); O==1 for hybrid
                leaf.copy_(canon['ff_delay'][fi]
                           .view(1, 1, m.in_channels, m.kernel_count)
                           .expand(1, leaf.shape[1], m.in_channels, m.kernel_count))
                fi += 1
            elif isinstance(m, torch.nn.Linear):
                m.weight.copy_(canon['ff_w'][fi])
                if m.bias is not None and canon['ff_b'][fi] is not None:
                    m.bias.copy_(canon['ff_b'][fi])
                fi += 1
            elif isinstance(m, axonal_recdel):
                m.recurrent_weights.copy_(canon['rec_w'][ri])
                if getattr(m, 'use_rec_bias', False) and canon['rec_b'][ri] is not None:
                    m.recurrent_bias.copy_(canon['rec_b'][ri])
                leaf = learned_delay_parameter(m, 'recurrent_delays')  # (N,) axonal/hybrid, (N, N) synaptic
                base = canon['rec_delay'][ri]
                leaf.copy_(base if leaf.dim() == 1 else base[None, :].expand_as(leaf))
                if hasattr(m, 'p_spread') and canon['rec_p_spread'][ri] is not None:
                    m.p_spread.copy_(canon['rec_p_spread'][ri])
                ri += 1


def matched_models(config):
    """Build the six models, all seeded from one global reference."""
    config = deepcopy(config)
    assert not config.no_delay_in_first_layer and not config.no_delay_in_last_layer, \
        'the reference needs a delay filter on every feedforward projection'
    # The recurrent-only models honour this flag; generate_matched_parameters ignores
    # it. Force recurrence in every hidden layer so they match the reference.
    config.no_recurrence_in_last_layer = False

    seed_everything(config.seed)
    canon = generate_matched_parameters(config)

    built = {}
    for label, _slug, class_name in MODELS:
        cfg = deepcopy(config)
        cfg.model = class_name
        model = getattr(networks, class_name)(cfg)
        _inject(model, canon)
        set_epoch(model, cfg, 0)
        model.eval()
        built[label] = (cfg, model)

    with torch.no_grad():
        probe = torch.rand(config.time_window, 2, config.input_size,
                           generator=torch.Generator().manual_seed(123))
        for name, axonal_label, synaptic_label in MATCHED_GROUPS:
            out_axonal = built[axonal_label][1](probe)
            out_synaptic = built[synaptic_label][1](probe)
            torch.testing.assert_close(out_axonal, out_synaptic, atol=1e-5, rtol=1e-5,
                                       msg=f'{name}: axonal vs synaptic initial outputs differ')
            reset_states(built[axonal_label][1])
            reset_states(built[synaptic_label][1])

    return [built[label] for label, _slug, _class_name in MODELS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('epochs', 'seed', 'dataset-seed', 'num-samples', 'hybrid-max-synaptic-delay', 'hybrid-delay-seed', 'delay-diagnostics-every'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--task-type', choices=['temporal', 'spatial'])
    parser.add_argument('--hidden-layers', help='Comma-separated widths')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    config = Config()
    for key in ('epochs', 'seed', 'dataset_seed', 'num_samples', 'task_type', 'hybrid_max_synaptic_delay', 'hybrid_delay_seed', 'delay_diagnostics_every'):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    if args.hidden_layers:
        config.hidden_layers = [int(n) for n in args.hidden_layers.split(',')]
    if min(config.delay_diagnostics_every, config.epochs, config.num_samples, *config.hidden_layers) < 1:
        parser.error('Diagnostic interval, epochs, samples and layer widths must be positive')
    torch.set_num_threads(config.cpu_threads)
    out = args.out or ROOT / 'exp' / 'MEM' / 'delay_comparison' / (
        f'{config.task_type}_seed{config.seed}_{datetime.now():%Y-%m-%d-%H-%M-%S-%f}')
    out.mkdir(parents=True, exist_ok=True)
    models = matched_models(config)
    print('Per pathway group: axonal/synaptic initial outputs matched; hybrids keep their own fixed offsets.', flush=True)
    results = {}
    histories = {}
    for (label, slug, _class_name), (cfg, model) in zip(MODELS, models, strict=True):
        history, final, directory = run(cfg, torch.device(args.device), out / slug, model=model)
        final['feedforward_delay_parameters'] = sum(learned_delay_parameter(m, 'P').numel() for m in model.layers if isinstance(m, dcls_module))
        final['recurrent_delay_parameters'] = sum(learned_delay_parameter(m, 'recurrent_delays').numel() for m in model.layers if isinstance(m, axonal_recdel))
        final['fixed_synaptic_offsets'] = sum(b.numel() for name, b in model.named_buffers() if name.endswith('.offsets'))
        results[label] = final
        histories[label] = history
    ref_data = torch.load(out / MODELS[0][1] / 'dataset.pt', weights_only=True)
    for _label, slug, _class_name in MODELS[1:]:
        other = torch.load(out / slug / 'dataset.pt', weights_only=True)
        assert all(torch.equal(ref_data[k], other[k]) for k in ref_data)
    (out / 'comparison.json').write_text(json.dumps({
        'initialization': 'All six models injected from one both-pathways axonal master '
                          '(feedforward weights/biases and axonal delays, recurrent '
                          'weights/biases and axonal delays). Within each pathway group '
                          'axonal and synaptic start from identical outputs; hybrids add '
                          'fixed random synaptic offsets before training.',
        'results': results,
    }, indent=2))
    plot_comparison(histories, results, config, out)


def plot_comparison(histories, results, config, out):
    """Render current or saved comparison metrics without rerunning training."""
    colors = [PARAM_COLORS[label.split()[-1].lower()] for label in results]
    styles = [PATHWAY_STYLES[label.split()[0]] for label in results]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), layout='constrained')
    for (label, history), color, style in zip(histories.items(), colors, styles):
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r['loss'] for r in history], label=label, color=color, linestyle=style)
        axes[1].plot(epochs, [r['accuracy_percent'] for r in history], label=label, color=color, linestyle=style)
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training loss')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Training accuracy', ylim=(0, 105))
    for axis in axes[:2]:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    values = [r['accuracy_percent'] for r in results.values()]
    labels = [f"{name}\n{result['trainable_parameters']:,} params"
              for name, result in results.items()]
    bars = axes[2].bar(labels, values, color=colors)
    axes[2].bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=4, fontsize=8)
    axes[2].set(ylabel='Accuracy (%)', title='Final training accuracy', ylim=(0, 110))
    axes[2].tick_params(axis='x', labelrotation=30, labelsize=8)
    for tick in axes[2].get_xticklabels():
        tick.set_ha('right')
    fig.suptitle(f'Delay parametrization x pathway (3 x 2) | {config.task_type}, '
                 f'{config.num_samples} samples, topology {config.input_size} → '
                 + ' → '.join(map(str, config.hidden_layers + [config.output_size])))
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'delay_comparison.{extension}', dpi=180)
    plt.close(fig)
    print(f'Comparison saved: {out}', flush=True)


if __name__ == '__main__':
    main()