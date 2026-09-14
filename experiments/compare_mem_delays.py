"""Axonal/synaptic/hybrid comparison on one selected delay pathway; all measurements use training data.

Run: .venv/bin/python experiments/compare_mem_delays.py
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


def matched_models(config, pathway="recurrent"):
    """Match weights and initial functions; synaptic delays subsequently untie."""
    suffixes = {'recurrent': 'recurrent_delays',
                'feedforward': 'feedforward_delays'}
    if pathway not in suffixes:
        raise ValueError('pathway must be recurrent or feedforward')
    suffix = suffixes[pathway]
    config = deepcopy(config)
    if pathway == 'recurrent':
        # Existing recurrent classes honor this flag; the comparison needs
        # recurrence in every hidden layer, including a single-hidden-layer run.
        config.no_recurrence_in_last_layer = False
    ax_config, sy_config = deepcopy(config), deepcopy(config)
    ax_config.delay_pathway = sy_config.delay_pathway = pathway
    ax_config.model = ('SNN_axonal_recurrent_delays' if pathway == 'recurrent'
                       else f'SNN_axonal_{suffix}')
    sy_config.model = f'SNN_synaptic_{suffix}'
    seed_everything(config.seed)
    ax = getattr(networks, ax_config.model)(ax_config)
    sy = getattr(networks, sy_config.model)(sy_config)
    ax_weights = [m for m in ax.layers if isinstance(m, torch.nn.Linear)]
    sy_weights = [m for m in sy.layers if isinstance(m, (torch.nn.Linear, dcls_module))]
    ax_delays = [m for m in ax.layers if isinstance(m, dcls_module)]
    sy_delays = [m for m in sy.layers if isinstance(m, dcls_module)]
    ax_recs = [m for m in ax.layers if isinstance(m, axonal_recdel)]
    sy_recs = [m for m in sy.layers if isinstance(m, axonal_recdel)]
    with torch.no_grad():
        for a, s in zip(ax_weights, sy_weights, strict=True):
            s.weight.copy_(a.weight.unsqueeze(-1) if isinstance(s, dcls_module) else a.weight)
            if s.bias is not None:
                s.bias.copy_(a.bias)
        for a, s in zip(ax_delays, sy_delays, strict=True):
            # P axes are (spatial dimension, output channel, input channel, tap).
            s.P.copy_(a.P[:, :, 0, :].unsqueeze(1).expand_as(s.P))
        for a, s in zip(ax_recs, sy_recs, strict=True):
            s.recurrent_weights.copy_(a.recurrent_weights)
            s.recurrent_delays.copy_(a.recurrent_delays[None, :].expand_as(s.recurrent_delays))
            if a.use_rec_bias:
                s.recurrent_bias.copy_(a.recurrent_bias)
        for model, cfg in ((ax, ax_config), (sy, sy_config)):
            set_epoch(model, cfg, 0)
            model.eval()
        generator = torch.Generator().manual_seed(123)
        probe = torch.rand(config.time_window, 2, config.input_size, generator=generator)
        torch.testing.assert_close(ax(probe), sy(probe), atol=1e-5, rtol=1e-5)
        reset_states(ax)
        reset_states(sy)
    hy_config = deepcopy(config)
    hy_config.delay_pathway = pathway
    hy_config.model = ('SNN_recurrent_hybrid_delays' if pathway == 'recurrent'
                       else 'SNN_hybrid_feedforward_delays')
    hy = getattr(networks, hy_config.model)(hy_config)
    # Copy common weights/biases and the learned base delays. Fixed offsets are
    # retained: hybrid starts with different effective delays by design.
    with torch.no_grad():
        for source, target in zip(sy.layers, hy.layers, strict=True):
            if isinstance(target, dcls_module):
                target.weight.copy_(source.weight)
                if target.bias is not None:
                    target.bias.copy_(source.bias)
                learned_delay_parameter(target, 'P').copy_(source.P[:, :1])
            elif isinstance(target, axonal_recdel):
                target.recurrent_weights.copy_(source.recurrent_weights)
                if target.use_rec_bias:
                    target.recurrent_bias.copy_(source.recurrent_bias)
                learned_delay_parameter(target, 'recurrent_delays').copy_(source.recurrent_delays[0])
                if hasattr(target, 'p_spread'):
                    target.p_spread.copy_(source.p_spread)
            else:
                target.load_state_dict(source.state_dict())
    return [(ax_config, ax), (sy_config, sy), (hy_config, hy)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('epochs', 'seed', 'dataset-seed', 'num-samples', 'hybrid-max-synaptic-delay', 'hybrid-delay-seed'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--pathway', choices=['recurrent', 'feedforward'], default='recurrent')
    parser.add_argument('--task-type', choices=['temporal', 'spatial'], default = 'temporal')
    parser.add_argument('--hidden-layers', help='Comma-separated widths')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    config = Config()
    config.delay_pathway = args.pathway
    for key in ('epochs', 'seed', 'dataset_seed', 'num_samples', 'task_type', 'hybrid_max_synaptic_delay', 'hybrid_delay_seed'):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    if args.hidden_layers:
        config.hidden_layers = [int(n) for n in args.hidden_layers.split(',')]
    if min(config.epochs, config.num_samples, *config.hidden_layers) < 1:
        parser.error('Epochs, samples and layer widths must be positive')
    torch.set_num_threads(config.cpu_threads)
    out = args.out or ROOT / 'exp' / 'MEM' / 'delay_comparison' / (
        f'{args.pathway}_{config.task_type}_seed{config.seed}_{datetime.now():%Y-%m-%d-%H-%M-%S-%f}')
    out.mkdir(parents=True, exist_ok=True)
    pair = matched_models(config, pathway=args.pathway)
    print('Axonal/synaptic initial outputs matched; hybrid shares base parameters plus fixed random offsets.', flush=True)
    results = {}
    histories = {}
    for label, (cfg, model) in zip(('Axonal', 'Synaptic', 'Hybrid'), pair):
        history, final, directory = run(cfg, torch.device(args.device), out / label.lower(), model=model)
        final['feedforward_delay_parameters'] = sum(learned_delay_parameter(m, 'P').numel() for m in model.layers if isinstance(m, dcls_module))
        final['recurrent_delay_parameters'] = sum(learned_delay_parameter(m, 'recurrent_delays').numel() for m in model.layers if isinstance(m, axonal_recdel))
        final['fixed_synaptic_offsets'] = sum(b.numel() for name, b in model.named_buffers() if name.endswith('.offsets'))
        results[label] = final
        histories[label] = history
    a_data = torch.load(out / 'axonal' / 'dataset.pt', weights_only=True)
    for label in ('synaptic', 'hybrid'):
        other = torch.load(out / label / 'dataset.pt', weights_only=True)
        assert all(torch.equal(a_data[k], other[k]) for k in a_data)
    (out / 'comparison.json').write_text(json.dumps({
        'initialization': 'Matched weights, biases, and base axonal delays. Axonal/synaptic initial outputs equivalent; hybrid adds fixed random synaptic offsets before training.',
        'results': results,
    }, indent=2))
    plot_comparison(histories, results, config, out)


def plot_comparison(histories, results, config, out):
    """Render current or saved comparison metrics without rerunning training."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for label, history in histories.items():
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r['loss'] for r in history], label=label)
        axes[1].plot(epochs, [r['accuracy_percent'] for r in history], label=label)
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training loss')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Training accuracy', ylim=(0, 105))
    for axis in axes[:2]:
        axis.legend()
        axis.grid(alpha=0.25)
    values = [r['accuracy_percent'] for r in results.values()]
    labels = [f"{name}\n{result['trainable_parameters']:,} parameters"
              for name, result in results.items()]
    bars = axes[2].bar(labels, values, color=['tab:blue', 'tab:orange', 'tab:green'])
    axes[2].bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=4)
    axes[2].set(ylabel='Accuracy (%)', title='Final training accuracy', ylim=(0, 110))
    fig.suptitle(f'Axonal / synaptic / hybrid — {config.delay_pathway} delays only | {config.task_type}, '
                 f'{config.num_samples} samples, topology {config.input_size} → '
                 + ' → '.join(map(str, config.hidden_layers + [config.output_size])))
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'delay_comparison.{extension}', dpi=180)
    plt.close(fig)
    print(f'Comparison saved: {out}', flush=True)


if __name__ == '__main__':
    main()
