"""Axonal/synaptic/hybrid comparison on one selected delay pathway; all measurements use training data.

Run: .venv/bin/python experiments/compare_mem_delays.py
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

from train_mem import ROOT, Config, run, torch, plt
from delrec.delay_layers import axonal_recdel
from delrec.networks import dcls_module, learned_delay_parameter
from delrec.comparison import matched_models


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
