"""Six-model memorization comparison: delay location (recurrent/feedforward) × axonal/synaptic/hybrid.

Run: .venv/bin/python experiments/compare_mem_snn_rsnn.py
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

from train_mem import ROOT, Config, run, torch, plt
from delrec.delay_layers import axonal_recdel
from delrec.comparison import six_models
from delrec.networks import dcls_module, learned_delay_parameter


def plot_comparison(histories, results, config, out):
    fig = plt.figure(figsize=(16, 9), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    loss_ax, acc_ax, bar_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    colors = {'Axonal': 'tab:blue', 'Synaptic': 'tab:orange', 'Hybrid': 'tab:green'}
    for label, history in histories.items():
        family, kind = label.split()
        style = '-' if family == 'Recurrent' else '--'
        epochs = [r['epoch'] for r in history]
        loss_ax.plot(epochs, [r['loss'] for r in history], color=colors[kind], linestyle=style, label=label)
        acc_ax.plot(epochs, [r['accuracy_percent'] for r in history], color=colors[kind], linestyle=style, label=label)
    loss_ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training loss')
    acc_ax.set(xlabel='Epoch', ylabel='Accuracy (%)', title='Training accuracy', ylim=(0, 105))
    for axis in (loss_ax, acc_ax):
        axis.legend(fontsize=9, ncol=2)
        axis.grid(alpha=0.25)
    labels = [f"{label.replace(' ', chr(10), 1)}\n{r['trainable_parameters']:,} parameters" for label, r in results.items()]
    bars = bar_ax.bar(labels, [r['accuracy_percent'] for r in results.values()],
                      color=[colors[label.split()[1]] for label in results])
    for bar, label in zip(bars, results):
        if label.startswith('Feedforward '):
            bar.set_hatch('//')
    bar_ax.bar_label(bars, labels=[f"{r['accuracy_percent']:.2f}%" for r in results.values()], padding=4)
    bar_ax.set(ylabel='Accuracy (%)', title='Final training accuracy', ylim=(0, 110))
    topology = ' → '.join(map(str, [config.input_size] + config.hidden_layers + [config.output_size]))
    fig.suptitle(f'Recurrent-only vs feedforward-only delays | {config.task_type}, {config.num_samples} samples, topology {topology}\n'
                 'Solid: recurrent delays only · Dashed: feedforward delays only')
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'delay_location_comparison.{extension}', dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    integer_options = ('epochs', 'seed', 'dataset-seed', 'num-samples',
                       'hybrid-max-synaptic-delay', 'hybrid-delay-seed')
    for name in integer_options:
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--task-type', choices=['temporal', 'spatial'], default = 'temporal')
    parser.add_argument('--hidden-layers')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    config = Config()
    for key in [s.replace('-', '_') for s in integer_options] + ['task_type']:
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    if args.hidden_layers:
        config.hidden_layers = [int(n) for n in args.hidden_layers.split(',')]
    if min(config.epochs, config.num_samples, *config.hidden_layers) < 1:
        parser.error('Epochs, samples and layer widths must be positive')
    torch.set_num_threads(config.cpu_threads)
    out = args.out or ROOT / 'exp' / 'MEM' / 'delay_location_comparison' / (
        f'{config.task_type}_seed{config.seed}_{datetime.now():%Y-%m-%d-%H-%M-%S-%f}')
    models = six_models(config)
    histories, results = {}, {}
    reference_data = None
    for label, cfg, model in models:
        directory = out / label.lower().replace(' ', '_')
        history, final, _ = run(cfg, torch.device(args.device), directory, model=model)
        final['feedforward_delay_parameters'] = sum(learned_delay_parameter(m, 'P').numel() for m in model.layers if isinstance(m, dcls_module))
        final['recurrent_delay_parameters'] = sum(learned_delay_parameter(m, 'recurrent_delays').numel() for m in model.layers if isinstance(m, axonal_recdel))
        final['fixed_synaptic_offsets'] = sum(b.numel() for n, b in model.named_buffers() if n.endswith('.offsets'))
        results[label], histories[label] = final, history
        data = torch.load(directory / 'dataset.pt', weights_only=True)
        if reference_data is None:
            reference_data = data
        else:
            assert all(torch.equal(data[k], reference_data[k]) for k in data)
    (out / 'comparison.json').write_text(json.dumps({
        'initialization': 'Matched connection weights and biases across delay locations. Recurrent models have only linear FF projections; feedforward models have no recurrent connections. Within each location, delay initialization is matched across types, with added fixed offsets for hybrid. Same dataset and batch order for all six models.',
        'results': results,
    }, indent=2))
    plot_comparison(histories, results, config, out)
    print(f'Six-model comparison saved: {out}', flush=True)


if __name__ == '__main__':
    main()
