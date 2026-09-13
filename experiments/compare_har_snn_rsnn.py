"""HAR: axonal, synaptic and hybrid delays on feedforward/recurrent connections.

No validation set: report each run's highest test accuracy over training epochs.

python experiments/compare_har_snn_rsnn.py --device cuda
python experiments/compare_har_snn_rsnn.py --device cpu --smoke
"""

import argparse
from copy import deepcopy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'exp' / '.matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.perf_HAR import Config
from delrec.comparison import six_models
from delrec.datasets import load_dataset
from delrec.delay_layers import axonal_recdel
from delrec.networks import dcls_module, learned_delay_parameter
from delrec.training import har
from delrec.utils import reset_states, seed_everything
from train import _FewBatches, apply_kernels, effective_config


def parameter_counts(model):
    """Count optimized parameters separately from fixed filters/widths/offsets."""
    counts = {
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'feedforward_delay_parameters': sum(learned_delay_parameter(m, 'P').numel()
            for m in model.modules() if isinstance(m, dcls_module)),
        'recurrent_delay_parameters': sum(learned_delay_parameter(m, 'recurrent_delays').numel()
            for m in model.modules() if isinstance(m, axonal_recdel)),
        'fixed_synaptic_offsets': sum(b.numel() for n, b in model.named_buffers()
                                     if n.endswith('.offsets')),
    }
    counts['weight_bias_parameters'] = (counts['trainable_parameters']
        - counts['feedforward_delay_parameters'] - counts['recurrent_delay_parameters'])
    return counts


SELECTION_PROTOCOL = 'best_test_accuracy_over_epochs'


def load_splits(config, out):
    # HAR has no validation set: use the full cached train/test partitions.
    original_train, _, original_test = load_dataset(config)
    datasets = (original_train.dataset, original_test.dataset)
    (out / 'split.json').write_text(json.dumps({
        'datasets_path': str(config.datasets_path),
        'train_samples': len(datasets[0]), 'test_samples': len(datasets[1]),
        'selection_protocol': SELECTION_PROTOCOL,
        'protocol': 'No validation set. Train on all cached training windows; evaluate cached test windows each epoch and report the highest test accuracy.',
    }, indent=2))
    return datasets


def make_loaders(datasets, config, device, smoke):
    # Separate RNG keeps batch order independent of each model's dropout draws.
    loaders = tuple(DataLoader(dataset, batch_size=config.batch_size, shuffle=(i == 0),
        generator=torch.Generator().manual_seed(config.seed), num_workers=config.num_workers,
        pin_memory=device.type == 'cuda') for i, dataset in enumerate(datasets))
    return tuple(_FewBatches(loader, 2) for loader in loaders) if smoke else loaders


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_model(label, config, model, datasets, device, args, directory):
    directory.mkdir(parents=True, exist_ok=False)
    config.results_dir = str(directory)
    seed_everything(config.seed, is_cuda=device.type == 'cuda')
    loaders = make_loaders(datasets, config, device, args.smoke)
    model.to(device)
    # Explicitly opt hybrid into the synaptic accelerator when requested; its
    # shared base is a leaf parameter and receives the summed synaptic gradient.
    apply_kernels(model, args.kernel, args.synaptic_kernel)
    optimizers, schedulers = har.init_optim_sche(model, config)
    counts = parameter_counts(model)
    print(f'\n{label}, seed {config.seed}: {counts["trainable_parameters"]:,} trainable '
          f'parameters ({counts["fixed_synaptic_offsets"]:,} fixed offsets)', flush=True)
    (directory / 'config.json').write_text(json.dumps({**effective_config(config),
        'device': str(device), 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'kernel': args.kernel, 'synaptic_kernel': args.synaptic_kernel, 'smoke': args.smoke,
        'selection_protocol': SELECTION_PROTOCOL,
    }, indent=2, default=str))
    (directory / 'parameters.json').write_text(json.dumps(counts, indent=2))
    history, best_acc = [], -float('inf')
    for epoch in range(config.epochs):
        for module in model.modules():
            if isinstance(module, axonal_recdel):
                module.update_sigma(epoch)
        train_acc, train_loss = har.train(loaders[0], model, optimizers, epoch, device, config)
        test_acc, test_loss = har.test(loaders[1], model, epoch, device, config)
        for scheduler in schedulers:
            scheduler.step()
        if not np.isfinite([train_loss, test_loss, train_acc, test_acc]).all():
            raise RuntimeError(f'Non-finite metrics for {label}, epoch {epoch + 1}')
        history.append(dict(epoch=epoch + 1, train_loss=train_loss,
            train_accuracy_percent=train_acc, test_loss=test_loss,
            test_accuracy_percent=test_acc))
        write_csv(directory / 'history.csv', history)
        reset_states(model)
        checkpoint = dict(epoch=epoch + 1, test_accuracy_percent=test_acc, test_loss=test_loss,
            selection_protocol=SELECTION_PROTOCOL,
            model=model.state_dict(), optim=[o.state_dict() for o in optimizers],
            sched=[s.state_dict() for s in schedulers],
            recurrent_sigma={name: m.sigma for name, m in model.named_modules()
                             if isinstance(m, axonal_recdel)})
        torch.save(checkpoint, directory / 'last.pth')
        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(checkpoint, directory / 'best.pth')
        print(f'{label} [{epoch + 1}/{config.epochs}]: test {test_acc:.2f}%, '
              f'best {best_acc:.2f}%', flush=True)
    del checkpoint
    # Report exactly the epoch measurement used for checkpoint selection. A
    # second evaluation with changed smoothing/rounding could produce a different
    # number from the maximum over epochs. Latest epoch wins accuracy ties.
    best = max(history, key=lambda row: (row['test_accuracy_percent'], row['epoch']))
    result = dict(label=label, model=config.model, seed=config.seed, smoke=args.smoke,
        selection_protocol=SELECTION_PROTOCOL,
        test_accuracy_percent=best['test_accuracy_percent'], test_loss=best['test_loss'],
        best_epoch=best['epoch'], **counts)
    (directory / 'final_test.json').write_text(json.dumps(result, indent=2))
    reset_states(model)
    model.to('cpu')
    return history, result


def plot_comparison(histories, results, out, smoke=False):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout='constrained')
    colors = {'Axonal': 'tab:blue', 'Synaptic': 'tab:orange', 'Hybrid': 'tab:green'}
    labels = list(histories)
    for label, runs in histories.items():
        family, kind = label.split()
        for axis, metric in zip(axes.flat[:3],
                ('train_loss', 'test_loss', 'test_accuracy_percent')):
            values = np.array([[row[metric] for row in run] for run in runs])
            epochs = [row['epoch'] for row in runs[0]]
            mean, std = values.mean(0), values.std(0)
            axis.plot(epochs, mean, label=label, color=colors[kind],
                      linestyle='-' if family == 'Recurrent' else '--')
            if len(runs) > 1:
                axis.fill_between(epochs, mean - std, mean + std, color=colors[kind], alpha=0.12)
    for axis, title, ylabel in zip(axes.flat[:3],
            ('Training loss', 'Test loss', 'Test accuracy'),
            ('Cross-entropy', 'Cross-entropy', 'Accuracy (%)')):
        axis.set(xlabel='Epoch', ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    means, stds, ticks = [], [], []
    for label in labels:
        runs = [r for r in results if r['label'] == label]
        values = [r['test_accuracy_percent'] for r in runs]
        means.append(np.mean(values))
        stds.append(np.std(values))
        ticks.append(f'{label.replace(" ", chr(10))}\n{runs[0]["trainable_parameters"]:,} params')
    bars = axes[1, 1].bar(ticks, means, yerr=stds, capsize=3,
                          color=[colors[label.split()[1]] for label in labels])
    for bar, label in zip(bars, labels):
        if label.startswith('Feedforward'):
            bar.set_hatch('//')
    axes[1, 1].bar_label(bars, labels=[f'{v:.2f}%' for v in means], padding=5)
    axes[1, 1].set(title='Highest test accuracy over training epochs', ylabel='Accuracy (%)', ylim=(0, 110))
    axes[1, 1].tick_params(axis='x', labelsize=8)
    fig.suptitle(('SMOKE TEST — partial data, not benchmark results\n' if smoke else '')
        + 'HAR: feedforward vs recurrent delays | solid: recurrent, dashed/hatched: feedforward\n'
          'Curves and bars: mean across seeds; shading/error bars: population standard deviation')
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'har_comparison.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seeds', default='0', help='Comma-separated initialization/batch-order seeds')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--hidden-layers', help='Comma-separated widths; default: 128,176,176')
    parser.add_argument('--datasets-path', type=Path, default=ROOT / 'Datasets' / 'HAR')
    parser.add_argument('--hybrid-max-synaptic-delay', type=int, default=4)
    parser.add_argument('--hybrid-delay-seed', type=int, default=123)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--kernel', choices=['v2', 'triton_exact'], default='v2',
                        help='Recurrent axonal scan; v2 works on CUDA and CPU')
    parser.add_argument('--synaptic-kernel', choices=['v2', 'eventdriven'], default='v2',
                        help='Recurrent synaptic/hybrid scan; eventdriven requires Triton/CUDA')
    parser.add_argument('--smoke', action='store_true', help='Two epochs, two batches per split, small hidden layers')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    config = Config()
    for key in ('epochs', 'batch_size', 'datasets_path', 'hybrid_max_synaptic_delay', 'hybrid_delay_seed'):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    try:
        seeds = [int(s) for s in args.seeds.split(',')]
        config.hidden_layers = ([int(n) for n in args.hidden_layers.split(',')]
                                if args.hidden_layers else list(config.hidden_layers))
    except ValueError:
        parser.error('Seeds and hidden layers must be comma-separated integers')
    if len(config.hidden_layers) < 2:
        parser.error('HAR needs at least two hidden layers: the last hidden layer has no recurrence')
    if min(config.epochs, config.batch_size, args.cpu_threads, *config.hidden_layers) < 1:
        parser.error('Epochs, batch size, thread count and hidden widths must be positive')
    if len(seeds) != len(set(seeds)) or min(seeds + [args.hybrid_delay_seed]) < 0:
        parser.error('Seeds must be nonnegative and run seeds must be unique')
    if args.hybrid_max_synaptic_delay < 0:
        parser.error('Maximum hybrid offset must be nonnegative')
    if args.smoke:
        config.epochs = 2
        if args.batch_size is None:
            config.batch_size = 4
        if args.hidden_layers is None:
            config.hidden_layers = [8, 8, 8]
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable. See docs/har_comparison.md for the RTX 5060 Ti setup.')
        if torch.cuda.get_device_capability()[0] >= 12 and tuple(map(int, (torch.version.cuda or '0.0').split('.')[:2])) < (12, 8):
            parser.error('Blackwell requires a newer CUDA build; see docs/har_comparison.md.')
        # Fail before loading data if the installed wheel cannot execute on this GPU.
        torch.ones(1, device=device).add_(1)
        torch.cuda.synchronize()
        print(f'CUDA: {torch.cuda.get_device_name()} | torch {torch.__version__}', flush=True)
    elif args.kernel != 'v2' or args.synaptic_kernel != 'v2':
        parser.error('Accelerated recurrent kernels require --device cuda')
    torch.set_num_threads(args.cpu_threads)
    out = args.out or ROOT / 'exp' / 'HAR' / 'delay_location_comparison' / f'{datetime.now():%Y-%m-%d-%H-%M-%S-%f}'
    out.mkdir(parents=True, exist_ok=False)
    datasets = load_splits(config, out)
    histories, results = {}, []
    for seed in seeds:
        config.seed = seed
        # HAR keeps its last hidden layer non-recurrent, as in perf_HAR.py.
        models = six_models(deepcopy(config), force_recurrence=False)
        while models:
            label, cfg, model = models.pop(0)
            directory = out / label.lower().replace(' ', '_') / f'seed{seed}'
            history, result = run_model(label, cfg, model, datasets, device, args, directory)
            histories.setdefault(label, []).append(history)
            results.append(result)
            write_csv(out / 'comparison.csv', results)
            (out / 'comparison.json').write_text(json.dumps({
                'smoke': args.smoke, 'seeds': seeds, 'results': results, 'histories': histories,
                'selection_protocol': SELECTION_PROTOCOL,
                'initialization': 'Matched projection weights/biases across all six; matched recurrent weights and base delays within each pathway. Hybrid adds fixed synaptic offsets.',
            }, indent=2))
            plot_comparison(histories, results, out, args.smoke)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    print(f'\nComparison saved: {out}', flush=True)
    for label in histories:
        runs = [r for r in results if r['label'] == label]
        values = [r['test_accuracy_percent'] for r in runs]
        print(f'{label:20s} {runs[0]["trainable_parameters"]:>9,} parameters | '
              f'best test {np.mean(values):.2f} +/- {np.std(values):.2f}%')


if __name__ == '__main__':
    main()
