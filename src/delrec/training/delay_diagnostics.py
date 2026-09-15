"""Per-layer delay snapshots and per-minibatch optimizer diagnostics for MEM runs.

Updates pool every parameter and minibatch in a selected epoch (not net epoch
movement). Raw gradients are captured before clipping, without LR scaling.
Actual updates include AdamW and delay clamping. Epoch zero has no updates.
"""

from pathlib import Path
import json

import numpy as np
import torch

from delrec.delay_layers import axonal_recdel
from delrec.networks import dcls_module, learned_delay_parameter


class DelayDiagnostics:
    def __init__(self, model, directory, every=10, show=False):
        if not isinstance(every, int) or isinstance(every, bool) or every < 1:
            raise ValueError('delay_diagnostics_every must be a positive integer')
        self.model = model
        self.directory = Path(directory) / 'delay_diagnostics'
        self.every = every
        self.show = show
        self.layers = []
        for name, module in model.named_modules():
            attribute = ('recurrent_delays' if isinstance(module, axonal_recdel)
                         else 'P' if isinstance(module, dcls_module) else None)
            if attribute:
                self.layers.append((name, module, attribute,
                                    learned_delay_parameter(module, attribute)))
        self.active = False
        self.begin_epoch(0, 0)

    def begin_epoch(self, epoch, final_epoch):
        self.active = epoch == 0 or epoch == final_epoch or epoch % self.every == 0
        self.epoch = epoch
        self.updates = {name: {'gradient': [], 'delta': [], 'missing_grad_steps': 0}
                        for name, *_ in self.layers}

    @torch.no_grad()
    def before_step(self):
        """Capture raw gradients and parameters after backward, before clipping."""
        if not self.active:
            return
        self.before = {}
        for name, _, _, parameter in self.layers:
            self.before[name] = parameter.detach().clone()
            record = self.updates[name]
            if parameter.grad is None:
                record['missing_grad_steps'] += 1
            else:
                record['gradient'].append(self._array(parameter.grad))

    @torch.no_grad()
    def after_step(self):
        if self.active:
            for name, _, _, parameter in self.layers:
                self.updates[name]['delta'].append(self._array(parameter - self.before[name]))
            self.before = {}

    @staticmethod
    def _array(tensor):
        return tensor.detach().float().cpu().numpy().reshape(-1).copy()

    @torch.no_grad()
    def snapshot(self):
        if not self.active or not self.layers:
            return
        import matplotlib.pyplot as plt

        self.directory.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(self.layers), 4, figsize=(20, 3.5 * len(self.layers)),
                                 squeeze=False, layout='constrained')
        arrays, summary = {}, {}
        for row, (name, module, attribute, parameter) in zip(axes, self.layers):
            effective = getattr(module, attribute)
            # DCLS is a cross-correlation: lag = left padding - kernel index.
            if attribute == 'P':
                effective = module.left_padding - (module.dilated_kernel_size[0] - 1) / 2 - effective
            record = self.updates[name]
            values = {'parameter': self._array(parameter), 'effective_delay': self._array(effective)}
            for key in ('gradient', 'delta'):
                values[key] = np.concatenate(record[key]) if record[key] else np.array([], dtype=np.float32)
            summary[name] = {'missing_grad_steps': record['missing_grad_steps']}
            titles = ('Learned P (DCLS position)' if attribute == 'P' else 'Learned recurrent delay',
                      'Effective delay (timesteps)', '|Gradient| (before clipping)',
                      '|Actual step| (after clamping)')
            for ax, (key, value), title in zip(row, values.items(), titles):
                arrays[f'{name}/{key}'] = value
                finite = value[np.isfinite(value)]
                stats = {'count': int(value.size), 'nonfinite_count': int(value.size - finite.size)}
                if finite.size:
                    stats.update(mean=float(finite.mean()), std=float(finite.std()),
                                 mean_abs=float(np.abs(finite).mean()), max_abs=float(np.abs(finite).max()),
                                 zero_fraction=float((finite == 0).mean()))
                    plotted = np.abs(finite) if key in ('gradient', 'delta') else finite
                    bins = 50 if key in ('gradient', 'delta') else int(np.ceil(np.sqrt(plotted.size)))
                    ax.hist(plotted, bins=bins)
                    if key in ('gradient', 'delta'):
                        ax.set_yscale('log')
                    ax.text(.98, .97, f"zero: {stats['zero_fraction']:.1%}\nmean |x|: {stats['mean_abs']:.3g}",
                            transform=ax.transAxes, ha='right', va='top', fontsize=8)
                else:
                    ax.text(.5, .5, 'No updates yet' if self.epoch == 0 else 'No gradient samples',
                            transform=ax.transAxes, ha='center')
                summary[name][key] = stats
                ax.set(title=title, xlabel='Value', ylabel=f'{name}\nCount')
                ax.grid(alpha=.2)
        phase = 'initial' if self.epoch == 0 else f'epoch {self.epoch}'
        fig.suptitle(f'{type(self.model).__name__} — {phase}\nUpdates pooled across minibatches of this epoch')
        stem = self.directory / f'epoch_{self.epoch:05d}'
        np.savez_compressed(stem.with_suffix('.npz'), **arrays)
        stem.with_suffix('.json').write_text(json.dumps(summary, indent=2))
        fig.savefig(stem.with_suffix('.png'), dpi=150)
        if self.show:
            plt.show()
        plt.close(fig)
        # Release pooled arrays as soon as they have been saved.
        self.updates = {}
