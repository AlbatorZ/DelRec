"""Per-layer delay snapshots and per-minibatch optimizer diagnostics for MEM runs.

Updates pool every parameter and minibatch in a selected epoch (not net epoch
movement). Raw gradients are captured before clipping, without LR scaling.
Actual updates include AdamW and delay clamping. Epoch zero has no updates.
"""

from pathlib import Path
import json

import numpy as np
import torch
from torch.nn.utils import parametrize

from delrec.delay_layers import axonal_recdel
from delrec.networks import dcls_module, learned_delay_parameter, spike_registrator


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
        # A feedforward delay belongs to its source neurons: the first stage
        # delays inputs, while the final projection delays the last hidden layer.
        self.hidden_delays = []
        hidden_index = 0
        for name, module in getattr(model, 'layers', torch.nn.Sequential()).named_children():
            if isinstance(module, spike_registrator):
                hidden_index += 1
            elif isinstance(module, axonal_recdel):
                self.hidden_delays.append((f'Hidden {hidden_index + 1} recurrent', module))
            elif isinstance(module, dcls_module) and hidden_index > 0:
                self.hidden_delays.append((f'Hidden {hidden_index} feedforward', module))
        self.delay_epochs = []
        self.delay_history = []
        self.active = False
        self.begin_epoch(0, 0)

    def begin_epoch(self, epoch, final_epoch, capture_all=False):
        self.active = epoch == 0 or epoch == final_epoch or epoch % self.every == 0
        self.capture = self.active or capture_all
        self.epoch = epoch
        self.updates = {name: {'gradient': [], 'delta': [], 'missing_grad_steps': 0}
                        for name, *_ in self.layers}

    @torch.no_grad()
    def before_step(self):
        """Capture raw gradients and parameters after backward, before clipping."""
        if not self.capture:
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
        if self.capture:
            for name, _, _, parameter in self.layers:
                self.updates[name]['delta'].append(self._array(parameter - self.before[name]))
            self.before = {}

    @staticmethod
    def _array(tensor):
        return tensor.detach().float().cpu().numpy().reshape(-1).copy()

    @torch.no_grad()
    def snapshot(self, force=False):
        self.record_hidden_delays()
        if not (self.active or force) or not self.layers:
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


    @torch.no_grad()
    def record_hidden_delays(self):
        """Hybrid axonal delay only; synaptic delays averaged over targets/kernels."""
        if not self.hidden_delays:
            return
        values = []
        for _, module in self.hidden_delays:
            if isinstance(module, axonal_recdel):
                delay = learned_delay_parameter(module, 'recurrent_delays')
                if delay.ndim == 2:
                    delay = delay.mean(dim=0)  # (target, source)
            else:
                position = learned_delay_parameter(module, 'P')
                if parametrize.is_parametrized(module, 'P'):
                    # Keep the recentering for the widened hybrid kernel, but
                    # omit fixed offsets: lag then equals the base axonal lag.
                    position = position + module.parametrizations.P[0].position_shift
                delay = module.left_padding - (module.dilated_kernel_size[0] - 1) / 2 - position
                if module.groups == module.in_channels == module.out_channels:
                    # Depthwise DCLS: (1, source, 1, kernel).
                    delay = delay.mean(dim=(0, 2, 3))
                else:
                    # Dense/hybrid DCLS: (1, target, source, kernel).
                    delay = delay.mean(dim=(0, 1, 3))
            values.append(self._array(delay))
        if self.delay_epochs and self.delay_epochs[-1] == self.epoch:
            self.delay_history[-1] = values
        else:
            self.delay_epochs.append(self.epoch)
            self.delay_history.append(values)

    def plot_hidden_delay_evolution(self):
        """Save one heatmap per model run, with a common color scale for all epochs."""
        if not self.delay_history:
            return
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        self.directory.mkdir(parents=True, exist_ok=True)
        blocks = [np.stack([epoch[i] for epoch in self.delay_history], axis=1)
                  for i in range(len(self.hidden_delays))]
        data = np.concatenate(blocks, axis=0)
        epochs = np.asarray(self.delay_epochs)
        fig, ax = plt.subplots(figsize=(12, max(4, 2 * len(blocks))), layout='constrained')
        mesh = ax.pcolormesh(np.r_[epochs - .5, epochs[-1] + .5],
                             np.arange(data.shape[0] + 1) - .5,
                             np.ma.masked_invalid(data), cmap='viridis', shading='flat')
        ax.invert_yaxis()
        ax.set(xlabel='Completed epoch (0 = initial)', ylabel='Hidden neuron index',
               title=f'{type(self.model).__name__}\nOutgoing delay per hidden neuron (hybrid: axonal only; synaptic: mean)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        offset = 0
        metadata = []
        for (label, _), block in zip(self.hidden_delays, blocks):
            metadata.append({'layer': label, 'row_start': offset, 'neuron_count': block.shape[0]})
            if len(blocks) > 1:
                ax.text(1.01, offset + (block.shape[0] - 1) / 2, label,
                        transform=ax.get_yaxis_transform(), va='center', fontsize=8)
                if offset:
                    ax.axhline(offset - .5, color='white', linewidth=1)
            offset += block.shape[0]
        fig.colorbar(mesh, ax=ax, label='Delay (timesteps)', pad=.2 if len(blocks) > 1 else .02)
        stem = self.directory / 'hidden_delay_evolution'
        fig.savefig(stem.with_suffix('.png'), dpi=180)
        np.savez_compressed(stem.with_suffix('.npz'), epochs=epochs, delays=data)
        stem.with_suffix('.json').write_text(json.dumps({
            'aggregation': 'Hybrid: learned axonal delay only, excluding fixed offsets; synaptic: mean outgoing delay over targets and kernels',
            'rows': metadata,
        }, indent=2))
        if self.show:
            plt.show()
        plt.close(fig)
