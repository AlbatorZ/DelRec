"""Markdown summaries of the resolved settings for MEM comparisons."""

from pathlib import Path

from delrec.delay_layers import axonal_recdel, vanilla_recurrent
from delrec.networks import dcls_module


_GROUPS = {
    'Data': 'dataset task_type num_samples input_size time_window output_size input_gain dataset_seed',
    'Training': 'epochs batch_size readout seed grad_clip weight_decay',
    'Learning rates (configured)': 'lr_w lr_positions lr_w_recurrent lr_positions_recurrent lr_w_feedforward lr_positions_feedforward',
    'Architecture': 'hidden_layers bias use_batch_norm feedforward_dropout_rate recurrent_dropout_rate no_delay_in_first_layer no_delay_in_last_layer no_recurrence_in_last_layer no_recurrent_delays',
    'Neuron': 'neuron_module surrogate_function tau decay_input v_threshold v_reset detach_reset step_mode backend store_v_seq',
    'Initialization': 'init_ff_weights init_dcls_weights init_rec_weights init_rec_delay rec_delay_init_gain init_recdel_offset delay_std_init',
    'Recurrent delays': 'max_rec_delay use_rec_bias use_sig_p sigma_init sigma_decay round_delays',
    'Feedforward delays': 'DCLSversion kernel_count max_feedforward_delay left_padding right_padding init_pos_a init_pos_b siginit round_pos_each_epoch',
    'Hybrid delays': 'hybrid_max_synaptic_delay hybrid_delay_seed',
    'Diagnostics': 'delay_diagnostics_every',
    'Execution': 'cpu_threads num_workers',
}


def _cell(value):
    if isinstance(value, type):
        value = f'{value.__module__}.{value.__qualname__}'
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', '<br>')


def save_training_recap(models, device, out):
    """Save and return Markdown before training; models are (label, config, model).

    Read the built models' configs so matching-time overrides are included.
    Report effective family learning rates separately from configured fallbacks.
    This function does not create optimizers or mutate model/config state.
    """
    models = list(models)
    if not models:
        raise ValueError('A training recap requires at least one model')
    categories = {key: group for group, keys in _GROUPS.items() for key in keys.split()}
    keys = list(categories)
    extras = sorted({key for _, cfg, _ in models for key in dir(cfg)
                     if not key.startswith('_') and key != 'model'
                     and not callable(getattr(cfg, key))} - set(keys))
    lines = ['# Training parameters', '',
             'Settings captured from the built models before training. Values shared by all models '
             'appear once; differing values are labeled by model.', '',
             '| Category | Parameter | Value |', '| --- | --- | --- |']
    for category, key, value in (
        ('Execution', 'device', device), ('Execution', 'output directory', Path(out)),
        ('Training', 'optimizer', 'AdamW'),
        ('Training', 'learning-rate scheduler', 'CosineAnnealingLR; T_max = epochs; eta_min = 0'),
        ('Training', 'objective', 'Cross-entropy; training-set-only loss and accuracy'),
        ('Training', 'batches', 'Shuffled training; deterministic measurement order'),
        ('Diagnostics', 'snapshot epochs', 'Initial (0), every delay_diagnostics_every epochs, and final'),
    ):
        lines.append(f'| {category} | {_cell(key)} | {_cell(value)} |')
    for key in keys + extras:
        present = [(label, _cell(getattr(cfg, key, '—'))) for label, cfg, _ in models]
        if all(not hasattr(cfg, key) for _, cfg, _ in models):
            continue
        value = present[0][1] if len({v for _, v in present}) == 1 else '<br>'.join(
            f'{_cell(label)}: {v}' for label, v in present)
        lines.append(f'| {categories.get(key, "Other")} | {key} | {value} |')
    lines += ['', '## Models and effective initial learning rates', '',
              'Family rates below override the configured `lr_w` / `lr_positions` fallbacks. '
              'Cosine scheduling subsequently changes these rates.', '',
              '| Model | Network class | Weight LR | Delay LR | Trainable parameters |',
              '| --- | --- | --- | --- | --- |']
    for label, cfg, model in models:
        family = 'recurrent' if any(isinstance(m, (axonal_recdel, vanilla_recurrent))
                                    for m in model.modules()) else 'feedforward'
        lr_w = getattr(cfg, f'lr_w_{family}', cfg.lr_w)
        lr_p = getattr(cfg, f'lr_positions_{family}', cfg.lr_positions)
        # DCLS SIG is frozen by make_optimizer before training.
        frozen_ids = {id(m.SIG) for m in model.modules()
                      if isinstance(m, dcls_module) and cfg.DCLSversion == 'gauss'}
        count = sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in frozen_ids)
        lines.append(f'| {_cell(label)} | {_cell(type(model).__name__)} | {lr_w} | {lr_p} | {count:,} |')
    markdown = '\n'.join(lines) + '\n'
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'training_parameters.md').write_text(markdown, encoding='utf-8')
    return markdown
