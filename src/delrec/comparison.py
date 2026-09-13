"""Matched initialization for six single-pathway delay networks.

The MEM comparison enables recurrence in every hidden layer. Other benchmarks
can retain their config's topology with ``force_recurrence=False``.
"""

from copy import deepcopy

import torch
from delrec import networks
from delrec.delay_layers import axonal_recdel
from delrec.networks import dcls_module, learned_delay_parameter
from delrec.utils import reset_states, seed_everything


def matched_models(config, pathway="recurrent", *, force_recurrence=True):
    """Match weights and initial functions; synaptic delays subsequently untie."""
    suffixes = {'recurrent': 'recurrent_only_delays',
                'feedforward': 'feedforward_only_delays'}
    if pathway not in suffixes:
        raise ValueError('pathway must be recurrent or feedforward')
    suffix = suffixes[pathway]
    config = deepcopy(config)
    if pathway == 'recurrent' and force_recurrence:
        # Existing recurrent classes honor this flag; the comparison needs
        # recurrence in every hidden layer, including a single-hidden-layer run.
        config.no_recurrence_in_last_layer = False
    ax_config, sy_config = deepcopy(config), deepcopy(config)
    ax_config.delay_pathway = sy_config.delay_pathway = pathway
    ax_config.model = f'SNN_axonal_{suffix}'
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
            for module in model.modules():
                if isinstance(module, axonal_recdel):
                    module.update_sigma(0)
                elif isinstance(module, dcls_module) and cfg.DCLSversion == 'gauss':
                    module.SIG.fill_(max(0.23, float(cfg.siginit)))
            model.eval()
        generator = torch.Generator().manual_seed(123)
        probe = torch.rand(config.time_window, 2, config.input_size, generator=generator)
        torch.testing.assert_close(ax(probe), sy(probe), atol=1e-5, rtol=1e-5)
        reset_states(ax)
        reset_states(sy)
    hy_config = deepcopy(config)
    hy_config.delay_pathway = pathway
    hy_config.model = f'SNN_hybrid_{suffix}'
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


def six_models(config, *, force_recurrence=True):
    """Delay location × delay type; match connection weights across locations."""
    recurrent = matched_models(config, pathway='recurrent', force_recurrence=force_recurrence)
    feedforward = matched_models(config, pathway='feedforward')
    models = []
    for kind, (rc, rec), (fc, ff) in zip(
            ('Axonal', 'Synaptic', 'Hybrid'), recurrent, feedforward, strict=True):
        # All projections have the same connectivity, but their delay operators
        # differ. Match connection weights/biases without copying delay tensors.
        rec_projections = [m for m in rec.layers if isinstance(m, torch.nn.Linear)]
        ff_projections = [m for m in ff.layers if isinstance(m, torch.nn.Linear)
                          or (isinstance(m, dcls_module) and m.weight.requires_grad)]
        with torch.no_grad():
            for source, target in zip(rec_projections, ff_projections, strict=True):
                target.weight.copy_(source.weight.unsqueeze(-1)
                                    if isinstance(target, dcls_module) else source.weight)
                if target.bias is not None:
                    target.bias.copy_(source.bias)
        assert not any(isinstance(m, dcls_module) for m in rec.modules())
        assert not any(isinstance(m, axonal_recdel) for m in ff.modules())
        models.extend([(f'Recurrent {kind}', rc, rec),
                       (f'Feedforward {kind}', fc, ff)])
    return models

