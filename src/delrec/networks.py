"""Network zoo for SSC, PS-MNIST, HAR, AL and Mackey-Glass.

A hidden block is Linear -> Dropout -> neuron -> spike_registrator -> [BatchNorm]. The network ends in a Linear plus a leaky-integrate layer with an infinite
threshold, so ``forward`` returns a membrane sequence rather than spikes.

The class hierarchy is the experimental matrix: ``SNN_axonal_recurrent_delays`` (axonal,
learned) is the base, and each sibling changes exactly one thing, the delay shape
(synaptic), whether the delays are trained (fixed), or whether they exist at all
(vanilla). ``spike_registrator`` is a pass-through that stashes spike trains for
the firing-rate penalty and the efficiency analyses.
"""

import torch
import math
from copy import copy

from spikingjelly.activation_based import layer
from DCLS.construct.modules import Dcls1d

from delrec.delay_layers import axonal_recdel, synaptic_recdel, common_recdel

class dcls_module(Dcls1d):
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        groups,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_count=config.kernel_count,
            groups=groups,
            dilated_kernel_size=config.max_feedforward_delay,
            bias=config.bias,
            version=config.DCLSversion,
            )
        
        self.config = config
        self.left_padding, self.right_padding = config.left_padding, config.right_padding

    def forward(self, x):
        assert x.dim() == 3 # (T, B, N)
        x = x.permute(1,2,0) # (batch, neurons, time)
        x = torch.nn.functional.pad(x, (self.left_padding, self.right_padding), 'constant', 0)
        # _conv_forward(self.P) is identical to super().forward().
        x = self._conv_forward(x, self.weight, self.bias, self.P, self.SIG)
        x = x.permute(2,0,1) # (time, batch, neurons)
        return x
        
class modified_batchnorm(layer.BatchNorm1d):
    def __init__(self, num_features, step_mode='m'):
        super().__init__(num_features, step_mode=step_mode)
        
    def forward(self, x):
        assert x.dim() == 3 # (T, B, N)
        return super().forward(x.unsqueeze(3)).squeeze(-1) # We apply batchnorm to shape (T, B, N, 1)

class spike_registrator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spikes = None

    def forward(self, x):
        assert x.dim() == 3 # (T, B, N)
        self.spikes = x.clone()
        return x

class SNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        assert config.dataset in ['SSC', 'PSMNIST', 'HAR', 'AL', 'MG', 'MEM'], "Unsupported SNN dataset."
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias)) # (T, B, N_in) -> (T, B, N_hidden)
            dim_buffer = layer_dim
            
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))
            
            layers.append(config.neuron_module(
                tau = config.tau,
                decay_input = config.decay_input,
                v_reset = config.v_reset,
                v_threshold = config.v_threshold,
                surrogate_function = config.surrogate_function,
                detach_reset = config.detach_reset,
                step_mode = config.step_mode,
                backend = config.backend,
                store_v_seq = config.store_v_seq,
                )
                          )
                
            layers.append(spike_registrator())
            
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
            
        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        
        self.layers = torch.nn.Sequential(*layers)
        
        self.init_weights()
        
    def forward(self, x): 
        assert x.dim() == 3 # (T, B, N)
        x = self.layers(x)
        return x
        
    def log_params(self):
        logs = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Linear):
                
                w = torch.abs(layer.weight).mean()
                w_grad_max = layer.weight.grad.abs().max().item() if layer.weight.grad is not None else 0.0
                logs.update({
                        f'w_linear_{idx}': w,
                        f'w_linear_grad_max_{idx}': w_grad_max,
                    })
                
        return logs
        
    def init_weights(self):
        for m in self.layers:
            if isinstance(m, torch.nn.Linear):
                if self.config.init_ff_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # For big models
                elif self.config.init_ff_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_ff_weights == 'default':
                    pass
    
class SNN_axonal_recurrent_delays(SNN):
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias)) # (T, B, N_in) -> (T, B, N_hidden)
            dim_buffer = layer_dim

            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))

            is_last_hidden = (idx == len(config.hidden_layers) - 1)
            if is_last_hidden and getattr(config, 'no_recurrence_in_last_layer', False):
                # plain (non-recurrent) LIF readout layer, matching neuroseqbench's HAR FFSNN
                layers.append(config.neuron_module(
                    tau = config.tau,
                    decay_input = config.decay_input,
                    v_reset = config.v_reset,
                    v_threshold = config.v_threshold,
                    surrogate_function = config.surrogate_function,
                    detach_reset = config.detach_reset,
                    step_mode = config.step_mode,
                    backend = config.backend,
                    store_v_seq = config.store_v_seq,
                    ))
            else:
                layers.append(axonal_recdel(config, layer_dim, config.neuron_module))

            layers.append(spike_registrator())

            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))

        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))

        self.layers = torch.nn.Sequential(*layers)

        self.init_weights()

    def clamp_delays(self):
        for m in self.layers:
            if isinstance(m, axonal_recdel):
                m.clamp_recurrent_delays()
                
    def round_pos(self):
        """Project the fractional recurrent delays onto the integer grid, in place."""
        stochastic = getattr(self.config, 'round_mode', 'nearest') == 'stochastic'
        with torch.no_grad():
            for m in self.layers:
                if isinstance(m, axonal_recdel):
                    d = m.recurrent_delays
                    if stochastic:
                        d.copy_(torch.floor(d + torch.rand_like(d)))
                    else:
                        d.round_()
                    m.clamp_recurrent_delays()

    def forward(self, x):
        return super().forward(x)
    
    def log_params(self):
        logs = super().log_params()
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, axonal_recdel):
                    logs[f'sigma_rec{idx}'] = layer.sigma
                    curr_pos_rec = layer.recurrent_delays.cpu().detach().numpy()
                    logs[f'pos_rec{idx}'] = curr_pos_rec.mean()
                    
                    
                    rec_w = layer.recurrent_weights
                    rec_w_mean = torch.abs(rec_w).mean()
                    rec_w_grad_max = rec_w.grad.abs().max().item() if rec_w.grad is not None else 0.0

                    logs.update({
                        f'recurrent_w_{idx}': rec_w_mean,
                        f'recurrent_w_grad_max_{idx}': rec_w_grad_max,
                    })
        
                    rec_d = layer.recurrent_delays
                    rec_d_grad_max = rec_d.grad.abs().max().item() if rec_d.grad is not None else 0.0

                    logs[f'recurrent_delay_grad_max_{idx}'] = rec_d_grad_max
                    
                    # if layer.use_sig_p:
                    #     logs[f"p_spread_mean_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().mean().item()
                    #     logs[f"p_spread_std_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().std().item()
                    
        return logs


class SNN_synaptic_recurrent_delays(SNN_axonal_recurrent_delays):
    """Identical to SNN_axonal_recurrent_delays but with per-synapse delays (synaptic_recdel,
    recurrent_delays shape (N, N)) instead of per-neuron axonal delays.
    """

    def __init__(self, config):
        # Skip SNN_axonal_recurrent_delays.__init__ (it hardcodes axonal_recdel). Rebuild
        # the same layer stack with synaptic_recdel at the one recurrent site.
        SNN.__init__(self, config)

        self.config = config

        layers = []
        dim_buffer = config.input_size

        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            dim_buffer = layer_dim

            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))

            is_last_hidden = (idx == len(config.hidden_layers) - 1)
            if is_last_hidden and getattr(config, 'no_recurrence_in_last_layer', False):
                layers.append(config.neuron_module(
                    tau = config.tau,
                    decay_input = config.decay_input,
                    v_reset = config.v_reset,
                    v_threshold = config.v_threshold,
                    surrogate_function = config.surrogate_function,
                    detach_reset = config.detach_reset,
                    step_mode = config.step_mode,
                    backend = config.backend,
                    store_v_seq = config.store_v_seq,
                    ))
            else:
                layers.append(synaptic_recdel(config, layer_dim, config.neuron_module))

            layers.append(spike_registrator())

            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))

        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))

        self.layers = torch.nn.Sequential(*layers)

        self.init_weights()


class SNN_common_recurrent_delays(SNN_axonal_recurrent_delays):
    """Identical to SNN_axonal_recurrent_delays but each recurrent layer learns a single shared delay
    (common_recdel, recurrent_delays shape (1,)) instead of per-neuron axonal delays.
    """

    def __init__(self, config):
        # Skip SNN_axonal_recurrent_delays.__init__ (it hardcodes axonal_recdel). Rebuild
        # the same layer stack with common_recdel at the one recurrent site.
        SNN.__init__(self, config)

        self.config = config

        layers = []
        dim_buffer = config.input_size

        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            dim_buffer = layer_dim

            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))

            is_last_hidden = (idx == len(config.hidden_layers) - 1)
            if is_last_hidden and getattr(config, 'no_recurrence_in_last_layer', False):
                layers.append(config.neuron_module(
                    tau = config.tau,
                    decay_input = config.decay_input,
                    v_reset = config.v_reset,
                    v_threshold = config.v_threshold,
                    surrogate_function = config.surrogate_function,
                    detach_reset = config.detach_reset,
                    step_mode = config.step_mode,
                    backend = config.backend,
                    store_v_seq = config.store_v_seq,
                    ))
            else:
                layers.append(common_recdel(config, layer_dim, config.neuron_module))

            layers.append(spike_registrator())

            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))

        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))

        self.layers = torch.nn.Sequential(*layers)

        self.init_weights()


class SNN_vanilla_recurrent(SNN_axonal_recurrent_delays):
    def __init__(self, config):
        super().__init__(config)

        for layer in self.layers:
            if isinstance(layer, axonal_recdel):
                with torch.no_grad():
                    layer.recurrent_delays.fill_(0.)  
                
                layer.recurrent_delays.requires_grad = False

                layer.sigma = 0.
                layer.config.sigma_init = 0.

                # Fixed delays don't learn a spread. Block p_spread even if
                # use_sig_p is set in the config.
                layer.use_sig_p = False
                if hasattr(layer, 'p_spread'):
                    del layer.p_spread

    def forward(self, x):
        return super().forward(x)

class SNN_fixed_recurrent_delays(SNN_axonal_recurrent_delays):
    def __init__(self, config):
        super().__init__(config)
        
        for layer in self.layers:
            if isinstance(layer, axonal_recdel):
                # Fixed delays are frozen at their init values. Round them to
                # integers so they match the round_pos eval regime with no epoch-0 jump.
                with torch.no_grad():
                    layer.recurrent_delays.round_()
                    layer.clamp_recurrent_delays()
                layer.recurrent_delays.requires_grad = False
                layer.sigma = 0.
                layer.config.sigma_init = 0.

                # Fixed delays don't learn a spread. Block p_spread even if
                # use_sig_p is set in the config.
                layer.use_sig_p = False
                if hasattr(layer, 'p_spread'):
                    del layer.p_spread

    def forward(self, x):
        return super().forward(x)

class SNN_fixed_synaptic_recurrent_delays(SNN_synaptic_recurrent_delays):
    """Per-synapse analogue of SNN_fixed_recurrent_delays: builds the synaptic
    delay stack, then freezes the (N, N) recurrent_delays at their rounded init
    values with sigma=0 (no learning, no spread)."""
    def __init__(self, config):
        super().__init__(config)

        for layer in self.layers:
            if isinstance(layer, axonal_recdel):   # covers synaptic_recdel
                with torch.no_grad():
                    layer.recurrent_delays.round_()
                    layer.clamp_recurrent_delays()
                layer.recurrent_delays.requires_grad = False
                layer.sigma = 0.
                layer.config.sigma_init = 0.

                layer.use_sig_p = False
                if hasattr(layer, 'p_spread'):
                    del layer.p_spread

    def forward(self, x):
        return super().forward(x)

class SNN_synaptic_feedforward_delays(SNN):
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            
            if config.no_delay_in_first_layer and idx == 0:
                layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            else:
                layers.append(
                    dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = layer_dim,
                        groups = 1,
                    )
                    ) # (T, B, N_in) -> (T, B, N_hidden)
            dim_buffer = layer_dim
            
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))
                
            layers.append(config.neuron_module(
            tau = config.tau,
            decay_input = config.decay_input,
            v_reset = config.v_reset,
            v_threshold = config.v_threshold,
            surrogate_function = config.surrogate_function,
            detach_reset = config.detach_reset,
            step_mode = config.step_mode,
            backend = config.backend,
            store_v_seq = config.store_v_seq,
            )
                            )
            
            layers.append(spike_registrator())
            
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
                
        if config.no_delay_in_last_layer:
            layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        else:
                layers.append(dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = config.output_size,
                        groups = 1,
                    ))
            
        self.layers = torch.nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.layers:
            # Feedforward weights init
            if isinstance(m, torch.nn.Linear):
                if self.config.init_ff_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # For big models
                elif self.config.init_ff_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_ff_weights == 'default':
                    pass
                
            if isinstance(m, dcls_module):
                if self.config.init_dcls_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # For big models
                elif self.config.init_dcls_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_dcls_weights == 'default':
                    torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # same as nn.Linear default
                
            # Feedforward delays init
            if isinstance(m, dcls_module):
                torch.nn.init.uniform_(m.P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                m.clamp_parameters()
                
    def clamp_delays(self, train=True):
        for m in self.layers:
            if isinstance(m, dcls_module):
                m.clamp_parameters()
                    
    def round_pos(self):
        with torch.no_grad():
            for m in self.layers:
                if isinstance(m, dcls_module):
                    m.P.round_()
                    m.clamp_parameters()
                    
    def log_params(self):
        logs = super().log_params()
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, dcls_module):
                
                curr_pos_ff = layer.P.cpu().detach().numpy()
                logs[f'pos_feedforward{idx}'] = curr_pos_ff.mean()
                
                
                ff_w = layer.weight
                ff_w_mean = torch.abs(ff_w).mean()
                ff_w_grad_max = ff_w.grad.abs().max().item() if ff_w.grad is not None else 0.0
                # With a hybrid parametrization the leaf is P.original, not P.
                ff_P = learned_delay_parameter(layer, 'P')
                ff_d_grad_max = ff_P.grad.abs().max().item() if ff_P.grad is not None else 0.0

                logs.update({
                    f'feedforward_dcls_w_{idx}': ff_w_mean,
                    f'feedforward_dcls_w_grad_max_{idx}': ff_w_grad_max,
                    f'feedforward_dcls_delay_grad_max_{idx}': ff_d_grad_max,
                })
                
        return logs

class SNN_axonal_feedforward_delays(SNN_synaptic_feedforward_delays):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        layers = []
        dim_buffer = config.input_size

        for idx, layer_dim in enumerate(config.hidden_layers):

            layers.extend(self._delay_stage(
                dim_buffer, layer_dim,
                delayed=not (config.no_delay_in_first_layer and idx == 0),
                ))
            dim_buffer = layer_dim

            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))

            layers.append(config.neuron_module(
            tau = config.tau,
            decay_input = config.decay_input,
            v_reset = config.v_reset,
            v_threshold = config.v_threshold,
            surrogate_function = config.surrogate_function,
            detach_reset = config.detach_reset,
            step_mode = config.step_mode,
            backend = config.backend,
            store_v_seq = config.store_v_seq,
            )
                            )

            layers.append(spike_registrator())

            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))

        layers.extend(self._delay_stage(
            dim_buffer, config.output_size, delayed=not config.no_delay_in_last_layer))

        self.layers = torch.nn.Sequential(*layers)

        self.init_weights()

    def _delay_stage(self, in_dim, out_dim, delayed):
        """Modules carrying the feedforward delay + projection for one layer.

        Axonal: a depthwise unit-weight DCLS filter that only time-shifts each
        source channel, followed by a trainable ``nn.Linear`` that owns the weights
        and bias. Subclasses override this to change the delay parametrization
        (see ``SNN_hybrid_feedforward_delays``).

        The depthwise conv must not add current: a bias there is one constant per
        source neuron applied at every timestep ahead of the Linear, so it fans out
        through that Linear and is integrated by the LIF over the whole sequence - a
        large DC term that swings the hidden neurons between dead and saturated and
        stalls training. Bias lives on the trainable Linear only.
        """
        if not delayed:
            return [torch.nn.Linear(in_dim, out_dim, bias=self.config.bias)]
        delay_config = copy(self.config)
        delay_config.bias = False
        return [
            dcls_module(delay_config, in_channels=in_dim, out_channels=in_dim, groups=in_dim),
            torch.nn.Linear(in_dim, out_dim, bias=self.config.bias),
            ]

    def init_weights(self):
        for m in self.layers:
            # Feedforward weights init
            if isinstance(m, torch.nn.Linear):
                if self.config.init_ff_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # For big models
                elif self.config.init_ff_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_ff_weights == 'default':
                    pass
                
            if isinstance(m, dcls_module):
                torch.nn.init.constant_(m.weight, 1.0)
                m.weight.requires_grad = False

            # Feedforward delays init
            if isinstance(m, dcls_module):
                torch.nn.init.uniform_(m.P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                m.clamp_parameters()

                if self.config.DCLSversion == 'gauss':
                    torch.nn.init.constant_(m.SIG, self.config.siginit)
                    m.SIG.requires_grad = False
                    
    
    def log_params(self):
        logs = super().log_params()
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, dcls_module):
                
                curr_pos_ff = layer.P.cpu().detach().numpy()
                logs[f'pos_feedforward{idx}'] = curr_pos_ff.mean()
                
                
                ff_w = layer.weight
                ff_w_mean = torch.abs(ff_w).mean()
                ff_w_grad_max = ff_w.grad.abs().max().item() if ff_w.grad is not None else 0.0
                ff_d_grad_max = layer.P.grad.abs().max().item() if layer.P.grad is not None else 0.0

                logs.update({
                    f'feedforward_w_{idx}': ff_w_mean,
                    f'feedforward_w_grad_max_{idx}': ff_w_grad_max,
                    f'feedforward_delay_grad_max_{idx}': ff_d_grad_max,
                })
                
                # Log sigma values for each DCLS layer
                if self.config.DCLSversion == 'gauss':
                    sigma_val = layer.SIG.item() if layer.SIG.numel() == 1 else layer.SIG.mean().item()
                    logs.update({
                        f'dcls/sigma_layer_{idx}': sigma_val,
                    })
                
        return logs

class _AxonalWithFixedOffsets(torch.nn.Module):
    """Reparametrize a delay tensor as ``axonal + position_shift + position_sign * offsets``.

    ``axonal`` is the learned leaf (one value per source neuron); ``offsets`` are the
    frozen per-synapse integers ``delta_ij`` drawn once and stored as a buffer. The
    result is the effective per-synapse delay tensor DCLS / the recurrent scan sees.

    Sign convention: on the feedforward (DCLS) path a DCLS position ``P`` grows toward
    the present, so a positive transmission delay *subtracts* from ``P``; the caller
    passes ``position_sign=-1.0`` and ``position_shift=maximum`` there, and widens the
    kernel by ``2 * maximum`` so that all-zero offsets reproduce the original lag. On
    the recurrent path the delay tensor is used directly (larger delay == later
    arrival), so the defaults ``position_shift=0.0``, ``position_sign=1.0`` apply.
    """

    def __init__(self, offsets, position_shift=0.0, position_sign=1.0):
        super().__init__()
        self.register_buffer('offsets', offsets)
        self.position_shift = position_shift
        self.position_sign = position_sign

    def forward(self, axonal):
        # axonal broadcasts against offsets (e.g. (1, 1, in, k) against (1, out, in, k),
        # or (N,) against (N, N)); a plain add would raise otherwise.
        assert torch.broadcast_shapes(axonal.shape, self.offsets.shape) == self.offsets.shape, (
            f'axonal {tuple(axonal.shape)} does not broadcast into offsets '
            f'{tuple(self.offsets.shape)}')
        return axonal + self.position_shift + self.position_sign * self.offsets


def learned_delay_parameter(module, name):
    """Return the leaf parameter, including for shared/parametrized delays."""
    from torch.nn.utils import parametrize
    if parametrize.is_parametrized(module, name):
        return getattr(module.parametrizations, name).original
    return getattr(module, name)


def _hybrid_params(config):
    """Validate and return ``(maximum, seed)`` for the hybrid delay classes.

    ``maximum`` (``config.hybrid_max_synaptic_delay``, default 4) is the inclusive
    upper bound on the frozen per-synapse offsets; ``maximum == 0`` makes a hybrid
    byte-for-byte its paired axonal model. ``seed`` (``config.hybrid_delay_seed``,
    default 123) seeds the offset draw, independent of the dataset/model seed.
    ``kernel_count == 1`` is required (one delay per axon/synapse).
    """
    maximum = getattr(config, 'hybrid_max_synaptic_delay', 4)
    if int(maximum) != maximum or maximum < 0:
        raise ValueError('hybrid_max_synaptic_delay must be a nonnegative integer')
    if config.kernel_count != 1:
        raise ValueError('hybrid delay models require kernel_count=1 (one delay per axon/synapse).')
    return int(maximum), getattr(config, 'hybrid_delay_seed', 123)


def _draw_offsets(shape, maximum, generator):
    """Frozen per-synapse integer offsets in ``[0, maximum]`` (all-zero when maximum == 0).

    ``torch.randint`` treats its first positional arg as ``high`` (exclusive), so the
    bound is ``maximum + 1``; ``torch.randint(1, shape)`` would mean ``high=1`` (a bug).
    """
    if maximum == 0:
        return torch.zeros(shape)
    return torch.randint(maximum + 1, shape, generator=generator).float()


def _reparametrize_feedforward_delay(module, maximum, generator, config, *, freeze_sig):
    """In place: turn a fused dense ``dcls_module`` into the hybrid feedforward delay.

    Widen the kernel by ``2 * maximum`` (and ``left_padding`` to match), then tie the
    position ``P`` to one learned delay per source (leaf shape ``(1, 1, in, k)``) plus
    the frozen offsets, via ``_AxonalWithFixedOffsets`` with ``position_sign=-1.0``
    (DCLS ``P`` grows toward the present) and ``position_shift=maximum`` (re-centred so
    all-zero offsets reproduce the original lag). ``freeze_sig`` pins ``SIG`` at
    ``config.siginit`` for the gauss kernel (the ``SNN_axonal_feedforward_delays``
    policy; the both-pathways axonal model leaves ``SIG`` trainable, so its hybrid
    passes ``freeze_sig=False``).
    """
    from torch.nn.utils import parametrize
    offsets = _draw_offsets(module.P.shape, maximum, generator)
    base = module.P[:, :1].detach().clone()
    module.dilated_kernel_size = (config.max_feedforward_delay + 2 * maximum,)
    module.left_padding += 2 * maximum
    module.DCK = type(module.DCK)(module.out_channels, module.in_channels,
        module.groups, module.kernel_count, module.dilated_kernel_size, module.version)
    parametrize.register_parametrization(module, 'P',
        _AxonalWithFixedOffsets(offsets, maximum, -1.0), unsafe=True)
    module.parametrizations.P.original = torch.nn.Parameter(base)
    if freeze_sig and config.DCLSversion == 'gauss':
        torch.nn.init.constant_(module.SIG, config.siginit)
        module.SIG.requires_grad = False


def _reparametrize_recurrent_delay(module, maximum, generator):
    """In place: tie a ``synaptic_recdel``'s (N, N) ``recurrent_delays`` to one learned
    (N,) leaf plus the frozen (N, N) offsets.

    ``forward_version`` is left unset and the module is tagged ``_hybrid_delay`` so
    ``synaptic_recdel.forward`` routes it: on CUDA to the dedicated fused hybrid
    Triton path (``delrec.triton_kernels.synaptic_hybrid``) when that is usable,
    to the spike-sparse event-driven kernel when the hybrid-specific gate declines,
    and to the pure-torch ``v2`` scan on CPU or with ``decay_input=True``.
    Explicit ``forward_version`` selections are respected. Kernel errors propagate,
    including errors raised later during backward.
    """
    from torch.nn.utils import parametrize
    offsets = _draw_offsets(module.recurrent_delays.shape, maximum, generator)
    base = module.recurrent_delays[0].detach().clone()
    parametrize.register_parametrization(module, 'recurrent_delays',
        _AxonalWithFixedOffsets(offsets), unsafe=True)
    module.parametrizations.recurrent_delays.original = torch.nn.Parameter(base)
    module._hybrid_delay = True
    module.forward_version = None


class SNN_hybrid_feedforward_delays(SNN_axonal_feedforward_delays):
    """Feedforward-only hybrid delays: learned axonal position + fixed random per-synapse offset.

    Effective feedforward delay d(i, j) = d_j + delta_ij (j = presynaptic source).
    d_j is one learned axonal delay per source, shared across its targets (as in
    ``SNN_axonal_feedforward_delays``); delta_ij is a per-synapse integer offset in
    ``[0, hybrid_max_synaptic_delay]``, drawn once and stored as a buffer, never
    trained.

    ``hybrid_max_synaptic_delay == 0`` is byte-for-byte ``SNN_axonal_feedforward_delays``
    (same modules, same init at a fixed seed, same forward). For ``> 0`` each delayed
    projection becomes a single fused dense ``dcls_module`` (groups=1, kernel_count=1):
    its weight is a trainable projection (kaiming_uniform, as an ``nn.Linear``), its
    bias follows ``config.bias``, its ``SIG`` is frozen at ``config.siginit`` (gauss),
    and its position ``P`` is one learned delay per source (leaf shape (1, 1, in, 1))
    plus the offsets. Old ``> 0`` checkpoints will not load.

    Config knobs (shared with the other hybrids):
      hybrid_max_synaptic_delay : inclusive upper bound on delta_ij, integer >= 0 (default 4).
      hybrid_delay_seed         : RNG seed for the fixed offsets (default 123).
    """

    def __init__(self, config):
        self._hybrid_max, seed = _hybrid_params(config)
        super().__init__(config)
        if self._hybrid_max == 0:
            return

        maximum = self._hybrid_max
        generator = torch.Generator().manual_seed(seed)
        self.base_position_bound = config.max_feedforward_delay // 2
        for module in self.layers:
            if isinstance(module, dcls_module):
                _reparametrize_feedforward_delay(module, maximum, generator, config,
                                                 freeze_sig=True)

    def _delay_stage(self, in_dim, out_dim, delayed):
        if self._hybrid_max == 0 or not delayed:
            return super()._delay_stage(in_dim, out_dim, delayed)
        # Fused dense DCLS: one operator does projection + per-synapse delay. A bias on
        # a groups=1 projection is the ordinary nn.Linear bias (added once per output),
        # not the fanned-out depthwise DC term, so config.bias is kept.
        return [dcls_module(self.config, in_channels=in_dim, out_channels=out_dim, groups=1)]

    def init_weights(self):
        if self._hybrid_max == 0:
            return super().init_weights()
        for m in self.layers:
            if isinstance(m, torch.nn.Linear):
                if self.config.init_ff_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.config.init_ff_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_ff_weights == 'default':
                    pass
            if isinstance(m, dcls_module):
                # Fused projection weight: trainable, initialised like nn.Linear's
                # default (kaiming_uniform, a=sqrt(5)) - what the axonal Linear produces.
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                torch.nn.init.uniform_(m.P, a=self.config.init_pos_a, b=self.config.init_pos_b)
                m.clamp_parameters()
                if self.config.DCLSversion == 'gauss':
                    torch.nn.init.constant_(m.SIG, self.config.siginit)
                    m.SIG.requires_grad = False

    def clamp_delays(self, train=True):
        if self._hybrid_max == 0:
            return super().clamp_delays(train=train)
        with torch.no_grad():
            for module in self.layers:
                if isinstance(module, dcls_module):
                    learned_delay_parameter(module, 'P').clamp_(
                        -self.base_position_bound, self.base_position_bound)

    def round_pos(self):
        if self._hybrid_max == 0:
            return super().round_pos()
        with torch.no_grad():
            for module in self.layers:
                if isinstance(module, dcls_module):
                    learned_delay_parameter(module, 'P').round_()
        self.clamp_delays()


class SNN_recurrent_hybrid_delays(SNN_axonal_recurrent_delays):
    """Recurrent-only hybrid delays: learned axonal position + fixed random synaptic offset.

    Effective recurrent delay d(i, j) = d_j + delta_ij. d_j is a single learned
    axonal delay per neuron, shared across all of its recurrent connections (as in
    ``SNN_axonal_recurrent_delays``); delta_ij is a per-synapse integer offset drawn once
    and stored as a buffer, never trained. The result keeps the per-synapse delay
    resolution of ``SNN_synaptic_recurrent_delays`` while optimizing only the
    axonal parameter count.

    No feedforward delays are involved, so at ``maximum == 0`` this is byte-for-byte
    ``SNN_axonal_recurrent_delays`` (it honours ``no_recurrence_in_last_layer`` and
    runs the same constructor). For ``maximum > 0`` the recurrent layers are built as
    ``synaptic_recdel`` (full (N, N) delay tensor, same layout as
    ``SNN_synaptic_recurrent_delays``) and then reparametrized in place so each (N, N)
    tensor is tied to one learned (N,) axonal leaf plus the frozen (N, N) offsets.

    Config knobs (shared with the other hybrids):
      hybrid_max_synaptic_delay : inclusive upper bound on delta_ij, integer >= 0 (default 4).
      hybrid_delay_seed         : RNG seed for the fixed offsets (default 123).
    """

    def __init__(self, config):
        maximum, seed = _hybrid_params(config)
        self._hybrid_max = maximum
        if maximum == 0:
            super().__init__(config)
            return
        SNN_synaptic_recurrent_delays.__init__(self, config)
        generator = torch.Generator().manual_seed(seed)
        for module in self.layers:
            if isinstance(module, axonal_recdel):   # a synaptic_recdel instance
                _reparametrize_recurrent_delay(module, maximum, generator)

    def clamp_delays(self):
        if self._hybrid_max == 0:
            return super().clamp_delays()
        with torch.no_grad():
            for module in self.layers:
                if isinstance(module, axonal_recdel):
                    learned_delay_parameter(module, 'recurrent_delays').clamp_(min=0)

    def round_pos(self):
        if self._hybrid_max == 0:
            return super().round_pos()
        with torch.no_grad():
            for module in self.layers:
                if isinstance(module, axonal_recdel):
                    learned_delay_parameter(module, 'recurrent_delays').round_()
        self.clamp_delays()
