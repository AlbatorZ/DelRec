"""Network zoo for SSC, PS-MNIST, HAR, AL and Mackey-Glass.

A hidden block is Linear -> Dropout -> neuron -> spike_registrator -> [BatchNorm]. The network ends in a Linear plus a leaky-integrate layer with an infinite
threshold, so ``forward`` returns a membrane sequence rather than spikes.

The class hierarchy is the experimental matrix: ``SNN_recurrent_delays`` (axonal,
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
    
class SNN_recurrent_delays(SNN):
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


class SNN_synaptic_recurrent_delays(SNN_recurrent_delays):
    """Identical to SNN_recurrent_delays but with per-synapse delays (synaptic_recdel,
    recurrent_delays shape (N, N)) instead of per-neuron axonal delays.
    """

    def __init__(self, config):
        # Skip SNN_recurrent_delays.__init__ (it hardcodes axonal_recdel). Rebuild
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


class SNN_common_recurrent_delays(SNN_recurrent_delays):
    """Identical to SNN_recurrent_delays but each recurrent layer learns a single shared delay
    (common_recdel, recurrent_delays shape (1,)) instead of per-neuron axonal delays.
    """

    def __init__(self, config):
        # Skip SNN_recurrent_delays.__init__ (it hardcodes axonal_recdel). Rebuild
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


class SNN_vanilla_recurrent(SNN_recurrent_delays):
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

class SNN_fixed_recurrent_delays(SNN_recurrent_delays):
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

class SNN_feedforward_delays(SNN):
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
                ff_d_grad_max = layer.P.grad.abs().max().item() if layer.P.grad is not None else 0.0

                logs.update({
                    f'feedforward_dcls_w_{idx}': ff_w_mean,
                    f'feedforward_dcls_w_grad_max_{idx}': ff_w_grad_max,
                    f'feedforward_dcls_delay_grad_max_{idx}': ff_d_grad_max,
                })
                
        return logs

class SNN_axonal_feedforward_delays(SNN_feedforward_delays):
    def __init__(self, config):
        super().__init__(config)  

        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            
            print(idx, layer_dim, dim_buffer)
            
            if config.no_delay_in_first_layer and idx == 0:
                layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            else:
                layers.append(
                    dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = dim_buffer,
                        groups = dim_buffer,
                    )
                    ) # (T, B, N_in) -> (T, B, N_hidden)
                layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias)) 
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
            
        if config.no_delay_in_last_layer:
            layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        else:
                layers.append(dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = dim_buffer,
                        groups = dim_buffer,
                    ))
                layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias)) 
            
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

class SNN_recurrent_and_feedforward_delays(SNN_feedforward_delays, SNN_recurrent_delays):

    def __init__(self, config):
        super(SNN_feedforward_delays, self).__init__(config)  

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

            layers.append(axonal_recdel(config, layer_dim, config.neuron_module))
                
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

        SNN_feedforward_delays.init_weights(self)

    def forward(self, x):
        return SNN.forward(self, x)
    
    def clamp_delays(self):
        SNN_feedforward_delays.clamp_delays(self)
        SNN_recurrent_delays.clamp_delays(self)
        
    def round_pos(self):
        SNN_feedforward_delays.round_pos(self)
        SNN_recurrent_delays.round_pos(self)
        
    def log_params(self):
        logs = SNN_feedforward_delays.log_params(self)
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
                    
                    if layer.use_sig_p:
                        logs[f"p_spread_mean_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().mean().item()
                        logs[f"p_spread_std_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().std().item()
                    
        return logs


class _AxonalWithFixedOffsets(torch.nn.Module):
    """Broadcast a learned axonal parameter, then add saved per-synapse offsets."""

    def __init__(self, offsets, position_shift=0.0, position_sign=1.0):
        super().__init__()
        self.register_buffer('offsets', offsets)
        self.position_shift = position_shift
        self.position_sign = position_sign

    def forward(self, axonal):
        return axonal + self.position_shift + self.position_sign * self.offsets


def learned_delay_parameter(module, name):
    """Return the leaf parameter, including for shared/parametrized delays."""
    from torch.nn.utils import parametrize
    if parametrize.is_parametrized(module, name):
        return getattr(module.parametrizations, name).original
    return getattr(module, name)


class SNN_axonal_feedforward_only_delays(SNN_feedforward_delays):
    """Axonal FF delays with the comparison's Linear/Dropout/LIF/recorder/BN order.

    Unlike the older SNN_axonal_feedforward_delays, the delay filter has no bias
    and normalization/dropout keep the same position as in the other models.
    """

    def __init__(self, config):
        torch.nn.Module.__init__(self)
        if config.kernel_count != 1:
            raise ValueError("These paired models require kernel_count=1 (one delay per axon/synapse).")
        self.config = config
        modules = []
        dim = config.input_size
        for idx, width in enumerate(config.hidden_layers):
            modules.extend(self._projection(dim, width, not (idx == 0 and config.no_delay_in_first_layer)))
            modules.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))
            modules.append(config.neuron_module(
                tau=config.tau, decay_input=config.decay_input,
                v_reset=config.v_reset, v_threshold=config.v_threshold,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset, step_mode=config.step_mode,
                backend=config.backend, store_v_seq=config.store_v_seq))
            modules.append(spike_registrator())
            if config.use_batch_norm:
                modules.append(modified_batchnorm(width, step_mode='m'))
            dim = width
        modules.extend(self._projection(dim, config.output_size, not config.no_delay_in_last_layer))
        self.layers = torch.nn.Sequential(*modules)
        self.init_weights()

    def _projection(self, inputs, outputs, delayed):
        if not delayed:
            return [torch.nn.Linear(inputs, outputs, bias=self.config.bias)]
        delay_config = copy(self.config)
        delay_config.bias = False  # The following Linear supplies the projection bias.
        return [dcls_module(delay_config, inputs, inputs, groups=inputs),
                torch.nn.Linear(inputs, outputs, bias=self.config.bias)]

    def init_weights(self):
        SNN_feedforward_delays.init_weights(self)
        for module in self.layers:
            if isinstance(module, dcls_module):
                torch.nn.init.ones_(module.weight)
                module.weight.requires_grad_(False)


class _HybridDelayMixin:
    """Add fixed synaptic offsets to an existing synaptic network's delays."""

    def __init__(self, config):
        super().__init__(config)
        self._init_hybrid_offsets()

    def _init_hybrid_offsets(self):
        from torch.nn.utils import parametrize
        config = self.config
        maximum = getattr(config, 'hybrid_max_synaptic_delay', 4)
        if int(maximum) != maximum or maximum < 0:
            raise ValueError('hybrid_max_synaptic_delay must be a nonnegative integer')
        maximum = int(maximum)
        generator = torch.Generator().manual_seed(getattr(config, 'hybrid_delay_seed', 123))
        self.base_position_bound = config.max_feedforward_delay // 2
        for module in self.layers:
            if isinstance(module, dcls_module):
                offsets = torch.randint(maximum + 1, module.P.shape, generator=generator).float()
                base = module.P[:, :1].detach().clone()
                # DCLS position increases toward the present: positive transmission
                # delay subtracts from P. Extend both sides of the kernel, then
                # shift the center so zero offsets preserve the original lag.
                module.dilated_kernel_size = (config.max_feedforward_delay + 2 * maximum,)
                module.left_padding += 2 * maximum
                module.DCK = type(module.DCK)(module.out_channels, module.in_channels,
                    module.groups, module.kernel_count, module.dilated_kernel_size, module.version)
                parametrize.register_parametrization(module, 'P',
                    _AxonalWithFixedOffsets(offsets, maximum, -1.0), unsafe=True)
                module.parametrizations.P.original = torch.nn.Parameter(base)
            elif isinstance(module, axonal_recdel):
                offsets = torch.randint(maximum + 1, module.recurrent_delays.shape, generator=generator).float()
                base = module.recurrent_delays[0].detach().clone()
                parametrize.register_parametrization(module, 'recurrent_delays',
                    _AxonalWithFixedOffsets(offsets), unsafe=True)
                module.parametrizations.recurrent_delays.original = torch.nn.Parameter(base)
                module.forward_version = 'v2'

    def clamp_delays(self):
        with torch.no_grad():
            for module in self.layers:
                if isinstance(module, dcls_module):
                    learned_delay_parameter(module, 'P').clamp_(-self.base_position_bound, self.base_position_bound)
                elif isinstance(module, axonal_recdel):
                    learned_delay_parameter(module, 'recurrent_delays').clamp_(min=0)

    def round_pos(self):
        stochastic = getattr(self.config, 'round_mode', 'nearest') == 'stochastic'
        with torch.no_grad():
            for module in self.layers:
                if isinstance(module, dcls_module):
                    learned_delay_parameter(module, 'P').round_()
                elif isinstance(module, axonal_recdel):
                    delay = learned_delay_parameter(module, 'recurrent_delays')
                    if stochastic:
                        delay.copy_(torch.floor(delay + torch.rand_like(delay)))
                    else:
                        delay.round_()
        self.clamp_delays()

    def log_params(self):
        logs = SNN.log_params(self)
        for idx, module in enumerate(self.layers):
            if isinstance(module, dcls_module):
                delay = learned_delay_parameter(module, 'P')
                weight = module.weight
                prefix = 'feedforward'
            elif isinstance(module, axonal_recdel):
                delay = learned_delay_parameter(module, 'recurrent_delays')
                weight = module.recurrent_weights
                prefix = 'recurrent'
                logs[f'sigma_rec{idx}'] = module.sigma
            else:
                continue
            logs[f'{prefix}_delay_mean_{idx}'] = delay.detach().mean()
            logs[f'{prefix}_delay_grad_max_{idx}'] = delay.grad.abs().max().item() if delay.grad is not None else 0.0
            logs[f'{prefix}_w_{idx}'] = weight.detach().abs().mean()
        return logs


class SNN_hybrid_recurrent_delays(_HybridDelayMixin, SNN_synaptic_recurrent_delays):
    """Existing recurrent synaptic network with learned axonal bases + fixed offsets."""


class SNN_hybrid_feedforward_only_delays(_HybridDelayMixin, SNN_feedforward_delays):
    """Existing dense FF delay network with learned axonal bases + fixed offsets."""


# Compatibility names used by the comparison: reuse existing implementations.
SNN_axonal_recurrent_only_delays = SNN_recurrent_delays
SNN_synaptic_recurrent_only_delays = SNN_synaptic_recurrent_delays
SNN_hybrid_recurrent_only_delays = SNN_hybrid_recurrent_delays
SNN_synaptic_feedforward_only_delays = SNN_feedforward_delays
