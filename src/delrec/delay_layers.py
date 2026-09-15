"""The recurrent spiking layers with learnable transmission delays.

    vanilla_recurrent   plain recurrent LIF, no delays (the control)
    axonal_recdel       one delay per neuron,      d_j   -> shape (N,)
    synaptic_recdel     one delay per connection,  d_ij  -> shape (N, N)
    common_recdel       one delay shared by a whole layer

A spike emitted by neuron j at time t reaches its targets at t + 1 + d_j, where
d is learned. To make d differentiable it is relaxed to the reals and the spike is
spread over neighboring integer steps by a triangular kernel whose width is annealed
to zero, so a trained network ends up with one integer delay per connection.

These classes hold the parameters. The computation lives in
:mod:`delrec.delay_layers_pytorch`, or in :mod:`delrec.delay_layers_triton` when a
fused GPU kernel is available. The two are numerically identical.
Tensors are time-major ``(T, B, N)`` throughout.
"""

import torch

from spikingjelly.activation_based import neuron, base, layer, surrogate

from delrec.delay_layers_pytorch import (
    VanillaTorchScans, AxonalTorchScans, SynapticTorchScans,
)
from delrec.delay_layers_triton import AxonalTritonScans, SynapticTritonScans


def _make_recurrent_dropout(config):
    """Recurrent dropout, applied in single-step mode inside the recurrence loop."""
    return layer.Dropout(config.recurrent_dropout_rate, step_mode='s')


class vanilla_recurrent(VanillaTorchScans, base.MemoryModule):
    def __init__(
        self,
        config, 
        neurons: int,
        neuron_module: torch.nn.Module = neuron.LIFNode, 
        ):

        super().__init__()
        
        self.config = config
        self.step_mode = config.step_mode
        self.store_v_seq = config.store_v_seq
        
        self.neuron_module = neuron_module(
            tau = config.tau,
            decay_input = config.decay_input,
            v_reset = config.v_reset,
            v_threshold = config.v_threshold,
            surrogate_function = config.surrogate_function,
            detach_reset = config.detach_reset,
            step_mode = 's',
            backend = config.backend,
        )
        
        self.neurons = neurons
        
        self.recurrent_weights = torch.nn.Parameter(torch.zeros(neurons, neurons), requires_grad=True)
        
        self.dropout = _make_recurrent_dropout(config)
        
        self.init_recurrent_weights()

    def init_recurrent_weights(self):
        torch.nn.init.orthogonal_(self.recurrent_weights, gain=1.0)

    def forward(self, x_seq: torch.Tensor):
        if self.step_mode == 's':
            return super().single_step_forward(x_seq)
        else:
            return self.multi_step_forward(x_seq)


# Recurrent forward kernel used on CUDA when a layer's `forward_version` is
# left unset (None). Both are exact (bit-identical forward to v2, matching
# gradients). Hardcode a different choice at the top of any script:
#     import delrec.delay_layers as rn
#     rn.AXONAL_CUDA_DEFAULT = 'v2'
AXONAL_CUDA_DEFAULT = 'triton_exact'
SYNAPTIC_CUDA_DEFAULT = 'eventdriven'


class axonal_recdel(AxonalTorchScans, AxonalTritonScans, base.MemoryModule):
    def __init__(
        self,
        config, 
        neurons: int,
        neuron_module: torch.nn.Module = neuron.LIFNode, 
        ):

        super().__init__()
        
        self.config = config
        self.step_mode = config.step_mode
        self.store_v_seq = config.store_v_seq
        
        self.neuron_module = neuron_module(
            tau = config.tau,
            decay_input = config.decay_input,
            v_reset = config.v_reset,
            v_threshold = config.v_threshold,
            surrogate_function = config.surrogate_function,
            detach_reset = config.detach_reset,
            step_mode = 's',
            backend = config.backend,
        )
        
        self.neurons = neurons
        
        self.sigma = float(self.config.sigma_init)
        # STE on the delay discretization: the forward taps at round(d) while the
        # gradient flows straight through to the fractional recurrent_delays.
        # Honored by every kernel (v1/v2/triton_*). Pair it with
        # round_pos_each_epoch=False. The in-place per-epoch round would destroy
        # the fractional master copy the STE exists to maintain.
        self.round_delays = getattr(config, 'round_delays', False)
        
        # For sanity check :
        # self.sigma = 0
        
        # Eq. (5): one delay per presynaptic neuron, shared by all its outgoing
        # connections, the axonal case, by analogy with a conduction delay set by
        # the myelination of a single axon.
        self.recurrent_weights = torch.nn.Parameter(torch.zeros(neurons, neurons), requires_grad=True)
        self.recurrent_delays = torch.nn.Parameter(torch.zeros(neurons), requires_grad=True)
        
        self.dropout = _make_recurrent_dropout(config)
        
        self.use_sig_p = config.use_sig_p
        if self.use_sig_p:
            self.p_spread = torch.nn.Parameter(torch.zeros(neurons), requires_grad=True)

        # Optional per-postsynaptic-neuron bias on the recurrent connection,
        # matching nn.Linear(N, N) in neuroseqbench's Recurrent_LIF.
        self.use_rec_bias = getattr(config, 'use_rec_bias', False)
        if self.use_rec_bias:
            self.recurrent_bias = torch.nn.Parameter(torch.zeros(neurons), requires_grad=True)

        self.init_recurrent_weights()
        self.init_recurrent_delays()

    def init_recurrent_weights(self):
        init_mode = getattr(self.config, 'init_rec_weights', 'orthogonal')
        if init_mode == 'orthogonal':
            torch.nn.init.orthogonal_(self.recurrent_weights, gain=self.config.rec_delay_init_gain)
        elif init_mode == 'kaiming_uniform':
            # Match nn.Linear(N, N) default init (kaiming_uniform with a=sqrt(5)).
            # Recurrent_weights is (N, N) so fan_in == N, giving entries U(-1/sqrt(N), 1/sqrt(N)).
            torch.nn.init.kaiming_uniform_(self.recurrent_weights, a=5 ** 0.5)
        else:
            raise ValueError(f"Unknown init_rec_weights '{init_mode}' (expected 'orthogonal' or 'kaiming_uniform').")
        if getattr(self, 'use_rec_bias', False):
            # nn.Linear bias init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)), fan_in == N
            bound = 1.0 / (self.recurrent_weights.shape[1] ** 0.5)
            torch.nn.init.uniform_(self.recurrent_bias, -bound, bound)

    def init_recurrent_delays(self):
        with torch.no_grad():
            if self.config.init_rec_delay == 'half_normal':
                half_normal = torch.abs(torch.randn_like(self.recurrent_delays) * self.config.delay_std_init)
                self.recurrent_delays.copy_(half_normal)
                self.recurrent_delays.clamp_(min=0.0)
            elif self.config.init_rec_delay == 'uniform':
                torch.nn.init.uniform_(self.recurrent_delays, a=self.config.init_recdel_offset, b=self.config.max_rec_delay)
                self.recurrent_delays.clamp_(min=0.0)

    def update_sigma(self, current_epoch):
        """Eq. (13): anneal the spread, sigma_epoch = sigma_init * decay^(100*epoch/N_epochs).

    Sigma is a plain attribute, not a buffer, so it is not carried in the state_dict
    and must be set explicitly at evaluation.
    """
        decay_per_epoch = self.config.sigma_decay ** (100 / self.config.epochs)
        self.sigma = self.config.sigma_init * (decay_per_epoch ** current_epoch)

    def clamp_recurrent_delays(self):
        with torch.no_grad():
            self.recurrent_delays.clamp_(min=0)

    def forward(self, x_seq: torch.Tensor):
        if self.step_mode == 's':
            return super().single_step_forward(x_seq)
        else:
            fv = getattr(self, 'forward_version', None)
            if fv is None:
                # Default: AXONAL_CUDA_DEFAULT (triton_exact) on CUDA, pure-torch v2 on
                # CPU. Both Triton paths assume decay_input=False, so fall back to v2
                # when decay_input is set.
                fv = (AXONAL_CUDA_DEFAULT
                      if (x_seq.is_cuda
                          and not getattr(self.neuron_module, 'decay_input', False))
                      else 'v2')
            if fv == 'v1':
                return self.multi_step_forward_v1(x_seq)
            elif fv == 'triton_exact':
                return self.multi_step_forward_triton(x_seq, approx=False)
            elif fv == 'triton_approx':
                return self.multi_step_forward_triton(x_seq, approx=True)
            elif fv == 'triton_ste':
                return self.multi_step_forward_triton(x_seq, ste_mode=1)
            elif fv in ('eventdriven', 'eventdriven_torch'):
                raise ValueError(
                    "axonal_recdel has no event-driven kernel in this release. "
                    "Use 'triton_exact' (the CUDA default), 'v2' or 'v1'.")
            return self.multi_step_forward_v2(x_seq)


class synaptic_recdel(SynapticTorchScans, SynapticTritonScans, axonal_recdel):
    def __init__(
        self,
        config, 
        neurons: int,
        neuron_module: torch.nn.Module = neuron.LIFNode, 
        ):

        super().__init__(
            config=config,
            neurons=neurons,
            neuron_module=neuron_module,
        )
        
        # For sanity check :
        # self.sigma = 0
        
        # Eq. (5), synaptic case: one delay d_ij per connection, hence (N, N).
        self.recurrent_delays = torch.nn.Parameter(torch.zeros(neurons, neurons), requires_grad=True) # (N, N)
        self.init_recurrent_delays()

    def forward(self, x_seq: torch.Tensor):
        if self.step_mode == 's':
            return super().single_step_forward(x_seq)
        else:
            fv = getattr(self, 'forward_version', None)
            if fv is None:
                # Default: SYNAPTIC_CUDA_DEFAULT (eventdriven) on CUDA, pure-torch v2 on
                # CPU. Eventdriven's LIF math assumes decay_input=False, so fall back to
                # v2 when decay_input is set. (synaptic has no triton_exact path.)
                fv = (SYNAPTIC_CUDA_DEFAULT
                      if (x_seq.is_cuda
                          and not getattr(self.neuron_module, 'decay_input', False))
                      else 'v2')
            if fv == 'v1':
                return self.multi_step_forward_v1(x_seq)
            elif fv == 'eventdriven_torch':
                raise ValueError(
                    "the pure-PyTorch event-driven reference was removed from this "
                    "release, use 'eventdriven' (the CUDA default), 'v2' or 'v1'.")
            elif fv == 'eventdriven':
                # Hybrid delays share one learned base per source. Use their
                # fused delay-gradient kernel when supported; explicit v1/v2
                # selections above/below still use the requested reference path.
                if getattr(self, '_hybrid_delay', False):
                    from delrec.triton_kernels.synaptic_hybrid import (
                        hybrid_kernel_usable, hybrid_trainable_forward,
                    )
                    if hybrid_kernel_usable(self, x_seq):
                        return hybrid_trainable_forward(self, x_seq)
                return self.multi_step_forward_eventdriven(x_seq)
            # A fused Triton scan loses to v2's cuBLAS-per-step GEMM for per-synapse
            # delays. The spike-sparse fast path is multi_step_forward_eventdriven.
            return self.multi_step_forward_v2(x_seq)


class common_recdel(axonal_recdel):
    """axonal_recdel with a SINGLE learnable delay shared across all neurons in the layer
    (recurrent_delays shape (1,), broadcast to (N,) by multi_step_forward_v2).
    """

    def __init__(
        self,
        config,
        neurons: int,
        neuron_module: torch.nn.Module = neuron.LIFNode,
        ):

        super().__init__(config=config, neurons=neurons, neuron_module=neuron_module)

        # Replace the (N,) per-neuron delay with one shared scalar.
        self.recurrent_delays = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.init_recurrent_delays()   # half_normal/uniform both work on shape (1,)
        self.forward_version = 'v2'    # broadcast-safe path
