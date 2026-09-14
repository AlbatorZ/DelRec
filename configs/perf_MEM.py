"""Small reproducible memorization experiment; no validation/test split.

Run: python experiments/train_mem.py
Change num_samples to explore capacity, keeping dataset_seed fixed for comparisons.
"""

from spikingjelly.activation_based import neuron, surrogate


class Config:
    dataset = "MEM"
    model = "SNN_recurrent_and_feedforward_delays"
    seed = 0
    dataset_seed = 0
    task_type = "temporal"
    num_samples = 256
    input_size = 16
    time_window = 32
    output_size = 4
    input_gain = 1.0
    hidden_layers = [64]
    epochs = 100
    delay_diagnostics_every = 10  # snapshots at 0, every X epochs, and final
    batch_size = 64
    num_workers = 0
    cpu_threads = 1
    readout = "mean"  # mean, sum, or last temporal output

    bias = False
    use_batch_norm = False
    feedforward_dropout_rate = 0.0
    recurrent_dropout_rate = 0.0
    init_ff_weights = "default"
    init_dcls_weights = "default"
    no_delay_in_first_layer = False
    no_delay_in_last_layer = False
    no_recurrence_in_last_layer = False

    neuron_module = neuron.LIFNode
    surrogate_function = surrogate.ATan(alpha=2.0)
    tau = 2.0
    decay_input = False
    v_threshold = 1.0
    v_reset = 0.0
    detach_reset = False
    step_mode = "m"
    backend = "torch"
    store_v_seq = False

    init_rec_weights = "orthogonal"
    rec_delay_init_gain = 0.5
    use_rec_bias = False
    init_rec_delay = "uniform"
    init_recdel_offset = 0.0
    max_rec_delay = 8.0
    delay_std_init = 2.0
    use_sig_p = False
    sigma_init = 0.0
    sigma_decay = 0.95

    #Hybrid delays configuration
    hybrid_max_synaptic_delay = 6 # fixed integer offsets in [0, 4] on both pathways
    hybrid_delay_seed = 123  # independent of dataset and model seed
    round_delays = False
    round_pos_each_epoch = False

    DCLSversion = "gauss"
    kernel_count = 1
    max_feedforward_delay = 9
    left_padding = max_feedforward_delay - 1
    right_padding = 0  # causal, length-preserving convolution
    init_pos_a = -(max_feedforward_delay // 2)
    init_pos_b = max_feedforward_delay // 2
    siginit = 1.0  # scheduled to 0.23 in the first half of training

    # Rates apply to all weights/delays in the selected network family.
    lr_w_recurrent = 0.005
    lr_positions_recurrent = 0.08
    lr_w_feedforward = 0.005
    lr_positions_feedforward = 0.08
    # Legacy fallback; make_optimizer records the selected effective rates here.
    lr_w = 0.005
    lr_positions = 0.08
    weight_decay = 0.0
    grad_clip = 1.0
