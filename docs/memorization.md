# Spike memorization

Run the combined recurrent/feedforward delay SNN using `configs/perf_MEM.py`:

```bash
.venv/bin/python experiments/train_mem.py
```

The default temporal task has 128 fixed samples, 16 input neurons, 32 time
steps, four independent random classes, and one hidden layer of 64 neurons.
Every input neuron spikes exactly once. The spatial alternative repeats one
random current vector throughout the sequence. Both use the supplied
`SpikeMemorization` generation procedure. There is no validation or test split.

Examples for comparing configurations:

```bash
.venv/bin/python experiments/train_mem.py --task-type spatial
.venv/bin/python experiments/train_mem.py --hidden-layers 64,64 --num-samples 256
.venv/bin/python experiments/train_mem.py --model SNN_vanilla_recurrent
.venv/bin/python experiments/train_mem.py --model SNN_feedforward_delays --seed 1
.venv/bin/python experiments/train_mem.py --epochs 300 --readout last
```

`--seed` controls model initialization and batch order; `--dataset-seed` controls
the fixed inputs and random labels independently. Keep dataset settings and the
dataset seed identical when comparing architectures. Other hyperparameters,
including delay ranges and learning rates, can be edited in `perf_MEM.py`.
`--device` accepts `auto`, `cpu`, `cuda`, or `mps`; auto selects CUDA when available,
otherwise CPU. The CPU default uses one thread for these small tensor operations.

Training minimizes cross entropy of time-averaged output logits by default.
`--readout sum` and `--readout last` select other reductions. The mean readout
measures random-pattern memorization using evidence throughout the sequence;
it does not require withholding the answer until a separate recall period.

Both feedforward positions and recurrent delays are optimized. Gaussian DCLS
widths are scheduled from `siginit` to 0.23 over the first half of training.
Recurrent smoothing uses the configured sigma schedule. Delays are clamped after
each update and remain fractional; measurements preserve the current smoothing
and positions, without rounding or a separate inference-time transformation.

Each run writes to a unique `exp/MEM/<model>/...` directory, or `--out PATH`:

- `config.json`: effective settings and device.
- `dataset.pt`: the exact input tensors and random labels.
- `train_res.csv`: epoch, loss, accuracy in percent, and online optimization loss.
  Loss/accuracy are recomputed on the entire training set after each epoch;
  online loss is the sample-weighted loss observed during that epoch's updates.
- `final_train.json`: final training loss, accuracy, correct count and model size.
- `last.pth`: final model, optimizer/scheduler, settings, metrics, and recurrent
  sigma values (which are not part of the model state dict).
- `training_summary.png` and `.pdf`: loss versus epoch and final training accuracy.

The final checkpoint is used directly; there is no validation-based selection.
Neuron and dropout states are reset between batches. Training accuracy quantifies
memorization of these samples, not generalization to new random labels.

## Axonal, synaptic and hybrid delays on one pathway

```bash
.venv/bin/python experiments/compare_mem_delays.py --pathway recurrent
.venv/bin/python experiments/compare_mem_delays.py --pathway feedforward
```

The three-model script now compares delay types on the selected pathway
(default: recurrent). It uses the same six single-pathway classes as the full
comparison below. The former three added combined-delay classes were removed.
The original `SNN_recurrent_and_feedforward_delays` remains available for the
original mixed-delay experiment via `train_mem.py`.

All models use the same topology, dataset, batch order and matched initial
connection weights/biases. Synaptic delays start as broadcast axonal delays;
hybrid delays share the learned base but add fixed random offsets. Each run
saves per-model checkpoints and metrics, plus `comparison.json` and
`delay_comparison.png` / `.pdf` with curves and parameter counts.

### Fixed random synaptic delays in the hybrid

The effective physical delay is `d[i,j] = axonal[j] + offset[i,j]` on the selected
delay pathway. Recurrence also retains its usual mandatory
one-step feedback lag. Offsets are drawn once **before training**, allowing the
learned weights and axonal delays to adapt to them. Adding random offsets after
training would instead measure robustness to a timing perturbation.

`hybrid_max_synaptic_delay = 4` samples integer offsets uniformly from 0 through
4 time steps. `hybrid_delay_seed = 123` controls this draw independently of the
dataset and weight seeds. Override them with `--hybrid-max-synaptic-delay` and
`--hybrid-delay-seed`. Offsets remain fixed throughout training and evaluation
and are saved as buffers in `last.pth`; they are not optimizer parameters.

For a 16 → 64 → 4 topology, the hybrid learns 80 delays and stores 1,280 fixed
offsets in feedforward-only mode, or learns 64 delays and stores 4,096 fixed
offsets in recurrent-only mode. It uses
synaptic computations to apply those distinct offsets, so equal learned parameter
counts do not imply equal runtime or storage costs.

DCLS positions run in the opposite direction from transmission delays. The
implementation subtracts fixed offsets from those positions and extends the
feedforward kernel window to accommodate the extra lag without clipping it.
The original learnable axonal position range is retained. Recurrent axonal
parameters are clamped nonnegative; their offsets are added afterwards.
Gaussian filtering remains the same type of smoothing, evaluated on the extended
window. The hybrid's longer effective delays are part of this experimental
condition; the experiment does not match maximum effective delays across models.

The comparison reads the current `perf_MEM.py`, so reruns may differ from older
results if neuron parameters have been edited. The recurrent-only architectures
apply recurrence to every hidden layer, even when `no_recurrence_in_last_layer`
is set; this preserves their existing topology.

## Six configurations: delay location × delay type

```bash
.venv/bin/python experiments/compare_mem_snn_rsnn.py
```

The corrected six-model experiment separates recurrent delays from feedforward
delays. Despite the historical script name, it no longer uses models with
delays on both pathways.

| Delay type | Recurrent delays only | Feedforward delays only |
| --- | --- | --- |
| Axonal | `SNN_axonal_recurrent_only_delays` | `SNN_axonal_feedforward_only_delays` |
| Synaptic | `SNN_synaptic_recurrent_only_delays` | `SNN_synaptic_feedforward_only_delays` |
| Hybrid | `SNN_hybrid_recurrent_only_delays` | `SNN_hybrid_feedforward_only_delays` |

Recurrent-only blocks are `Linear → Dropout → recurrent LIF → recorder →
[BatchNorm]`, with a final Linear output projection. They contain no DCLS
feedforward delay layers. Every hidden layer is recurrent; the legacy
`no_recurrence_in_last_layer` flag is not used by these comparison classes.

Feedforward-only blocks use a delayed projection followed by dropout, a plain
LIF, recorder and optional BatchNorm. They have no recurrent weights, biases
or delays. For the hybrid, fixed synaptic offsets exist only on the selected
pathway: recurrent in the recurrent-only model, feedforward in the other.
All learned connection weights remain trainable.

All six use the current `perf_MEM.py`, the same samples/labels and batch order.
Feedforward connection weights and biases are matched across locations. Within
each location, axonal/synaptic learned delay initialization is matched, and
hybrid starts from the same learned base plus fixed random offsets. Recurrent
and feedforward delays are separate parameter sets with different shapes;
they are not copied across locations. No validation or test set is used.

Output is saved under `exp/MEM/delay_location_comparison/<run>/`, with one full
run folder per model, `comparison.json`, and `delay_location_comparison.png` /
`.pdf`. Plots contain six curves and six bars with parameter counts. Solid
curves represent recurrent-only delays; dashed curves and hatched bars represent
feedforward-only delays. Historical results under `snn_rsnn_comparison` describe
the previous experiment and must not be treated as results of this correction.

The three-model script selects one pathway with `--pathway`. The original
`SNN_recurrent_and_feedforward_delays` class is preserved, but the three added
combined-delay classes have been removed. Historical configs naming those
removed classes are not supported by the current code. The six-model script supports `--epochs`,
`--seed`, `--dataset-seed`, `--num-samples`, `--task-type`, `--hidden-layers`,
`--hybrid-max-synaptic-delay`, `--hybrid-delay-seed`, `--device`, and `--out`.

The recurrent axonal and synaptic configurations reuse `SNN_recurrent_delays`
and `SNN_synaptic_recurrent_delays` directly. Synaptic feedforward uses
`SNN_feedforward_delays`. The hybrid classes extend those existing synaptic
networks with a shared offset mixin. Compatibility aliases preserve the
comparison's `*_only_delays` names without duplicating implementations.
The axonal feedforward class retains a small specialized builder to preserve
matching layer order and keep the temporal filter bias-free.

Existing recurrent classes honor `no_recurrence_in_last_layer` outside the
comparison. The comparison explicitly sets it to False in copied configs so
all recurrent hidden layers remain enabled, including one-hidden-layer runs.
