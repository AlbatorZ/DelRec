# Six-model HAR comparison

Run from the repository root:

```bash
python experiments/compare_har_snn_rsnn.py --device cuda
```

This trains all six models sequentially using `configs/perf_HAR.py` and the HAR
training, loss, readout, optimizer and scheduler from `delrec.training.har`.
Defaults are 100 epochs, batch size 256, seed 0 and topology
**3 → 128 → 176 → 176 → 18**. Only one model occupies the GPU at a time.

| Delay location | Type | Trainable parameters | Learned delays | Fixed offsets |
| --- | --- | ---: | ---: | ---: |
| Feedforward | Axonal | 58,037 | 483 | 0 |
| Feedforward | Hybrid | 58,037 | 483 | 57,056 |
| Feedforward | Synaptic | 114,610 | 57,056 | 0 |
| Recurrent | Axonal | 105,522 | 304 | 0 |
| Recurrent | Hybrid | 105,522 | 304 | 47,360 |
| Recurrent | Synaptic | 152,578 | 47,360 | 0 |

These counts are computed from the default models after optimizer setup. Counts
are recomputed for every run. Scheduled Gaussian widths and frozen unit filters
are excluded from trainable parameters; fixed offsets are buffers. The JSON/CSV
also report total parameter elements and weight/bias counts. Equal trainable
counts do not imply equal memory use or runtime.

Recurrent models have ordinary feedforward Linear projections and delayed
feedback in the first two hidden layers. HAR's `no_recurrence_in_last_layer=True`
is retained. Feedforward models have delayed projections (including input and
output by default) and no recurrent connections. The six compatibility classes
are the same ones used in `compare_mem_snn_rsnn.py`; their shared initialization
now lives in `delrec.comparison`.

Weights and biases of feedforward projections are matched across all six models.
Within each pathway, recurrent weights and initial learned delays are matched:
synaptic delays start as broadcast axonal delays, then train independently.
Hybrid delays learn an axonal base plus fixed per-synapse offsets sampled
uniformly from 0 through 4 steps. Offsets are present throughout training,
controlled by `--hybrid-delay-seed` (default 123) and
`--hybrid-max-synaptic-delay` (default 4). The hybrid feedforward kernel is extended
to accommodate the offsets. Maximum effective delays and parameter budgets are
not equalized across variants.

## RTX 5060 Ti setup

Use a separate Python 3.10–3.12 environment on the GPU computer. For example,
on Linux (on Windows activate with `.venv-har\Scripts\activate`):

```bash
python -m venv .venv-har
source .venv-har/bin/activate
python -m pip install -r requirements-blackwell.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
python experiments/compare_har_snn_rsnn.py --device cuda --smoke
python experiments/compare_har_snn_rsnn.py --device cuda
```

`requirements-blackwell.txt` selects PyTorch 2.7.1 / torchvision 0.22.1 CUDA 12.8
wheels while retaining the other benchmark dependency versions. PyTorch added
Blackwell support in [2.7 with CUDA 12.8](https://pytorch.org/blog/pytorch-2-7/);
the wheel combination is listed in the
[official installation commands](https://docs.pytorch.org/get-started/previous-versions/).
An NVIDIA driver compatible with these wheels must be installed. The original
`requirements.txt` and package metadata pin PyTorch 2.5.1; use this separate
requirements file for the 5060 Ti and run the entry point directly. The script
already adds `src/` to the import path, so an editable package install is unnecessary.

The default recurrent `v2` kernels are ordinary float32 PyTorch operations running
on CUDA. They work without Triton, including on native Windows. On Linux with
Triton, opt into the existing accelerators (including the hybrid synaptic scan):

```bash
python experiments/compare_har_snn_rsnn.py --device cuda --smoke --kernel triton_exact --synaptic-kernel eventdriven
python experiments/compare_har_snn_rsnn.py --device cuda --kernel triton_exact --synaptic-kernel eventdriven
```

The accelerated kernels have not been tested on a 5060 Ti in this workspace.
First validate them with the smoke command on that computer. Keep the default
kernels if the local compiler/driver cannot run them. Batch size 256 preserves
`perf_HAR.py`; memory also depends on the learned delays and convolution workspace.
If it exceeds the GPU's available memory, use `--batch-size 64` (or lower) for the
entire comparison so every model receives the same batch size. Mixed precision
and compilation are not enabled.

## Data and checkpoint selection

`Datasets/HAR` is the default, resolved relative to the repository rather than
the shell's working directory. Override it with `--datasets-path PATH`.
The existing loader reuses `x_train.npy`, `y_train.npy`, `x_test.npy` and
`y_test.npy`, or creates them from the local WISDM watch/gyro files if absent.

The comparison follows the paper's HAR reporting rule: **no validation set;
report the highest test accuracy over the training epochs**. Every model trains
on the full cached training partition and is evaluated on the cached test
partition after each epoch. `best.pth` contains the epoch with the highest test
accuracy (latest checkpoint wins ties); `last.pth` contains the final epoch.
All models and run seeds use the same cached partitions. DataLoader uses a
separate seeded generator so architecture-specific dropout random draws cannot
change batch order.

This preserves the cached, overlapping-window benchmark partition. It is not
a subject-independent evaluation: overlapping windows or subjects can occur
across partitions.

Loss is cross entropy of the time-averaged output logits. Training loss is the
online loss during updates; test loss and accuracy are evaluated after each
epoch. Gaussian feedforward widths anneal over the first half of training.
Recurrent smoothing and per-epoch delay rounding follow the epoch loop in
`train.py`. The reported accuracy is exactly the maximum recorded epoch accuracy,
and the reported loss comes from that same epoch. There is no additional final
measurement that changes smoothing or rounding. Checkpoints retain Gaussian
widths and recurrent sigmas so the selected epoch's evaluation can be reproduced.
Hybrid base delays are optimized directly, while fixed offsets remain unchanged.

For multiple seeds, the bars show the mean of each seed's best epoch accuracy;
error bars show the population standard deviation of those maxima. Each seed
can select a different epoch. Results record
`selection_protocol="best_test_accuracy_over_epochs"`.

## Outputs and options

Each invocation creates a new directory under
`exp/HAR/delay_location_comparison/<timestamp>/`, or the new path passed with
`--out PATH`. Existing output directories are rejected to avoid overwriting runs.

- `har_comparison.png` and `.pdf`: training loss, test loss, test
  accuracy, and best test accuracy bars annotated with trainable parameter counts.
- `comparison.csv` and `.json`: per-model/per-seed test metrics and parameter
  breakdowns; JSON includes histories for replotting.
- `split.json`: full train/test sample counts and the reporting protocol.
- `<model>/seed<N>/`: `config.json`, `parameters.json`, `history.csv`, `best.pth`,
  `last.pth`, and `final_test.json`. Checkpoints contain the model (including
  fixed offsets and Gaussian widths), optimizers, schedulers and recurrent sigmas.

Metrics and plots are refreshed after each completed model. To compare several
initializations, plot mean curves and best test accuracies with population standard
deviation across seeds:

```bash
python experiments/compare_har_snn_rsnn.py --device cuda --seeds 0,1,2
python experiments/compare_har_snn_rsnn.py --device cuda --epochs 50 --batch-size 64 --hidden-layers 64,96,96
```

`--smoke` uses two epochs, two batches per split, widths `[8,8,8]` and batch size
4 unless widths/batch size are explicitly supplied. Figures and results are
marked as smoke measurements, not benchmark results. It runs on CPU as well:

```bash
python experiments/compare_har_snn_rsnn.py --device cpu --smoke
```

To replot saved measurements without training:

```python
import json
import sys
from pathlib import Path
sys.path.insert(0, 'experiments')
from compare_har_snn_rsnn import plot_comparison
out = Path('exp/HAR/delay_location_comparison/YOUR_RUN')
data = json.loads((out / 'comparison.json').read_text())
plot_comparison(data['histories'], data['results'], out, data['smoke'])
```
