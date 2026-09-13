# DelRec

Reference implementation of *DelRec: learning delays in recurrent spiking neural networks*.

[![arXiv](https://img.shields.io/badge/arXiv-2509.24852-b31b1b.svg)](https://arxiv.org/abs/2509.24852)
[![Checkpoints](https://img.shields.io/badge/checkpoints-10.5281%2Fzenodo.21704323-1682d4.svg)](https://doi.org/10.5281/zenodo.21704323)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Method

A spike emitted by neuron *j* at time *t* reaches neuron *i* at *t* + 1 + d<sub>j</sub>. DelRec
learns d jointly with the synaptic weights, by surrogate gradient learning and backpropagation
through time.

An index into a buffer has no derivative. So d is relaxed to a real number, and each scheduled
spike is spread over neighboring integer time steps by a triangle function centered on 1 + d,
of half-width 1 + sigma. That function is differentiable in d, so the delay receives a gradient
of its own.

At sigma = 0 the triangle is a linear interpolation between the two closest integers. Where a
wider spread helps, sigma is annealed to 0 over training. 

| | delay parameter | shape |
|---|---|---|
| Axonal | one per presynaptic neuron, shared by its outgoing connections | `(N,)` |
| Synaptic | one per connection | `(N, N)` |

Either is learned jointly with the weights, or frozen at a rounded random initialization. The
frozen models are the control that isolates delay learning. All experiments use leaky
integrate-and-fire neurons.

## Install

```bash
git clone https://github.com/alexmaxad/DelRec.git && cd DelRec
pip install -r requirements.txt
```

Python 3.10 or newer. `requirements.txt` pins the environment that produced every number
here. `delrec` itself needs no install: every entry point puts the repository and `src/` on
the import path, so run the commands below from the repository root.
Triton is optional: the layers are plain PyTorch and run without it, only slower.
[docs/installation.md](docs/installation.md)

For the six-model HAR comparison (axonal, hybrid and synaptic delays on
feedforward or recurrent connections), see [docs/har_comparison.md](docs/har_comparison.md).
It includes the RTX 5060 Ti setup, run commands, parameter counts and saved plots.

## Reproduce the paper

Every figure and table, with no GPU, no dataset and no download:

```bash
python experiments/make_figures/make_all.py --measurements-only
python experiments/make_figures/make_all.py --list      # what each script produces and needs
```

The weights themselves (`best.pth`, 385 MB in total) are on Zenodo at
[10.5281/zenodo.21704323](https://doi.org/10.5281/zenodo.21704323). They are needed only to run
a model forward, which one figure and any recomputation of an analysis require:

```bash
python fetch_checkpoints.py --list
python fetch_checkpoints.py --dataset ssc
```

[docs/reproducing.md](docs/reproducing.md) · [docs/checkpoints.md](docs/checkpoints.md)

## Train

```bash
python experiments/train.py --dataset ssc --model SNN_recurrent_delays --seeds 0,1,2,3,4
python experiments/train.py --dataset al --smoke                 # 2 epochs on a few batches
python experiments/sweeps/parallel_sweep_HAR_delaystd.py         # one process per run per GPU
python experiments/profiling/profile_kernels.py                  # kernel time and memory
```

| model | delays |
|---|---|
| `SNN_recurrent_delays` | axonal, learned |
| `SNN_fixed_recurrent_delays` | axonal, fixed |
| `SNN_synaptic_recurrent_delays` | synaptic, learned |
| `SNN_fixed_synaptic_recurrent_delays` | synaptic, fixed |
| `SNN_common_recurrent_delays` | one delay shared by a whole layer |
| `SNN_vanilla_recurrent` | none, the delay-free control |

Those are the classes of `src/delrec/networks.py`. SHD has its own zoo, `networks_shd.py`, with
only the axonal and vanilla ones.

Hyperparameters are in `configs/perf_<BENCHMARK>.py`, one plain class per benchmark. Edit those
rather than the entry point. [docs/training.md](docs/training.md)

## Datasets

Needed only to train, or to recompute an analysis. Place under `Datasets/` at the repository
root. Paths and preprocessing in [docs/installation.md](docs/installation.md).

| Benchmark | Task | Source |
|---|---|---|
| SSC | Spiking Speech Commands, 35 classes | [zenkelab.org/datasets](https://zenkelab.org/datasets), place the three `.h5` files manually |
| PS-MNIST | permuted sequential MNIST | torchvision, downloaded on first use |
| AL | Autonomous Localization, 2 classes | generated on first use, following the Neuromorphic Sequential Arena |
| HAR | WISDM smartwatch activity, 18 classes | [WISDM 2019](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset), extract manually |
| Mackey-Glass | chaotic time-series forecasting | integrated in memory, nothing to download |
| SHD | Spiking Heidelberg Digits, 20 classes | downloaded on first use |

SSC and SHD are from Cramer et al. (2022), AL and HAR follow the pipeline and splits of Chen et
al. (2025), Mackey-Glass follows Jaeger and Haas (2004).

## File tree

```text
DelRec/
├── src/delrec/                  the library
│   ├── delay_layers.py          recurrent layers, parameters, forward dispatch
│   ├── delay_layers_pytorch.py  plain PyTorch implementation, the reference
│   ├── delay_layers_triton.py   the same as fused GPU kernels, optional
│   ├── triton_kernels/          the raw Triton kernels the latter calls
│   ├── networks.py              architectures for SSC, PS-MNIST, HAR, AL, Mackey-Glass
│   ├── networks_shd.py          architectures for SHD
│   ├── datasets/                loaders for the six benchmarks
│   ├── training/                one train and test loop per benchmark
│   └── utils.py                 seeding, state reset, losses, metrics
│
├── configs/                     hyperparameters, one module per benchmark
├── experiments/
│   ├── train.py                 train one model on one benchmark
│   ├── sweeps/                  multi-run sweeps, each with a parallel launcher
│   ├── profiling/               kernel time and memory benchmarks
│   └── make_figures/            one script per result, plus make_all.py
│
├── trained_models/              one directory per training run, 691 in total
├── figures/                     the generated figures and tables
│   ├── measurements/            cached results of the analyses that need a GPU
│   └── tables/                  the accuracy and significance tables
│
├── docs/                        installation, training, reproducing, architecture, checkpoints
└── fetch_checkpoints.py         download the weights
```

How those pieces fit together: [docs/architecture.md](docs/architecture.md).

For training-set-only random-label memorization with recurrent and feedforward
delays, see [docs/memorization.md](docs/memorization.md).

## Citation

```bibtex
@article{queant2025delrec,
  title         = {DelRec: learning delays in recurrent spiking neural networks},
  author        = {Queant, Alexandre and Ran{\c c}on, Ulysse and Cottereau, Benoit R
                   and Masquelier, Timoth{\'e}e},
  journal       = {arXiv preprint arXiv:2509.24852},
  year          = {2025},
  eprint        = {2509.24852},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2509.24852}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff). To cite the trained weights
themselves, use [10.5281/zenodo.21704323](https://doi.org/10.5281/zenodo.21704323), which
resolves to the latest version of the deposit.

## Acknowledgements

Built on [SpikingJelly](https://github.com/fangwei123456/spikingjelly). Supported by the French
Defense Innovation Agency under grant 2023 65 0082 and by the Agence Nationale de la Recherche
under grant ANR-20-CE45-0005 BRAIN-Net.

## License

MIT, see [LICENSE](LICENSE).
