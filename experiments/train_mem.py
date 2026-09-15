"""Train on fixed random labels and plot training loss and final training accuracy.

python experiments/train_mem.py
python experiments/train_mem.py --task-type spatial --hidden-layers 64,64 --seed 1
python experiments/train_mem.py --model SNN_vanilla_recurrent --num-samples 256
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "exp" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from configs.perf_MEM import Config
from delrec import networks
from delrec.datasets.spike_memorization import SpikeMemorization
from delrec.delay_layers import axonal_recdel
from delrec.training.delay_diagnostics import DelayDiagnostics
from delrec.training.mem import make_optimizer, run_epoch, set_epoch
from delrec.utils import seed_everything


def plot_results(history, final, config, run_dir):
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    loss_ax.plot([r["epoch"] for r in history], [r["loss"] for r in history])
    loss_ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training set (after each epoch)")
    loss_ax.grid(alpha=0.25)
    accuracy = final["accuracy_percent"]
    acc_ax.bar(["Final training accuracy"], [accuracy], width=0.45)
    acc_ax.set(ylabel="Accuracy (%)", ylim=(0, 110))
    acc_ax.axhline(100 / config.output_size, color="gray", linestyle="--", label="Uniform random guess")
    acc_ax.text(0, accuracy + 2, f"{accuracy:.2f}% ({final['correct']}/{final['num_samples']})", ha="center")
    acc_ax.legend(loc="lower right")
    fig.suptitle(f"{config.model}\n{config.task_type}, {config.num_samples} samples, seed {config.seed}")
    fig.savefig(run_dir / "training_summary.png", dpi=180)
    fig.savefig(run_dir / "training_summary.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[name for name in vars(networks) if name.startswith("SNN")])
    parser.add_argument("--task-type", choices=["temporal", "spatial"])
    for name in ("epochs", "num-samples", "batch-size", "seed", "dataset-seed",
                 "input-size", "time-window", "output-size", "cpu-threads"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--hidden-layers", help="Comma-separated widths, e.g. 64,64")
    parser.add_argument("--readout", choices=["mean", "sum", "last"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = Config()
    for key, value in vars(args).items():
        if key not in ("device", "out", "hidden_layers") and value is not None:
            setattr(config, key, value)
    if args.hidden_layers:
        config.hidden_layers = [int(n) for n in args.hidden_layers.split(",")]
    if min(config.epochs, config.batch_size, config.cpu_threads, *config.hidden_layers) < 1:
        parser.error("Epochs, batch size, CPU threads and hidden widths must be positive")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    run(config, device, args.out)


def run(config, device, out=None, model=None):
    """Run one complete experiment, optionally with a preinitialized model."""
    torch.set_num_threads(config.cpu_threads)
    seed_everything(config.seed, is_cuda=torch.cuda.is_available())
    dataset = SpikeMemorization(config.num_samples, config.input_size, config.time_window,
                               config.output_size, config.dataset_seed, config.input_gain, config.task_type)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed), num_workers=config.num_workers)
    # Same dataset, deterministic order: this is training-set measurement, not a split.
    measure_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers)
    model = (getattr(networks, config.model)(config) if model is None else model).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    run_dir = out or ROOT / "exp" / "MEM" / config.model / (
        f"{config.task_type}_seed{config.seed}_{datetime.now():%Y-%m-%d-%H-%M-%S-%f}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "config.json").exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    settings = {k: getattr(config, k) for k in dir(config)
                if not k.startswith("_") and not callable(getattr(config, k))}
    settings["device"] = str(device)
    settings["surrogate_function"] = repr(config.surrogate_function)
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2))
    torch.save({"inputs": dataset.inputs, "labels": dataset.labels}, run_dir / "dataset.pt")
    print(f"Device: {device}; model: {config.model}; output: {run_dir}", flush=True)
    diagnostics = DelayDiagnostics(model, run_dir,
                                   every=getattr(config, "delay_diagnostics_every", 10),
                                   show=False)
    diagnostics.snapshot()
    history = []
    with (run_dir / "train_res.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "accuracy_percent", "online_loss"])
        writer.writeheader()
        for epoch in range(config.epochs):
            set_epoch(model, config, epoch)
            diagnostics.begin_epoch(epoch + 1, config.epochs)
            online = run_epoch(loader, model, device, config, optimizer, diagnostics=diagnostics)
            diagnostics.snapshot()
            # Preserve fractional delays and the training-time smoothing for measurement.
            final = run_epoch(measure_loader, model, device, config)
            row = {"epoch": epoch + 1, "loss": final["loss"],
                   "accuracy_percent": final["accuracy_percent"], "online_loss": online["loss"]}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            scheduler.step()
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
                print(f"Epoch {epoch + 1}/{config.epochs}: loss={final['loss']:.6f}, "
                      f"training accuracy={final['accuracy_percent']:.2f}%", flush=True)
    final.update(epoch=config.epochs, model=config.model, seed=config.seed,
                 dataset_seed=config.dataset_seed, task_type=config.task_type,
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (run_dir / "final_train.json").write_text(json.dumps(final, indent=2))
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": config.epochs,
                "config": settings, "metrics": final,
                "recurrent_sigmas": {name: m.sigma for name, m in model.named_modules()
                                     if isinstance(m, axonal_recdel)}}, run_dir / "last.pth")
    diagnostics.plot_hidden_delay_evolution()
    plot_results(history, final, config, run_dir)
    print(f"Final training accuracy: {final['accuracy_percent']:.2f}%\nSaved: {run_dir}", flush=True)

    return history, final, run_dir


if __name__ == "__main__":
    main()
