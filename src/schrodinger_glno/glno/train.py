"""Training and evaluation of the adapted GLNO on the sphere or torus datasets.

The optimizer follows the paper's code (``train.py``): AdamW with weight decay
1e-4, learning rate 1e-3 halved every 100 epochs, 300 epochs (Table 5,
Poisson). The loss is the mass-weighted relative L2 error.

``run_dir`` holds ``last.pt`` (saved every epoch, used to resume after a
disconnection), ``best.pt`` (lowest validation loss) and ``history.json``.
Resuming checks that the dataset, the network and the settings are the same,
except ``epochs``: asking for more epochs continues the same run.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Callable
import numpy as np
import torch
from torch.utils.data import DataLoader
from schrodinger_glno.data_utils import dataset_identity
from .data import SurfaceSplit, glno_config, mass_weighted_relative_l2
from .layers import GLNONet


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 300          # Table 5 (Poisson)
    batch_size: int = 4        # GPU memory per sample: ~1.2 GiB sphere, ~3.3 GiB torus (25k vertices, k_eig 800)
    lr: float = 1e-3           # Table 5 (Poisson)
    weight_decay: float = 1e-4  # AdamW, as in the repository's train.py
    step_size: int = 100       # the paper halves the learning rate every 100 epochs
    gamma: float = 0.5
    seed: int = 0
    num_workers: int = 2


def _save(obj: dict, path: Path) -> None:
    """Write to a temporary file first, so an interrupted save never leaves a broken checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _loader(dataset: SurfaceSplit, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      collate_fn=dataset.collate, num_workers=num_workers)


@torch.no_grad()
def mean_loss(model: GLNONet, dataset: SurfaceSplit, batch_size: int, device: str, num_workers: int = 0) -> float:
    model.eval()
    total = 0.0
    for batch in _loader(dataset, batch_size, False, 0, num_workers):
        pred = model(batch)
        loss = mass_weighted_relative_l2(pred, batch["labels"].to(device), batch["mass"].to(device))
        total += loss.item() * len(batch["input"])
    return total / len(dataset)


def train_glno(data_dir: Path, run_dir: Path, settings: TrainSettings = TrainSettings(),
               glno_overrides: dict | None = None, device: str | None = None,
               log: Callable[[str], None] = print) -> list[dict]:
    """Train (or resume) and return the per-epoch history."""
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set = SurfaceSplit(data_dir, "train"), SurfaceSplit(data_dir, "val")
    config = glno_config(C_out=train_set.n_out, k_eig=train_set.mesh["evals"].shape[0], device=device,
                         **(glno_overrides or {}))
    # epochs and num_workers can change between sessions: more epochs extends the same run.
    meta = {"dataset_identity": dataset_identity(data_dir, verify=False),
            "glno_config": {k: v for k, v in config.items() if k != "device"},
            "settings": {k: v for k, v in asdict(settings).items() if k not in ("epochs", "num_workers")}}
    torch.manual_seed(settings.seed)
    model = GLNONet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=settings.step_size, gamma=settings.gamma)
    history, best_val, start_epoch = [], float("inf"), 0
    last = run_dir / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["meta"] != meta:
            raise ValueError(f"{run_dir} holds a run with another dataset, network or settings; use a new run_dir")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best_val, start_epoch = state["history"], state["best_val"], state["epoch"] + 1
        log(f"Resuming at epoch {start_epoch} (best validation loss so far {best_val:.4g})")
    log(f"device={device}  parameters={sum(p.numel() for p in model.parameters()):,}  "
        f"train={len(train_set)}  val={len(val_set)}")

    for epoch in range(start_epoch, settings.epochs):
        start = time.perf_counter()
        model.train()
        total = 0.0
        # A new seed per epoch: after resuming, the shuffling continues exactly as before.
        for batch in _loader(train_set, settings.batch_size, True, settings.seed * 100_003 + epoch,
                             settings.num_workers):
            pred = model(batch)
            loss = mass_weighted_relative_l2(pred, batch["labels"].to(device), batch["mass"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch["input"])
        scheduler.step()
        val = mean_loss(model, val_set, settings.batch_size, device, settings.num_workers)
        record = {"epoch": epoch, "train_loss": total / len(train_set), "val_loss": val,
                  "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.perf_counter() - start, 1)}
        if device.startswith("cuda"):
            record["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        history.append(record)
        if val < best_val:
            best_val = val
            _save({"model": model.state_dict(), "config": config, "meta": meta, "epoch": epoch,
                   "val_loss": val}, run_dir / "best.pt")
        _save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "history": history, "best_val": best_val,
               "epoch": epoch, "meta": meta}, last)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        log(f"epoch {epoch + 1}/{settings.epochs}  train {record['train_loss']:.4f}  "
            f"val {val:.4f}  lr {record['lr']:.1e}  {record['seconds']} s")
    return history


def load_model(run_dir: Path, checkpoint: str = "best.pt", device: str | None = None) -> GLNONet:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(Path(run_dir) / checkpoint, map_location=device, weights_only=False)
    model = GLNONet({**state["config"], "device": device}).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


@torch.no_grad()
def predict(model: GLNONet, dataset: SurfaceSplit, index: int) -> np.ndarray:
    """Complex trajectory (N, n_t) predicted for one sample."""
    batch = dataset.collate([dataset[index]])
    out = model(batch)[0].cpu().numpy().reshape(len(dataset.mesh["mass"]), -1, 2)
    return out[..., 0] + 1j * out[..., 1]


@torch.no_grad()
def evaluate_glno(data_dir: Path, run_dir: Path, split: str = "test", checkpoint: str = "best.pt",
                  device: str | None = None, batch_size: int = 4) -> dict:
    """Mass-weighted relative L2 per sample and per saved time, and ψᴴMψ of predictions and references.

    Both norms use the lumped mass of ``mesh.npz``. The data are normalised with the
    solver's mass, so the reference norm is slightly above 1 (up to ~1 % at high k0).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run_dir, checkpoint, device)
    dataset = SurfaceSplit(data_dir, split)
    mass = dataset.mesh["mass"].to(device)
    n_t = dataset.n_out // 2
    per_sample, per_time, norms, ref_norms = [], [], [], []
    for batch in _loader(dataset, batch_size, False, 0, 0):
        pred = model(batch).reshape(len(batch["input"]), len(mass), n_t, 2)
        target = batch["labels"].to(device).reshape(pred.shape)
        w = mass[None, :, None]
        diff = torch.sum(w * torch.sum((pred - target)**2, dim=-1), dim=1)       # (B, n_t)
        ref = torch.sum(w * torch.sum(target**2, dim=-1), dim=1)
        per_time.append(torch.sqrt(diff / ref).cpu())
        per_sample.append(torch.sqrt(diff.sum(1) / ref.sum(1)).cpu())
        norms.append(torch.sum(w * torch.sum(pred**2, dim=-1), dim=1).cpu())
        ref_norms.append(ref.cpu())
    per_sample, per_time = torch.cat(per_sample), torch.cat(per_time)
    norms, ref_norms = torch.cat(norms), torch.cat(ref_norms)
    return {"split": split, "samples": len(dataset), "times": dataset.times.tolist(),
            "relative_l2_mean": per_sample.mean().item(), "relative_l2_median": per_sample.median().item(),
            "relative_l2_per_time": per_time.mean(0).tolist(),
            "predicted_norm_per_time": norms.mean(0).tolist(),
            "reference_norm_per_time": ref_norms.mean(0).tolist()}
