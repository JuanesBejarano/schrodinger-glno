"""Stage-3 training of the GLNO on many shapes: the data of ``ShapeStore``, one process, resumable.

The optimizer is the paper's (AdamW, weight decay 1e-4, learning rate 1e-3 halved every 100
epochs, 300 epochs, batches of 4) and the loss the mass-weighted relative L2 error over the saved
times after t = 0. What stage 3 adds:

- every batch holds packets of one family (``shapes_data.family_batches``) and, while training,
  its basis gets random signs and rotations inside the eigenvalue groups (``augment.py``);
  validation uses the canonical basis;
- a step whose loss or gradient is not finite is dropped without touching the weights or the
  optimizer, and counted; more than ``max_skipped`` of an epoch stops the run;
- the gradient is clipped at ``grad_clip`` times the median of its recent norms (the step-4 trial
  saw isolated spikes of up to 280 times the median);
- ``best.pt`` is chosen on ``val_select``, the validation error without the tori of revolution
  (their validation copies are near-copies of training shapes).

``run_dir`` holds ``last.pt`` (every epoch), ``last_prev.pt`` (the one before, used when
``last.pt`` cannot be read: a file cut short by Drive), ``best.pt``, ``epoch_<n>.pt`` for the
epochs in ``snapshots`` and ``history.json``. A resume checks that ``best.pt`` is the checkpoint
the state names; when it is not (Drive kept an older one, or it cannot be read) a line says so
and ``best.pt`` becomes the best from that resume on. Every checkpoint carries the run's meta (dataset,
preprocessing, network and settings) and, beside it, the fingerprint of the Ritz basis. Resuming
asks for the same meta, compared with ``==``, and for a compatible fingerprint, compared with
``shapes_data.fingerprint_mismatch`` (its last digits change with the CPU). ``epochs``,
``grad_clip``, ``max_skipped`` and ``snapshots`` are not part of the meta: asking for more epochs
continues the run, and the clip can be switched on to rescue a run that keeps failing on the same
batches. The batches, the gauges and the clip threshold depend only on ``(seed, epoch, step)`` and
on what the checkpoint keeps, so a resumed run repeats an uninterrupted one. One process per
``run_dir``: two sessions writing the same run would mix their checkpoints.
"""
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import reprlib
import statistics
import time
from typing import Callable
import numpy as np
from numpy.typing import NDArray
import torch
from .augment import augment_batch, gauge_generator, tf32_matmul_enabled
from .data import glno_config, mass_weighted_relative_l2
from .layers import GLNONet
from .shapes_data import ShapesPreprocess, ShapeStore, family_batches, fingerprint_mismatch, group_means

__all__ = ["VARIANTS", "ShapesTrainSettings", "shapes_glno_config", "check_config", "run_meta", "train_step",
           "clip_threshold", "packet_errors", "evaluate_rows", "validation_summary", "train_shapes",
           "load_checkpoint", "checkpoint_preprocess", "predict_shape", "calibrate", "ieee_matmuls"]

TRAINER_VERSION = 1
# The two networks of the plan (D2). "full" is the paper's layer, with the gaussian_factor that did best in
# the step-4 trial; "periodic" keeps only the branch Φ diag(H(λ)) Φᵀ M, which does not see the signs of the basis.
VARIANTS = {"periodic": {"aperiodic": False},
            "full": {"aperiodic": True, "gaussian_factor": 50.0}}
GRAD_WINDOW = 200       # recent gradient norms kept for the clip threshold: one epoch of data/shapes
GRAD_WARMUP = 20        # no clipping before this many norms: the median of a handful says little
LAST, LAST_PREVIOUS, BEST, HISTORY = "last.pt", "last_prev.pt", "best.pt", "history.json"
_UNTRACKED_SETTINGS = ("epochs", "grad_clip", "max_skipped", "snapshots")
_LAST_KEYS = {"model", "optimizer", "scheduler", "history", "best_val", "grad_norms", "epoch", "meta", "fingerprint"}


@dataclass(frozen=True)
class ShapesTrainSettings:
    epochs: int = 300                   # Table 5 (Poisson)
    batch_size: int = 4                 # 200 steps per epoch on data/shapes
    lr: float = 1e-3                    # Table 5 (Poisson)
    weight_decay: float = 1e-4          # AdamW, as in the repository's train.py
    step_size: int = 100                # the paper halves the learning rate every 100 epochs
    gamma: float = 0.5
    seed: int = 0
    augment: bool = True                # random signs and rotations of the basis while training
    grad_clip: float | None = 10.0      # times the median of the recent gradient norms; None: no clipping
    max_skipped: float = 0.02           # fraction of an epoch's steps that may be dropped as not finite
    snapshots: tuple[int, ...] = (100, 200)     # epochs kept apart: 100 full epochs at each learning rate

    def __post_init__(self) -> None:
        if self.epochs < 0 or self.batch_size < 1 or self.step_size < 1:
            raise ValueError("epochs must be >= 0, batch_size and step_size >= 1")
        if not (self.lr > 0 and self.weight_decay >= 0 and 0 < self.gamma <= 1):
            raise ValueError("lr must be > 0, weight_decay >= 0 and gamma in (0, 1]")
        if self.grad_clip is not None and not (math.isfinite(self.grad_clip) and self.grad_clip > 0):
            raise ValueError("grad_clip must be positive, or None")
        if not 0 <= self.max_skipped <= 1:
            raise ValueError("max_skipped is a fraction of an epoch")
        if any(epoch < 1 for epoch in self.snapshots):
            raise ValueError("snapshots are numbers of finished epochs, from 1")


def shapes_glno_config(store: ShapeStore, variant: str, device: str | torch.device | None = None,
                       **overrides) -> dict:
    """The network of stage 3 for the data of ``store``: the paper's (4 blocks, 64 channels, 1 pole), fed with
    ψ0 alone (``C_in = 2``, no coordinates) and giving the states after t = 0 (``C_out = store.c_out``).

    ``variant`` is a key of ``VARIANTS``. ``device`` (the store's by default) is where the network
    runs; ``overrides`` change ``glno_config`` keys or the two switches ``aperiodic`` and
    ``gaussian_stable`` (on here: the same weights as the repository's filter without its 0/0).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; there are {tuple(VARIANTS)}")
    options = {"C_in": 2, "gaussian_stable": True} | VARIANTS[variant] | overrides
    switches = {key: options.pop(key) for key in ("aperiodic", "gaussian_stable")}
    config = glno_config(C_out=store.c_out, k_eig=store.k_eig, device=str(device or store.device), **options)
    config.update(switches)
    check_config(config, store)
    return config


def check_config(config: dict, store: ShapeStore) -> None:
    """Refuse the network options that break on padded batches of many meshes.

    Batch, instance and group normalisations average over the vertices, the padded ones included;
    ``normalize_basis`` does the same to the basis (the padding is then exact only to ~4e-7). The
    eigenvalue groups of the store are computed over all its modes, so the network must use all
    of them. The rest needs mesh connectivity or coordinates, which a batch does not carry.
    """
    problems = [f"{key}={value!r}" for key, value in config.items()
                if "norm" in key and not key.startswith("normalize") and value not in (None, "layer")]
    required = {"connect_method": "mlp", "encoder": "mlp", "decoder": "mlp", "outputs_at": "vertices",
                "k_min": 0, "k_max": store.k_eig, "k_eig": store.k_eig, "C_in": 2, "C_out": store.c_out}
    problems += [f"{key}={config.get(key)!r} (must be {value!r})" for key, value in required.items()
                 if config.get(key, value) != value]
    problems += [f"{key}={config[key]!r}" for key in ("rotate", "normalize_pole", "geo_preprocess", "normalize_basis",
                                                      "learned_geo_feat", "high_fre") if config.get(key)]
    if config.get("gaussian", "expx2") not in ("expx2", "x2", "expx"):
        problems.append(f"gaussian={config['gaussian']!r}")
    if problems:
        raise ValueError("stage 3 cannot train a network with " + ", ".join(problems))


def run_meta(store: ShapeStore, config: dict, settings: ShapesTrainSettings) -> dict:
    """What a run is: a checkpoint is continued only by a call with the same meta.

    ``shapes`` are the ones that train and the ones that choose ``best.pt``; a store that also
    holds test shapes, or does not, is the same run.
    """
    loaded = {name: sorted(set(store.table["shape"][store.rows(sets=name)].tolist())) for name in ("train", "val")}
    return {"trainer_version": TRAINER_VERSION, "dataset": store.identity, "preprocess": asdict(store.preprocess),
            "shapes": loaded,
            "network": {key: value for key, value in config.items() if key != "device"},
            "settings": {key: value for key, value in asdict(settings).items() if key not in _UNTRACKED_SETTINGS}}


def ieee_matmuls() -> None:
    """Turn TF32 matmuls off on CUDA: they would round the rotated basis to ~1e-3 (``augment_batch`` refuses them)."""
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    if tf32_matmul_enabled():
        raise RuntimeError("could not turn TF32 matmuls off")


def train_step(model: GLNONet, optimizer: torch.optim.Optimizer, batch: dict,
               clip: float | None = None) -> tuple[float | None, float | None, bool]:
    """One optimisation step on ``batch``: the loss, the gradient norm before clipping, and whether it was clipped.

    When the loss or the gradient is not finite the step is dropped: the weights and the optimizer
    stay as they were and ``(None, None, False)`` comes back. A gradient longer than ``clip`` is
    scaled down to that norm.
    """
    loss = mass_weighted_relative_l2(model(batch), batch["labels"], batch["mass"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients]))
    loss, norm = loss.item(), norm.item()
    if not (math.isfinite(loss) and math.isfinite(norm)):
        optimizer.zero_grad(set_to_none=True)
        return None, None, False
    clipped = clip is not None and norm > clip
    if clipped:
        for gradient in gradients:
            gradient.mul_(clip / norm)
    optimizer.step()
    return loss, norm, clipped


def clip_threshold(norms, factor: float | None) -> float | None:
    """``factor`` times the median of the recent gradient norms, once there are ``GRAD_WARMUP`` of them."""
    if factor is None or len(norms) < GRAD_WARMUP:
        return None
    return factor * statistics.median(norms)


def packet_errors(pred: torch.Tensor, target: torch.Tensor, mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``‖pred - target‖_M / ‖target‖_M`` of (B, N, 2T) outputs: (B, T) at each saved time and (B,) over all
    of them. The second is the loss of each packet; padded vertices have mass 0 and do not count."""
    size, n, channels = pred.shape
    weights = mass.reshape(size, n, 1)
    missed = torch.sum(weights * (pred - target).reshape(size, n, channels // 2, 2).square().sum(-1), dim=1)
    norm = torch.sum(weights * target.reshape(size, n, channels // 2, 2).square().sum(-1), dim=1)
    return torch.sqrt(missed / norm), torch.sqrt(missed.sum(1) / norm.sum(1))


def _device(model: GLNONet) -> torch.device:
    return next(model.parameters()).device


def _batch(store: ShapeStore, rows, device: torch.device) -> dict:
    return store.batch(rows, None if device == store.device else device)


@torch.no_grad()
def evaluate_rows(model: GLNONet, store: ShapeStore, rows, batch_size: int = 4) -> dict:
    """Errors of ``model`` on the packets ``rows`` in the canonical basis: ``per_time`` (n, T) and ``total`` (n,),
    in the order of ``rows``.

    The batches come family by family, not in that order, so every result is filed by its packet
    number (``batch["rows"]``) and read back through ``rows``.
    """
    rows = store._checked(rows, allow_empty=True)
    per_time, total = np.full((len(store), len(store.steps)), np.nan), np.full(len(store), np.nan)
    was_training, device = model.training, _device(model)
    model.eval()
    try:
        for members in family_batches(store, rows, batch_size):
            batch = _batch(store, members, device)
            each, whole = packet_errors(model(batch), batch["labels"], batch["mass"])
            filed = batch["rows"].numpy()
            per_time[filed], total[filed] = each.double().cpu().numpy(), whole.double().cpu().numpy()
    finally:
        model.train(was_training)
    return {"per_time": per_time[rows], "total": total[rows]}


def validation_summary(store: ShapeStore, rows, errors: dict) -> dict:
    """Means of ``evaluate_rows`` by reporting group, as plain numbers: ``total`` and ``per_time`` for each of
    ``shapes_data.GROUPS`` (``None`` for a group without packets)."""
    groups = store.group_positions(rows)
    total, per_time = group_means(errors["total"], groups), group_means(errors["per_time"], groups)
    return {"total": {name: None if value is None else float(value) for name, value in total.items()},
            "per_time": {name: None if value is None else value.tolist() for name, value in per_time.items()}}


def _save(state: dict, path: Path) -> None:
    """Write to a temporary file first, so an interrupted save never leaves a broken checkpoint."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _save_last(state: dict, run_dir: Path) -> None:
    """The new ``last.pt`` is complete on disk before the old one becomes ``last_prev.pt``."""
    last, temporary = run_dir / LAST, run_dir / "last.tmp"
    torch.save(state, temporary)
    if last.exists():
        os.replace(last, run_dir / LAST_PREVIOUS)
    os.replace(temporary, last)


def _differences(saved: dict, asked: dict) -> list[str]:
    """The entries of the meta that differ, named one by one."""
    found, absent = [], "<absent>"
    for key, value in asked.items():
        old = saved.get(key)
        if old == value:
            continue
        if isinstance(old, dict) and isinstance(value, dict):
            found += [f"{key}.{name} ({reprlib.repr(old.get(name, absent))} in the checkpoint, "
                      f"{reprlib.repr(value.get(name, absent))} now)"
                      for name in sorted(set(old) | set(value)) if old.get(name, absent) != value.get(name, absent)]
        else:
            found.append(f"{key} ({reprlib.repr(old)} in the checkpoint, {reprlib.repr(value)} now)")
    return found


def _checked_best(run_dir: Path, state: dict, log: Callable[[str], None]) -> float:
    """``best_val`` of the state when ``best.pt`` is the checkpoint it names; otherwise ``inf``, so that the
    next epoch writes a ``best.pt`` that agrees with the history from here on."""
    best = state["best_val"]
    if not math.isfinite(best):
        return best
    try:
        held = torch.load(run_dir / BEST, map_location="cpu", weights_only=False)
        if held["epoch"] <= state["epoch"] and held["record"]["val_select"] == best:
            return best
        problem = f"is the end of epoch {held['epoch'] + 1} with val_select {held['record']['val_select']:.6g}"
    except Exception as error:
        problem = f"cannot be read ({type(error).__name__})"
    log(f"{BEST} {problem}, not the best of this state (val_select {best:.6g}): the next epoch replaces it")
    return float("inf")


def _write_history(history: list[dict], run_dir: Path) -> None:
    temporary = run_dir / "history.tmp"
    temporary.write_text(json.dumps(history, indent=1) + "\n")
    os.replace(temporary, run_dir / HISTORY)


def _read_last(run_dir: Path, device: torch.device, log: Callable[[str], None]) -> tuple[dict | None, str | None]:
    """The newest readable state of the run and the file it came from, or ``(None, None)`` for a new run; an
    unreadable one is never overwritten."""
    found = [run_dir / name for name in (LAST, LAST_PREVIOUS) if (run_dir / name).exists()]
    for path in found:
        try:
            state = torch.load(path, map_location=device, weights_only=False)
            if not (isinstance(state, dict) and _LAST_KEYS <= set(state)):
                raise ValueError("it is not a checkpoint of this trainer")
        except Exception as error:      # a file cut short by Drive fails in pickle, zipfile or torch
            log(f"{path.name} cannot be read ({type(error).__name__}: {error})")
            continue
        if path.name != LAST:
            log(f"continuing from {path.name}: one epoch is repeated")
        return state, path.name
    if found:
        raise ValueError(f"{run_dir}: {' and '.join(path.name for path in found)} cannot be read; "
                         f"use a new run_dir to start again ({BEST} and the snapshots here still hold weights)")
    if (run_dir / HISTORY).exists():
        raise ValueError(f"{run_dir} holds the {HISTORY} of a run but neither {LAST} nor {LAST_PREVIOUS} "
                         "(is Drive mounted and complete?); use a new run_dir to start again")
    return None, None


def _kept(epoch: int) -> str:
    """What is on disk when the run stops inside the epoch of index ``epoch``."""
    return f"{LAST} is the end of epoch {epoch}" if epoch else "no checkpoint was written yet"


def _epoch_seed(seed: int, epoch: int) -> int:
    return int(np.random.SeedSequence([seed, epoch, 1]).generate_state(1, dtype=np.uint64)[0])


def train_shapes(store: ShapeStore, run_dir: Path, config: dict,
                 settings: ShapesTrainSettings = ShapesTrainSettings(),
                 log: Callable[[str], None] = print) -> list[dict]:
    """Train (or resume) the network ``config`` on the ``train`` packets of ``store``; returns the history.

    ``config`` comes from ``shapes_glno_config`` and names the device. ``store`` must hold the
    ``train`` and ``val`` sets; with the four sets, the fingerprint in the checkpoints covers
    every shape an evaluation will load.
    """
    run_dir = Path(run_dir)
    check_config(config, store)
    device = torch.device(config["device"])
    if device.type == "cuda":
        ieee_matmuls()
    train_rows, val_rows = store.rows(sets="train"), store.rows(sets="val")
    if len(train_rows) == 0:
        raise ValueError("the store holds no training packets")
    if len(store.group_positions(val_rows)["select"]) == 0:
        raise ValueError("the store holds no validation packets outside the tori of revolution: "
                         "there is nothing to choose best.pt with")
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = run_dir / "write_test.tmp"          # a folder that cannot be written fails here, not after an epoch
    probe.write_bytes(b"run_dir can be written")
    probe.unlink()
    meta, fingerprint = run_meta(store, config, settings), store.fingerprint()
    torch.manual_seed(settings.seed)
    model = GLNONet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=settings.step_size, gamma=settings.gamma)
    history, best, start_epoch, norms = [], float("inf"), 0, deque(maxlen=GRAD_WINDOW)
    state, source = _read_last(run_dir, device, log)
    if state is not None:
        differ = _differences(state["meta"], meta)
        if differ:
            raise ValueError(f"{run_dir} holds a run with another {', '.join(differ)}; use a new run_dir")
        reason = fingerprint_mismatch(state["fingerprint"], fingerprint)
        if reason is not None:
            raise ValueError(f"{run_dir}: the Ritz basis built here is not the checkpoint's ({reason})")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, start_epoch = state["history"], _checked_best(run_dir, state, log), state["epoch"] + 1
        norms.extend(state["grad_norms"])
        if source != LAST and (run_dir / LAST).exists():    # the next save must not make it last_prev.pt
            os.replace(run_dir / LAST, run_dir / "last_unreadable.pt")
        _write_history(history, run_dir)                    # a cut between last.pt and history.json
        log(f"Resuming at epoch {start_epoch} (best val_select so far {state['best_val']:.4g})")
    else:
        log(f"New run in {run_dir}")
    log(f"device={device}  parameters={sum(p.numel() for p in model.parameters()):,}  "
        f"train={len(train_rows)}  val={len(val_rows)}  store={store.nbytes / 2**30:.2f} GiB")

    for epoch in range(start_epoch, settings.epochs):
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(_epoch_seed(settings.seed, epoch))        # dropout, if any, repeats after a resume
        model.train()
        batches = family_batches(store, train_rows, settings.batch_size, seed=settings.seed, epoch=epoch)
        total, samples, clipped, skipped, epoch_norms = 0.0, 0, 0, 0, []
        for step, members in enumerate(batches):
            batch = _batch(store, members, device)
            if settings.augment:
                batch = augment_batch(batch, gauge_generator(settings.seed, epoch, step))
            loss, norm, was_clipped = train_step(model, optimizer, batch, clip_threshold(norms, settings.grad_clip))
            if loss is None:
                skipped += 1
                if skipped > settings.max_skipped * len(batches):
                    raise RuntimeError(
                        f"epoch {epoch + 1}: {skipped} of {len(batches)} steps had a loss or a gradient that is "
                        f"not finite (allowed: {settings.max_skipped:.0%}). The weights were not touched and "
                        f"{_kept(epoch)}; grad_clip can be changed without starting again")
                continue
            norms.append(norm)
            epoch_norms.append(norm)
            total, samples, clipped = total + loss * len(members), samples + len(members), clipped + was_clipped
        train_seconds = time.perf_counter() - start
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        summary = validation_summary(store, val_rows, evaluate_rows(model, store, val_rows, settings.batch_size))
        val_select = summary["total"]["select"]
        record = {"epoch": epoch, "train_loss": total / samples if samples else None, "val_select": val_select,
                  "val_round": summary["total"]["round"], "val_sphere": summary["total"]["sphere"],
                  "val_torus": summary["total"]["torus"], "val_per_time": summary["per_time"],
                  "lr": lr, "steps": len(batches), "skipped": skipped,
                  "clipped": clipped, "grad_clip": settings.grad_clip,
                  "grad_norm_median": statistics.median(epoch_norms) if epoch_norms else None,
                  "grad_norm_max": max(epoch_norms) if epoch_norms else None,
                  "ms_per_sample": round(1000 * train_seconds / len(train_rows), 2),
                  "seconds": round(time.perf_counter() - start, 1)}
        if device.type == "cuda":
            record["reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 2)
        history.append(record)
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise RuntimeError(f"epoch {epoch + 1}: the weights are not finite; nothing was saved and "
                               f"{_kept(epoch)}")
        checkpoint = {"model": model.state_dict(), "config": config, "meta": meta, "fingerprint": fingerprint,
                      "epoch": epoch, "record": record, "torch": str(torch.__version__)}
        if not math.isfinite(val_select):
            log(f"epoch {epoch + 1}: val_select is not finite; {BEST} is not updated")
        if val_select < best:
            best = val_select
            _save(checkpoint, run_dir / BEST)
        if epoch + 1 in settings.snapshots:
            _save(checkpoint, run_dir / f"epoch_{epoch + 1}.pt")
        _save_last(checkpoint | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                 "history": history, "best_val": best, "grad_norms": list(norms)}, run_dir)
        _write_history(history, run_dir)
        trained = "none" if record["train_loss"] is None else f"{record['train_loss']:.4f}"
        log(f"epoch {epoch + 1}/{settings.epochs}  train {trained}  val_select {val_select:.4f}  "
            f"lr {record['lr']:.1e}  skipped {skipped}  clipped {clipped}  {record['seconds']} s")
    return history


def load_checkpoint(path: Path, device: str | torch.device | None = None) -> tuple[GLNONet, dict]:
    """The network of a checkpoint, in evaluation mode on ``device``, and everything else the file holds."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(Path(path), map_location=device, weights_only=False)
    model = GLNONet({**state["config"], "device": str(device)}).to(device)
    model.load_state_dict(state.pop("model"))
    return model.eval(), state


def checkpoint_preprocess(state: dict) -> ShapesPreprocess:
    """The preprocessing a checkpoint was trained with (``geo_scale`` and the scales are not recomputed)."""
    return ShapesPreprocess(**state["meta"]["preprocess"])


@torch.no_grad()
def predict_shape(model: GLNONet, store: ShapeStore, shape: int, *, sets=None,
                  batch_size: int = 4) -> tuple[NDArray, NDArray]:
    """Packet numbers and predicted states of the loaded packets of one shape, as the dataset stores them:
    complex (P, N, T) at ``store.config.output_times()``, without padding or amplitude.

    When the labels leave t = 0 out, the first state is the input ψ0.
    """
    if isinstance(shape, (bool, np.bool_)) or not isinstance(shape, (int, np.integer)):
        raise ValueError("predict_shape takes one shape index")       # a list would be cut to its first mesh
    rows = store.rows(sets=sets, shapes=shape)
    if len(rows) == 0:
        raise ValueError(f"the store holds no packets of shape {shape}")
    device, amplitude, was_training = _device(model), store.preprocess.amplitude, model.training
    model.eval()
    try:
        states = []
        for start in range(0, len(rows), batch_size):
            batch = _batch(store, rows[start:start + batch_size], device)
            n = int(batch["n_verts"][0])
            out = model(batch)[:, :n].double().reshape(len(batch["rows"]), n, len(store.steps), 2)
            if store.preprocess.drop_initial:
                out = torch.cat([batch["input"][:, :n].double().unsqueeze(2), out], dim=2)
            states.append(torch.complex(out[..., 0], out[..., 1]).cpu().numpy() / amplitude)
    finally:
        model.train(was_training)
    return rows, np.concatenate(states)


def calibrate(store: ShapeStore, config: dict, settings: ShapesTrainSettings = ShapesTrainSettings(), *,
              warmup: int = 3, steps: int = 10, epochs: tuple[int, ...] = (50, 300)) -> dict:
    """Milliseconds per sample and reserved memory by family, measured on real batches, and the hours that
    ``epochs`` epochs would take.

    A throwaway network runs ``warmup`` training steps and then ``steps`` timed ones on batches of
    each family, and one timed pass over validation batches; nothing is saved and the run that
    follows is not affected (``train_shapes`` seeds itself). On CUDA the clock starts and stops on
    a synchronised device. ``reserved_gib`` is the peak PyTorch reserved on the device during that
    family's steps, the store included when it lives there (subtract ``store.nbytes`` for the cost
    of the step alone); the same holds for the ``reserved_gib`` of each epoch in the history.
    """
    check_config(config, store)
    device = torch.device(config["device"])
    cuda = device.type == "cuda"
    if cuda:
        ieee_matmuls()
    torch.manual_seed(settings.seed)
    model = GLNONet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)

    def clock() -> float:
        if cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    families, epoch_seconds = {}, 0.0
    for family in store.n_max:
        train_rows = store.rows(sets="train", family=family)
        val_rows = store.rows(sets="val", family=family)
        if len(train_rows) == 0:
            continue
        if cuda:                # without the blocks cached for the other family, so each one's peak is its own
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        batches = family_batches(store, train_rows, settings.batch_size, seed=settings.seed)
        model.train()
        seconds = samples = 0
        for step in range(warmup + steps):
            members = batches[step % len(batches)]
            start = clock()
            batch = _batch(store, members, device)
            if settings.augment:
                batch = augment_batch(batch, gauge_generator(settings.seed, 0, step))
            train_step(model, optimizer, batch)
            if step >= warmup:
                seconds, samples = seconds + clock() - start, samples + len(members)
        measured = {"train_ms_per_sample": 1000 * seconds / samples, "train_packets": len(train_rows),
                    "val_packets": len(val_rows), "eval_ms_per_sample": None,
                    "reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if cuda else None}
        epoch_seconds += seconds / samples * len(train_rows)
        if len(val_rows):
            timed = val_rows[:settings.batch_size * max(1, steps // 2)]
            evaluate_rows(model, store, timed[:settings.batch_size], settings.batch_size)       # warm-up
            start = clock()
            evaluate_rows(model, store, timed, settings.batch_size)
            per_sample = (clock() - start) / len(timed)
            measured["eval_ms_per_sample"] = 1000 * per_sample
            epoch_seconds += per_sample * len(val_rows)
        families[family] = measured
    return {"device": str(device), "batch_size": settings.batch_size, "families": families,
            "epoch_seconds": epoch_seconds,
            "hours": {str(count): count * epoch_seconds / 3600 for count in epochs}}
