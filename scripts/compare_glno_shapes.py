"""Step 10 of Plan_GLNO_etapa3.md: the trained GLNO against Crank–Nicolson on the test shapes, with figures.

Loads ``best.pt`` of each run in --runs (trained on Colab, copied to runs/), rebuilds the test shapes of
--data with the preprocessing of the checkpoint (the Ritz basis is computed here and its fingerprint must
match the one the checkpoint was trained with), and predicts every test and out-of-distribution packet.
The test errors of the first run must repeat the ``test.json`` of its run folder.

Writes in --out:

- ``summary.json`` and a printed table: mass-weighted relative L² error by run, set (test, ood), group
  (select = not a round torus, sphere, torus, round) and time, with the mean, the median and a 95 %
  interval from resampling shapes within each family (none for a group of fewer than 5 shapes, such as
  the 2 round tori of the test); the error of predicting ψ0 as a reference; and the norm of the
  prediction over that of Crank–Nicolson, which is 1 at every time if the network conserves it.
- ``errors.png``: error against t of every run and set, the norm ratio against t, and the error of each
  packet against its k0.
- ``compare_<shape>_<set><packet>.png``: |ψ|² of Crank–Nicolson and of the GLNO on the surface, and
  their difference, at a few times, for a median sphere, a median deformed torus, a median round torus,
  the worst test packet and a median out-of-distribution torus (errors of the first run).

Only forward passes: on the local GPU or the CPU it takes a few minutes, most of them the Ritz bases.

    python scripts/compare_glno_shapes.py
"""
import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm
import numpy as np
import torch

from schrodinger_glno.glno.shapes_data import ShapeStore, family_batches, fingerprint_mismatch
from schrodinger_glno.glno.shapes_train import (checkpoint_preprocess, evaluate_rows, load_checkpoint, packet_errors,
                                                predict_shape)
from schrodinger_glno.mesh import vertex_normals
from schrodinger_glno.visualization import CMAP
from visualize_shapes_dataset import KINDS, surface_panel, view_direction

ROOT = Path(__file__).resolve().parents[1]
SETS = ("test", "ood")
GROUPS = ("select", "sphere", "torus", "round")
LABELS = {"shapes_glno_periodic_v1": "rama periódica", "shapes_glno_full_v1": "GLNO completo"}
COLOURS = {"shapes_glno_periodic_v1": "#2a78d6", "shapes_glno_full_v1": "#eb6834", "psi0": "#888888"}
FIGURE_TIMES = (0.0, 0.1, 0.25, 0.5)
RESAMPLES = 2000
MIN_SHAPES = 5                  # fewer shapes than this give no interval: resampling 2 shapes says nothing


def group_masks(store: ShapeStore, rows: np.ndarray) -> dict:
    family, round_ = store.table["family"][rows], store.table["round"][rows].astype(bool)
    sphere = family == 0
    return {"select": ~round_, "sphere": sphere, "torus": ~sphere & ~round_, "round": round_}


def interval(store: ShapeStore, rows: np.ndarray, values: np.ndarray, mask: np.ndarray, rng) -> list | None:
    """95 % interval of the mean of ``values[mask]``, resampling shapes with replacement within each family.

    None when the group has fewer than MIN_SHAPES shapes.
    """
    shapes = store.table["shape"][rows]
    if len(np.unique(shapes[mask])) < MIN_SHAPES:
        return None
    strata = []
    for code in (0, 1):
        members = [values[mask & (shapes == s)] for s in np.unique(shapes[mask & (store.table["family"][rows] == code)])]
        if members:
            strata.append(members)
    means = []
    for _ in range(RESAMPLES):
        drawn = [members[i] for members in strata for i in rng.integers(len(members), size=len(members))]
        means.append(np.concatenate(drawn).mean())
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(store: ShapeStore, rows: np.ndarray, totals: np.ndarray, per_time: np.ndarray, rng,
              ratios: np.ndarray | None = None) -> dict:
    out = {}
    for group, mask in group_masks(store, rows).items():
        if mask.any():
            shapes = len(np.unique(store.table["shape"][rows][mask]))
            out[group] = {"packets": int(mask.sum()), "shapes": shapes, "mean": float(totals[mask].mean()),
                          "median": float(np.median(totals[mask])), "ci95": interval(store, rows, totals, mask, rng),
                          "per_time": per_time[mask].mean(axis=0).tolist()}
            if ratios is not None:
                out[group]["norm_ratio"] = {"per_time": ratios[mask].mean(axis=0).tolist(),
                                            "min": float(ratios[mask].min()), "max": float(ratios[mask].max())}
    return out


@torch.no_grad()
def norm_ratios(model, store: ShapeStore, rows: np.ndarray, batch_size: int) -> np.ndarray:
    """``‖ψ_GLNO(t)‖ / ‖ψ_CN(t)‖`` of each packet at each saved time, (n, T) in the order of ``rows``.

    Both norms with the lumped mass of the store, the one of the errors. Crank–Nicolson conserves the
    norm, so a ratio of 1 at every time means that the network does too.
    """
    device = next(model.parameters()).device
    ratios = np.full((len(store), len(store.steps)), np.nan)
    for members in family_batches(store, rows, batch_size):
        batch = store.batch(members, None if device == store.device else device)
        predicted, labels = model(batch), batch["labels"]
        size, n, channels = predicted.shape
        weights = batch["mass"].reshape(size, n, 1)

        def norms(states: torch.Tensor) -> torch.Tensor:
            return torch.sum(weights * states.reshape(size, n, channels // 2, 2).square().sum(-1), dim=1).sqrt()

        ratios[batch["rows"].numpy()] = (norms(predicted) / norms(labels)).double().cpu().numpy()
    return ratios[rows]


def psi0_errors(store: ShapeStore, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Errors of predicting ψ(t) = ψ0 at every saved time."""
    per_time, totals = [], []
    for row in rows:                                                    # one packet at a time: a batch is one family
        batch = store.batch([row])
        labels = batch["labels"]
        each, total = packet_errors(batch["input"].repeat(1, 1, labels.shape[-1] // 2), labels, batch["mass"])
        per_time.append(each.numpy()[0]), totals.append(total.numpy()[0])
    return np.array(totals), np.array(per_time)


def errors_figure(results: dict, reference: dict, store: ShapeStore, rows: dict, out: Path) -> Path:
    times = store.times
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(19.5, 4.8), layout="constrained")
    for run, result in results.items():
        for name, style in (("test", "-"), ("ood", "--")):
            curve = result[name]["summary"]["select"]["per_time"]
            left.plot(times, np.array(curve) * 100, style, color=COLOURS[run], marker="o", markersize=4,
                      label=f"{LABELS.get(run, run)}, {name}")
    for name, style in (("test", "-"), ("ood", "--")):
        left.plot(times, np.array(reference[name]["select"]["per_time"]) * 100, style, color=COLOURS["psi0"],
                  label=f"predecir ψ0, {name}")
    left.set(yscale="log", xlabel="t", ylabel="error relativo L² (%)",
             title="Error contra CN por tiempo (formas de test sin toros de revolución)")
    left.grid(alpha=0.3, which="both")
    left.legend(fontsize=8, frameon=False)
    for run, result in results.items():
        for name, style in (("test", "-"), ("ood", "--")):
            ratios = (result[name]["ratios"][group_masks(store, rows[name])["select"]] - 1) * 100
            middle.plot(times, ratios.mean(axis=0), style, color=COLOURS[run], marker="o", markersize=4,
                        label=f"{LABELS.get(run, run)}, {name}")
            middle.fill_between(times, *np.percentile(ratios, [5, 95], axis=0), color=COLOURS[run], alpha=0.12, lw=0)
    middle.axhline(0, color="#444444", lw=1)
    middle.set(xlabel="t", ylabel="‖ψ_GLNO‖ / ‖ψ_CN‖ − 1 (%)",
               title="Norma de la predicción contra la de CN (media; banda: 5–95 % de los paquetes)")
    middle.grid(alpha=0.3)
    middle.legend(fontsize=8, frameon=False)
    for run, result in results.items():
        for name, marker in (("test", "o"), ("ood", "^")):
            k0 = store.params[rows[name], 2]
            right.scatter(k0, result[name]["totals"] * 100, s=14, marker=marker, color=COLOURS[run], alpha=0.6,
                          label=f"{LABELS.get(run, run)}, {name}")
    right.set(yscale="log", xlabel="k0 del paquete", ylabel="error relativo L² (%)",
              title="Error de cada paquete contra su k0 (entrenamiento: k0 en [1, 6])")
    right.axvline(6.0, color="#444444", lw=1, ls=":")
    right.grid(alpha=0.3, which="both")
    right.legend(fontsize=8, frameon=False)
    path = out / "errors.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def surface_comparison(verts: np.ndarray, faces: np.ndarray, saved: np.ndarray, cn: np.ndarray, glno: np.ndarray, *,
                       torus: bool, run: str, title: str, path: Path) -> Path:
    """CN, GLNO and their difference on the surface, one column per time in FIGURE_TIMES.

    ``cn`` and ``glno`` are complex (N, T) at the times ``saved``; ``torus`` picks the camera rule.
    """
    columns = [int(np.argmin(np.abs(saved - t))) for t in FIGURE_TIMES]
    normals = vertex_normals(verts, faces)
    density_norm = PowerNorm(gamma=0.5, vmin=0, vmax=float(np.abs(cn[:, 0]).max() ** 2), clip=True)
    difference = np.abs(glno - cn) / np.abs(cn).max(axis=0, keepdims=True) * 100      # % of the peak of |ψ_CN(t)|
    error_norm = Normalize(vmin=0, vmax=max(float(difference[:, columns].max()), 1e-6))
    fig = plt.figure(figsize=(2.9 * len(columns) + 1.6, 8.6), layout="constrained")
    grid = fig.add_gridspec(3, len(columns) + 2, width_ratios=[0.1] + [1] * len(columns) + [0.06])
    labels = ("Crank–Nicolson\n|ψ|²", f"GLNO ({LABELS.get(run, run)})\n|ψ|²", "|ψ_GLNO − ψ_CN|\n(% del máx. de |ψ_CN(t)|)")
    for i, label in enumerate(labels):                  # a column of its own, so the layout leaves room for it
        ax = fig.add_subplot(grid[i, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, label, rotation=90, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    for j, column in enumerate(columns):
        peak = int(np.argmax(np.abs(cn[:, column])))
        view = view_direction(normals[peak], verts[peak], torus)
        fields = (np.abs(cn[:, column]) ** 2, np.abs(glno[:, column]) ** 2, difference[:, column])
        for i, (field, norm) in enumerate(zip(fields, (density_norm, density_norm, error_norm))):
            ax = fig.add_subplot(grid[i, j + 1], projection="3d", computed_zorder=False)
            surface_panel(ax, verts, faces, field, norm, view)
            if i == 0:
                ax.set_title(f"t = {saved[column]:.3g}" + (" (la entrada ψ0)" if j == 0 else ""), fontsize=10)
    fig.colorbar(ScalarMappable(density_norm, CMAP), cax=fig.add_subplot(grid[:2, -1]), label="|ψ|²")
    fig.colorbar(ScalarMappable(error_norm, "magma"), cax=fig.add_subplot(grid[2, -1]), label="%")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def comparison_figure(store: ShapeStore, data: Path, model, run: str, row: int, error: float, out: Path) -> Path:
    """``surface_comparison`` of the packet ``row`` of the store."""
    shape, name = int(store.table["shape"][row]), ("train", "val", "test", "ood")[int(store.table["set"][row])]
    entry = next(s for s in store.shapes if s["index"] == shape)
    packet = int(store.table["packet"][row])
    with np.load(data / entry["id"] / "mesh.npz") as mesh:
        verts, faces, saved = mesh["verts"], mesh["faces"], mesh["t_out"]
    prefix = "ood_" if name == "ood" else ""
    cn = np.load(data / entry["id"] / f"{prefix}outputs.npy", mmap_mode="r")[packet].astype(float)
    cn = cn[..., 0] + 1j * cn[..., 1]                                               # (N, T)
    predicted_rows, states = predict_shape(model, store, shape, sets=name, batch_size=1)
    glno = states[list(predicted_rows).index(row)]                                  # (N, T)
    center, _, k0, sigma = store.params[row]
    title = (f"{entry['id']} ({KINDS[entry['family'], entry['round']]}), {name} {packet}: k0 = {k0:.2f}, "
             f"σ = {sigma:.3f}; error relativo L² sobre t > 0: {error * 100:.2f} %")
    return surface_comparison(verts, faces, saved, cn, glno, torus=entry["family"] == "torus", run=run, title=title,
                              path=out / f"compare_{entry['id']}_{name}{packet}.png")


def chosen_packets(store: ShapeStore, rows: dict, totals: dict) -> list[int]:
    """A median sphere, deformed torus and round torus of the test, the worst test packet, a median ood torus."""
    def median_of(name, mask):
        candidates = rows[name][mask]
        values = totals[name][mask]
        return int(candidates[np.argsort(values)[len(values) // 2]])

    test = group_masks(store, rows["test"])
    ood = group_masks(store, rows["ood"])
    worst = int(rows["test"][np.argmax(totals["test"])])
    picked = [median_of("test", test["sphere"]), median_of("test", test["torus"]), median_of("test", test["round"]),
              worst, median_of("ood", ood["torus"])]
    return list(dict.fromkeys(picked))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "shapes")
    parser.add_argument("--runs", type=Path, nargs="+",
                        default=[ROOT / "results" / "glno_schrodinger" / "runs" / "shapes_glno_periodic_v1", ROOT / "results" / "glno_schrodinger" / "runs" / "shapes_glno_full_v1"])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "glno_shapes_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    torch.exp(torch.zeros(8))                   # the first vector-math call of this torch build can be off on CPU
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    models = {}
    for run_dir in args.runs:
        models[run_dir.name] = load_checkpoint(run_dir / "best.pt", args.device)
    first_state = next(iter(models.values()))[1]
    preprocess = checkpoint_preprocess(first_state)
    for name, (_, state) in models.items():
        if checkpoint_preprocess(state) != preprocess:
            raise SystemExit(f"{name} was trained with another preprocessing than {args.runs[0].name}")
    store = ShapeStore(args.data, preprocess, sets=SETS, progress=lambda line: print(line, flush=True))
    if store.identity != first_state["meta"]["dataset"]:
        raise SystemExit(f"{args.data} is not the dataset the checkpoints were trained on")
    fingerprint = store.fingerprint()
    for name, (_, state) in models.items():
        reason = fingerprint_mismatch(state["fingerprint"], fingerprint)
        if reason is not None:
            raise SystemExit(f"the Ritz basis computed here is not the one {name} was trained with: {reason}")
    print(f"{len(store.shapes)} test shapes, {len(store)} packets, bases match the checkpoints "
          f"({time.perf_counter() - start:.0f} s)", flush=True)

    rng = np.random.default_rng(0)
    rows = {name: store.rows(sets=name) for name in SETS}
    results, reference = {}, {}
    for name in SETS:
        totals, per_time = psi0_errors(store, rows[name])
        reference[name] = summarize(store, rows[name], totals, per_time, rng)
    for run, (model, state) in models.items():
        results[run] = {"epoch": state["epoch"] + 1}
        for name in SETS:
            errors = evaluate_rows(model, store, rows[name], args.batch_size)
            ratios = norm_ratios(model, store, rows[name], args.batch_size)
            results[run][name] = {"totals": errors["total"], "per_time": errors["per_time"], "ratios": ratios,
                                  "summary": summarize(store, rows[name], errors["total"], errors["per_time"], rng,
                                                       ratios)}
        print(f"{run}: evaluated ({time.perf_counter() - start:.0f} s)", flush=True)

    # the first run's test errors must repeat the test.json written on Colab
    first_run = args.runs[0]
    if (first_run / "test.json").is_file():
        colab = json.loads((first_run / "test.json").read_text())
        here = results[first_run.name]["test"]["summary"]["select"]["mean"]
        there = colab["test"]["total"]["select"]
        print(f"test select here {here * 100:.4f} %, on Colab {there * 100:.4f} % (difference {abs(here - there):.1e})")
        if abs(here - there) > 1e-4 * there + 1e-6:
            raise SystemExit("the local evaluation does not repeat the one of Colab")

    print(f"\n{'run':<26}{'set':<6}{'group':<8}{'packets':>8}{'mean %':>9}{'median %':>10}{'95 % (shapes)':>18}"
          f"{'t=0.05 %':>10}{'t=0.5 %':>9}")
    for run, result in [*results.items(), ("predecir ψ0", reference)]:
        for name in SETS:
            summary = result[name]["summary"] if run in results else result[name]
            for group in GROUPS:
                if group in summary:
                    s = summary[group]
                    span = (f"{100 * s['ci95'][0]:.3f}–{100 * s['ci95'][1]:.3f}" if s["ci95"]
                            else f"— ({s['shapes']} formas)")
                    print(f"{LABELS.get(run, run):<26}{name:<6}{group:<8}{s['packets']:>8}{s['mean'] * 100:>9.3f}"
                          f"{s['median'] * 100:>10.3f}{span:>18}{s['per_time'][0] * 100:>10.3f}"
                          f"{s['per_time'][-1] * 100:>9.3f}")

    print(f"\n‖ψ_GLNO‖ / ‖ψ_CN‖ (1 = conserva la norma)\n{'run':<26}{'set':<6}{'group':<8}{'t=0.05':>9}{'t=0.25':>9}"
          f"{'t=0.5':>9}{'mín':>9}{'máx':>9}")
    for run, result in results.items():
        for name in SETS:
            for group in GROUPS:
                if group in result[name]["summary"]:
                    ratio = result[name]["summary"][group]["norm_ratio"]
                    curve = ratio["per_time"]
                    print(f"{LABELS.get(run, run):<26}{name:<6}{group:<8}{curve[0]:>9.4f}{curve[len(curve) // 2 - 1]:>9.4f}"
                          f"{curve[-1]:>9.4f}{ratio['min']:>9.4f}{ratio['max']:>9.4f}")

    paths = [errors_figure(results, reference, store, rows, args.out)]
    main_run = args.runs[0].name
    model = models[main_run][0]
    totals = {name: results[main_run][name]["totals"] for name in SETS}
    for row in chosen_packets(store, rows, totals):
        name = ("train", "val", "test", "ood")[int(store.table["set"][row])]
        error = float(totals[name][list(rows[name]).index(row)])
        paths.append(comparison_figure(store, args.data, model, main_run, row, error, args.out))
    summary = {"data": str(args.data), "identity": store.identity, "times": store.times.tolist(),
               "psi0": reference,
               "runs": {run: {"epoch": r["epoch"], **{name: r[name]["summary"] for name in SETS}}
                        for run, r in results.items()}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\nwrote", *(str(path) for path in [*paths, args.out / "summary.json"]), sep="\n  ")
    print(f"{time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
