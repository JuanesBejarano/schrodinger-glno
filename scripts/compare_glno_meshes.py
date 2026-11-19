"""The trained GLNO against Crank–Nicolson on the real meshes of generate_meshes_dataset.py, with figures.

Nothing is trained: ``best.pt`` of each run in --runs predicts every packet of every mesh folder of --data,
with the preprocessing of its checkpoint (the Ritz basis is computed here; the eigenvalue scale and the
``geo_scale`` are the ones of the training shapes). The network saw deformed spheres and tori only, so
this measures how far it carries to other surfaces.

Writes in --out:

- ``summary.json`` and a printed table, by run, mesh and set (test: k0 and σ in the training ranges; ood:
  larger k0): the mass-weighted relative L² error against Crank–Nicolson (mean, median, range, first and
  last saved time) and the norm of the prediction over that of Crank–Nicolson. Three references by mesh
  and set: the error of predicting ψ0; the spectral oracle, the Crank–Nicolson factor of each mode applied
  in the same 700-mode basis, which is what a network of this form could reach
  (``spectral.spectral_oracle_error``); and the time-stepping error of the data itself, estimated from dt
  against dt/2 when they were generated. The table ends with the errors on the stage-3 test shapes, read
  from --reference (the summary.json of compare_glno_shapes.py).
- ``errors.png``, for the first run: error against t by mesh and set, the norm ratio against t, and the
  error of each packet against its k0.
- ``compare_<mesh>_<set><packet>.png``: |ψ|² of Crank–Nicolson and of the GLNO on the surface, and their
  difference, at a few times, for the median and the worst test packet of each mesh (first run).

Only forward passes: a couple of minutes on the local GPU or the CPU, most of them the Ritz bases.

    python scripts/compare_glno_meshes.py
"""
import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from schrodinger_glno.glno.shapes_data import shape_arrays
from schrodinger_glno.glno.shapes_train import checkpoint_preprocess, load_checkpoint, packet_errors
from schrodinger_glno.glno.spectral import propagate
from schrodinger_glno.shapes_dataset import CONFIG_FILE, DIAGNOSTIC_COLUMNS, SHAPE_FILE, ShapesConfig
from compare_glno_shapes import LABELS, surface_comparison

ROOT = Path(__file__).resolve().parents[1]
SETS = ("test", "ood")
PREFIX = {"test": "", "ood": "ood_"}
COLOURS = ("#2a78d6", "#2e9e6b", "#eb6834", "#8e5bc9", "#b08d1e")       # one per mesh, in the order of --data
UPRIGHT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # a rotation: these meshes have y up, the figures z


def load_mesh(folder: Path, config: ShapesConfig, preprocess) -> dict:
    """What the network sees of the mesh in ``folder`` (``shape_arrays``) and its packets, set by set."""
    record = json.loads((folder / SHAPE_FILE).read_text())
    with np.load(folder / "mesh.npz") as stored:
        verts, faces = stored["verts"], stored["faces"]
    sets = {name: {"outputs": np.load(folder / f"{PREFIX[name]}outputs.npy"),               # (n, N, T, 2), t = 0 first
                   "params": np.load(folder / f"{PREFIX[name]}params.npy"),
                   "diagnostics": np.load(folder / f"{PREFIX[name]}diagnostics.npy")} for name in SETS}
    return {"id": folder.name, "label": record["label"], "verts": verts, "faces": faces, "sets": sets,
            "arrays": shape_arrays(folder, config, preprocess)}


@torch.no_grad()
def predict(model, mesh: dict, name: str, amplitude: float, batch_size: int) -> dict:
    """Errors (as ``packet_errors``), norm ratios and predicted states of the packets of one set of a mesh.

    ``per_time`` and ``ratios`` are (n, T - 1), ``total`` (n,) and ``states`` complex (n, N, T) with ψ0 first,
    as the dataset stores them.
    """
    device = next(model.parameters()).device
    outputs = mesh["sets"][name]["outputs"]
    n_packets, n, n_times, _ = outputs.shape
    shared = {key: torch.from_numpy(mesh["arrays"][key]).to(device) for key in ("evecs", "evals", "mass", "geo_feat")}
    result = {"per_time": [], "total": [], "ratios": [], "states": []}
    for start in range(0, n_packets, batch_size):
        chunk = torch.from_numpy(np.float32(amplitude) * outputs[start:start + batch_size]).to(device)
        size = len(chunk)
        batch = {key: value.unsqueeze(0).repeat(size, *[1] * value.dim()) for key, value in shared.items()}
        batch["input"], labels = chunk[:, :, 0], chunk[:, :, 1:].reshape(size, n, 2 * (n_times - 1))
        predicted = model(batch)
        each, whole = packet_errors(predicted, labels, batch["mass"])
        weights = batch["mass"].reshape(size, n, 1)

        def norms(states: torch.Tensor) -> torch.Tensor:
            return torch.sum(weights * states.reshape(size, n, n_times - 1, 2).square().sum(-1), dim=1).sqrt()

        states = torch.cat([batch["input"].unsqueeze(2), predicted.reshape(size, n, n_times - 1, 2)], dim=2).double()
        result["per_time"].append(each.double().cpu().numpy())
        result["total"].append(whole.double().cpu().numpy())
        result["ratios"].append((norms(predicted) / norms(labels)).double().cpu().numpy())
        result["states"].append(torch.complex(states[..., 0], states[..., 1]).cpu().numpy() / amplitude)
    return {key: np.concatenate(values) for key, values in result.items()}


def references(mesh: dict, name: str, config: ShapesConfig, preprocess) -> dict:
    """Per packet of a set: the error of predicting ψ0, the spectral oracle and the data's own CN estimate."""
    arrays, data = mesh["arrays"], mesh["sets"][name]
    states = data["outputs"][..., 0].astype(np.float64) + 1j * data["outputs"][..., 1]          # (n, N, T)
    mass = arrays["mass"].astype(np.float64)[:, None]
    mu, steps = arrays["evals"].astype(np.float64) * preprocess.eval_scale, config.output_indices()[1:]

    def error(predicted: np.ndarray, target: np.ndarray) -> float:               # over every saved time, as the loss
        return float(np.sqrt(np.sum(mass * np.abs(predicted - target) ** 2) / np.sum(mass * np.abs(target) ** 2)))

    psi0 = [error(np.repeat(state[:, :1], state.shape[1] - 1, axis=1), state[:, 1:]) for state in states]
    oracle = [error(propagate(arrays["evecs"], mu, mass[:, 0], state[:, 0], steps, config.dt, "cn"), state[:, 1:])
              for state in states]
    return {"psi0": np.array(psi0), "oracle": np.array(oracle),
            "cn_estimate": data["diagnostics"][:, DIAGNOSTIC_COLUMNS.index("cn_error_estimate")]}


def summarize(errors: dict) -> dict:
    total, ratios = errors["total"], errors["ratios"]
    return {"packets": len(total), "mean": float(total.mean()), "median": float(np.median(total)),
            "min": float(total.min()), "max": float(total.max()), "per_time": errors["per_time"].mean(axis=0).tolist(),
            "norm_ratio": {"per_time": ratios.mean(axis=0).tolist(), "min": float(ratios.min()),
                           "max": float(ratios.max())}}


def errors_figure(meshes: list[dict], result: dict, run: str, times: np.ndarray, synthetic: dict | None,
                  out: Path) -> Path:
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(19.5, 4.8), layout="constrained")
    for mesh, colour in zip(meshes, COLOURS):
        for name, style, marker in (("test", "-", "o"), ("ood", "--", "^")):
            errors = result[mesh["id"]][name]
            label = f"{mesh['label']}, {name}"
            left.plot(times, errors["per_time"].mean(axis=0) * 100, style, color=colour, marker="o", markersize=4,
                      label=label)
            ratios = (errors["ratios"] - 1) * 100
            middle.plot(times, ratios.mean(axis=0), style, color=colour, marker="o", markersize=4, label=label)
            middle.fill_between(times, ratios.min(axis=0), ratios.max(axis=0), color=colour, alpha=0.1, lw=0)
            right.scatter(mesh["sets"][name]["params"][:, 2], errors["total"] * 100, s=18, marker=marker, color=colour,
                          alpha=0.7, label=label)
    if synthetic is not None:
        for name, style in (("test", "-"), ("ood", "--")):
            left.plot(times, np.array(synthetic[name]["select"]["per_time"]) * 100, style, color="#888888",
                      label=f"esferas y toros de test, {name}")
    left.set(yscale="log", xlabel="t", ylabel="error relativo L² (%)",
             title=f"Error contra CN por tiempo ({LABELS.get(run, run)}, sin reentrenar)")
    left.grid(alpha=0.3, which="both")
    left.legend(fontsize=8, frameon=False)
    middle.axhline(0, color="#444444", lw=1)
    middle.set(xlabel="t", ylabel="‖ψ_GLNO‖ / ‖ψ_CN‖ − 1 (%)",
               title="Norma de la predicción contra la de CN (media; banda: todos los paquetes)")
    middle.grid(alpha=0.3)
    middle.legend(fontsize=8, frameon=False)
    right.axvline(6.0, color="#444444", lw=1, ls=":")
    right.set(yscale="log", xlabel="k0 del paquete", ylabel="error relativo L² (%)",
              title="Error de cada paquete contra su k0 (entrenamiento: k0 en [1, 6])")
    right.grid(alpha=0.3, which="both")
    right.legend(fontsize=8, frameon=False)
    path = out / "errors.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "meshes")
    parser.add_argument("--config", type=Path, default=ROOT / "data" / "shapes" / CONFIG_FILE,
                        help="config.json of the dataset the meshes were generated with")
    parser.add_argument("--runs", type=Path, nargs="+",
                        default=[ROOT / "results" / "glno_schrodinger" / "runs" / "shapes_glno_periodic_v1", ROOT / "results" / "glno_schrodinger" / "runs" / "shapes_glno_full_v1"])
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts" / "glno_shapes_eval" / "summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "glno_meshes_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    torch.exp(torch.zeros(8))                   # the first vector-math call of this torch build can be off on CPU
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    config = ShapesConfig(**json.loads(args.config.read_text())["config"])
    models = {run_dir.name: load_checkpoint(run_dir / "best.pt", args.device) for run_dir in args.runs}
    preprocess = checkpoint_preprocess(next(iter(models.values()))[1])
    for name, (_, state) in models.items():
        if checkpoint_preprocess(state) != preprocess:
            raise SystemExit(f"{name} was trained with another preprocessing than {args.runs[0].name}")
    if not preprocess.drop_initial:
        raise SystemExit("the labels of these checkpoints include t = 0; this script assumes they do not")
    folders = sorted(folder for folder in args.data.iterdir() if (folder / SHAPE_FILE).is_file())
    if not folders:
        raise SystemExit(f"no mesh folders in {args.data}; run generate_meshes_dataset.py first")
    meshes = []
    for folder in folders:
        meshes.append(load_mesh(folder, config, preprocess))
        print(f"{folder.name}: {len(meshes[-1]['verts'])} vertices, largest geo_feat "
              f"{meshes[-1]['arrays']['geo_feat'].max():.3f} (training shapes: at most 1), "
              f"{time.perf_counter() - start:.0f} s", flush=True)
    times = config.output_times()

    results = {run: {mesh["id"]: {name: predict(model, mesh, name, preprocess.amplitude, args.batch_size)
                                  for name in SETS} for mesh in meshes} for run, (model, _) in models.items()}
    baselines = {mesh["id"]: {name: references(mesh, name, config, preprocess) for name in SETS} for mesh in meshes}
    print(f"evaluated ({time.perf_counter() - start:.0f} s)", flush=True)
    synthetic = json.loads(args.reference.read_text()) if args.reference.is_file() else None

    def span(low: float, high: float, digits: int = 2) -> str:
        return f"{low:.{digits}f}–{high:.{digits}f}"

    def row(run: str, label: str, name: str, s: dict, spread: str) -> str:
        ratio = s["norm_ratio"]
        return (f"{LABELS.get(run, run):<18}{label:<24}{name:<6}{s['packets']:>8}{s['mean'] * 100:>8.2f}"
                f"{s['median'] * 100:>10.2f}{spread:>14}{s['per_time'][0] * 100:>10.2f}{s['per_time'][-1] * 100:>9.2f}"
                f"{span(ratio['min'], ratio['max'], 3):>16}")

    print(f"\n{'run':<18}{'mesh':<24}{'set':<6}{'packets':>8}{'mean %':>8}{'median %':>10}{'min–max %':>14}"
          f"{'t=0.05 %':>10}{'t=0.5 %':>9}{'norm min–max':>16}")
    for run, result in results.items():
        for mesh in meshes:
            for name in SETS:
                s = summarize(result[mesh["id"]][name])
                print(row(run, mesh["label"], name, s, span(s["min"] * 100, s["max"] * 100)))
        if synthetic is not None and run in synthetic["runs"]:
            for name in SETS:
                print(row(run, "esferas y toros (test)", name, synthetic["runs"][run][name]["select"], ""))
    print(f"\nreferences, mean % (max %)\n{'mesh':<24}{'set':<6}{'oracle':>16}{'CN dt vs dt/2':>16}{'predict ψ0':>12}"
          f"{'k0':>12}")
    for mesh in meshes:
        for name in SETS:
            b, k0 = baselines[mesh["id"]][name], mesh["sets"][name]["params"][:, 2]
            oracle = f"{b['oracle'].mean() * 100:.2f} ({b['oracle'].max() * 100:.2f})"
            estimate = f"{b['cn_estimate'].mean() * 100:.2f} ({b['cn_estimate'].max() * 100:.2f})"
            print(f"{mesh['label']:<24}{name:<6}{oracle:>16}{estimate:>16}{b['psi0'].mean() * 100:>12.1f}"
                  f"{span(k0.min(), k0.max()):>12}")

    main_run = args.runs[0].name
    paths = [errors_figure(meshes, results[main_run], main_run, times[1:],
                           synthetic["runs"].get(main_run) if synthetic is not None else None, args.out)]
    for mesh in meshes:
        errors = results[main_run][mesh["id"]]["test"]
        order = np.argsort(errors["total"])
        for packet in dict.fromkeys((int(order[len(order) // 2]), int(order[-1]))):        # the median and the worst
            outputs = mesh["sets"]["test"]["outputs"][packet].astype(np.float64)
            _, _, k0, sigma = mesh["sets"]["test"]["params"][packet]
            title = (f"{mesh['label']}, test {packet}: k0 = {k0:.2f}, σ = {sigma:.3f}; error relativo L² sobre t > 0: "
                     f"{errors['total'][packet] * 100:.2f} %")
            paths.append(surface_comparison(mesh["verts"] @ UPRIGHT, mesh["faces"], times,
                                            outputs[..., 0] + 1j * outputs[..., 1],
                                            errors["states"][packet], torus=False, run=main_run, title=title,
                                            path=args.out / f"compare_{mesh['id']}_test{packet}.png"))
    summary = {"data": str(args.data), "times": times[1:].tolist(),
               "meshes": {mesh["id"]: {"label": mesh["label"], "n_verts": len(mesh["verts"]),
                                       "max_geo_feat": float(mesh["arrays"]["geo_feat"].max()),
                                       "references": {name: {key: {"mean": float(values.mean()), "max": float(values.max())}
                                                             for key, values in baselines[mesh["id"]][name].items()}
                                                      for name in SETS}} for mesh in meshes},
               "runs": {run: {mesh["id"]: {name: summarize(result[mesh["id"]][name]) for name in SETS} for mesh in meshes}
                        for run, result in results.items()}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\nwrote", *(str(path) for path in [*paths, args.out / "summary.json"]), sep="\n  ")
    print(f"{time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
