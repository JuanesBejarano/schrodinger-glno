"""Step 5 of Plan_GLNO_etapa3.md: walk a whole stage-3 dataset through ``ShapeStore`` and compare it with the files.

For every packet of every shape, from the batches the network would get:

- ``input`` and ``labels`` divided by the amplitude are the stored ψ0 and the stored states at every
  saved time after t = 0;
- the padded rows are exactly zero;
- the basis is the Ritz basis recomputed from the files (signs included), orthonormal in the
  batch's mass; the eigenvalues ascend and stay near [0, 1];
- ``geo_feat`` is ``‖x - c‖ / geo_scale`` and ``params`` and ``diagnostics`` are the files' rows;
- the spectral oracle computed from the batch (``ShapeStore.oracle_error``) is the one computed
  from the files with that Ritz basis (``spectral.spectral_oracle_error``, step 1).

Those batches hold one set of one shape. The batches of training mix the shapes of a family in
any order, so some shuffled batches of ``family_batches`` are compared, item by item, with each
packet alone.

A mistake in a shape, a scale, the padding or the order of the label channels shows here, on the
CPU, instead of in a paid run. The shapes go through in chunks so the memory stays bounded; a
chunk pads as the whole dataset does, so the bytes add up to those of one store of everything.
With no problems, the fingerprint of the basis is written next to the report, to compare with
the one another machine computes (``shapes_data.fingerprint_mismatch``).

    python scripts/check_glno_shapes_data.py --data data/shapes
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from schrodinger_glno.glno import shapes_data
from schrodinger_glno.glno.shapes_data import (SETS, ShapesPreprocess, ShapeStore, family_batches,
                                               training_geo_scale)
from schrodinger_glno.glno.spectral import eigen_clusters, ritz_polar_basis, spectral_oracle_error
from schrodinger_glno.shapes_dataset import MANIFEST_FILE, ShapesConfig

ROOT = Path(__file__).resolve().parents[1]
# Tolerances, with the largest value measured on data/shapes (200 shapes, 1100 packets) on 2026-11-12:
LABEL_TOLERANCE = 5e-7                # 5.9e-8: 1.5 ulp of float32 for x·A/A, plus the rounding of A to float32
ORTHONORMALITY_TOLERANCE = 2e-6       # 1.05e-7: Eᵀ M E - I of the float32 basis
ORACLE_TOLERANCE = 2e-6               # 6.8e-8: the batch's float32 basis and eigenvalues against the float64 ones
ORACLE_LIMIT = 2.5e-3                 # step 1: the oracle stays below this on the dataset
GEO_TOLERANCE = 1e-6                  # 3.0e-8 on 15 shapes: float32 rounding of a number <= 1
EVAL_RANGE = (0.0, 1.05)
MIXED_BATCHES = 2                     # shuffled batches per family and chunk, of MIXED_BATCH_SIZE packets
MIXED_BATCH_SIZE = 4


def check_shape(store: ShapeStore, directory: Path, shape: dict) -> dict:
    """Every comparison for one shape; the batches hold all the packets of one set of that shape."""
    config, preprocess = store.config, store.preprocess
    folder = directory / shape["id"]
    with np.load(folder / "mesh.npz") as stored:
        verts, faces, mass, evecs = (stored[key] for key in ("verts", "faces", "mass", "evecs"))
    ritz = ritz_polar_basis(verts, faces, evecs)
    n = shape["n_verts"]
    geo = np.linalg.norm(verts - (mass[:, None] * verts).sum(axis=0) / mass.sum(), axis=1) / preprocess.geo_scale
    result = {"id": shape["id"], "family": shape["family"], "split": shape["split"], "round": shape["round"],
              "n_verts": n, "sets": {}}
    for name, prefix, count, _ in config.packet_sets(shape["index"]):
        rows = store.rows(sets=name, shapes=shape["index"])
        if len(rows) != count:
            raise ValueError(f"{shape['id']}: {len(rows)} packets of set {name} in the store, {count} on disk")
        batch = store.batch(rows)
        inputs, outputs = np.load(folder / f"{prefix}inputs.npy"), np.load(folder / f"{prefix}outputs.npy")
        states = outputs[:, :, -len(store.steps):]
        labels = batch["labels"].double().numpy().reshape(count, -1, len(store.steps), 2) / preprocess.amplitude
        start = batch["input"].double().numpy() / preprocess.amplitude
        # relative to the largest |ψ| of each state: single entries near zero carry no information
        scale = np.abs(states).max(axis=(1, 3), keepdims=True)
        label_error = float((np.abs(labels[:, :n] - states) / scale).max())
        input_error = float((np.abs(start[:, :n] - inputs) / np.abs(inputs).max(axis=(1, 2), keepdims=True)).max())
        padding = max(float(batch[key][:, n:].abs().max()) if batch[key].shape[1] > n else 0.0
                      for key in ("input", "labels", "mass", "geo_feat", "evecs"))
        basis, weights = batch["evecs"][0, :n].double().numpy(), batch["mass"][0, :n].double().numpy()
        gram = float(np.abs(basis.T @ (weights[:, None] * basis) - np.eye(store.k_eig)).max())
        evals = batch["evals"][0].double().numpy()
        from_batch = store.oracle_error(batch)
        from_files = np.stack([spectral_oracle_error(
            ritz.E, ritz.mu, mass, inputs[p, :, 0] + 1j * inputs[p, :, 1],
            states[p, :, :, 0] + 1j * states[p, :, :, 1], store.steps, store.dt) for p in range(count)])
        result["sets"][name] = {
            "packets": count, "label_error": label_error, "input_error": input_error, "padding": padding,
            "orthonormality": gram, "evals_min": float(evals.min()), "evals_max": float(evals.max()),
            "evals_ascending": bool(np.all(np.diff(evals) >= 0)),
            "same_basis_in_batch": bool((batch["evecs"] == batch["evecs"][:1]).all()),
            "basis_exact": bool(np.array_equal(batch["evecs"][0, :n].numpy(), ritz.E)),
            "geo_error": float(np.abs(batch["geo_feat"][:, :n].double().numpy() - geo).max()),
            "tables_match": bool(np.array_equal(store.params[rows], np.load(folder / f"{prefix}params.npy"))
                                 and np.array_equal(store.diagnostics[rows],
                                                    np.load(folder / f"{prefix}diagnostics.npy"))),
            "clusters_match": bool(np.array_equal(batch["clusters"][0].numpy(),
                                                  eigen_clusters(ritz.mu, preprocess.cluster_gap))),
            "geo_feat_max": float(batch["geo_feat"].max()),
            "input_rms": float(np.sqrt(np.mean(batch["input"][:, :n].double().numpy() ** 2))),
            "oracle_max": float(from_batch.max()), "oracle_last_time_max": float(from_batch[:, -1].max()),
            "oracle_difference": float(np.abs(from_batch - from_files).max()),
            "cn_error_estimate_max": float(store.diagnostics[rows, 1].max())}
    return result


def failures(shape: dict) -> list[str]:
    found = []
    for name, r in shape["sets"].items():
        checks = {"labels differ from the stored states": r["label_error"] <= LABEL_TOLERANCE,
                  "input differs from the stored ψ0": r["input_error"] <= LABEL_TOLERANCE,
                  "padded rows are not zero": r["padding"] == 0.0,
                  "basis is not orthonormal in the mass": r["orthonormality"] <= ORTHONORMALITY_TOLERANCE,
                  "eigenvalues leave the expected range": EVAL_RANGE[0] <= r["evals_min"] and r["evals_max"] <= EVAL_RANGE[1],
                  "eigenvalues do not ascend": r["evals_ascending"],
                  "packets of one shape got different bases": r["same_basis_in_batch"],
                  "basis is not the Ritz basis of the files (signs included)": r["basis_exact"],
                  "geo_feat is not ‖x - c‖ / geo_scale": r["geo_error"] <= GEO_TOLERANCE,
                  "params or diagnostics are not the files' rows": r["tables_match"],
                  "eigenvalue groups differ": r["clusters_match"],
                  "oracle from the batch differs from the files": r["oracle_difference"] <= ORACLE_TOLERANCE,
                  "oracle above the step-1 limit": r["oracle_max"] <= ORACLE_LIMIT}
        found += [f"{shape['id']} [{name}]: {text}" for text, passed in checks.items() if not passed]
    return found


def check_mixed_batches(store: ShapeStore) -> tuple[int, list[str]]:
    """Shuffled batches of several shapes, each item against its packet alone (which ``check_shape`` compares
    with the files); how many packets were compared and what differed."""
    compared, found = 0, []
    for family in store.n_max:
        rows = store.rows(family=family)
        for members in family_batches(store, rows, MIXED_BATCH_SIZE, seed=0)[:MIXED_BATCHES]:
            batch = store.batch(members)
            oracle = store.oracle_error(batch)
            for b, row in enumerate(members):
                alone = store.batch([row])
                differ = [key for key in ("input", "labels", "mass", "evals", "evecs", "geo_feat", "clusters", "n_verts")
                          if not torch.equal(batch[key][b], alone[key][0])]
                if int(batch["rows"][b]) != row:
                    differ.append("rows")
                if not np.allclose(oracle[b], store.oracle_error(alone)[0], rtol=1e-9, atol=0):
                    differ.append("oracle")
                if differ:
                    found.append(f"mixed {family} batch {members.tolist()}, packet {row}: {', '.join(differ)} "
                                 "differ from the packet alone")
            compared += len(members)
    return compared, found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "shapes")
    parser.add_argument("--chunk", type=int, default=20, help="shapes per store (20 shapes of data/shapes: ~0.8 GiB)")
    parser.add_argument("--out", type=Path, help="JSON report (default artifacts/glno_shapes_data_check_<data>.json)")
    parser.add_argument("--verify", action="store_true", help="check the SHA-256 of every file first")
    args = parser.parse_args()
    out = args.out or ROOT / "artifacts" / f"glno_shapes_data_check_{args.data.name}.json"
    manifest = json.loads((args.data / MANIFEST_FILE).read_text())
    config, preprocess = ShapesConfig(**manifest["config"]), ShapesPreprocess()
    code = {Path(source).name: hashlib.sha256(Path(source).read_bytes()).hexdigest()[:16]
            for source in (shapes_data.__file__, __file__)}
    largest = {family: max(entry["n_verts"] for entry in manifest["shapes"] if entry["family"] == family)
               for family in sorted({entry["family"] for entry in manifest["shapes"]})}
    start = time.perf_counter()
    shapes, problems, nbytes, packets, mixed, fingerprint, n_max, identity = [], [], 0, 0, 0, None, None, None
    for first in range(0, config.n_shapes, args.chunk):
        indices = range(first, min(first + args.chunk, config.n_shapes))
        store = ShapeStore(args.data, preprocess, shapes=indices, verify=args.verify and first == 0)
        nbytes, packets, n_max, identity = nbytes + store.nbytes, packets + len(store), store.n_max, store.identity
        if dict(sorted(n_max.items())) != largest:
            problems.append(f"shapes {indices[0]}-{indices[-1]}: padded sizes {n_max}, the dataset's largest meshes "
                            f"are {largest}")
        part = store.fingerprint()
        fingerprint = part | {"shapes": (fingerprint["shapes"] if fingerprint else {}) | part["shapes"]}
        for shape in store.shapes:
            shapes.append(check_shape(store, args.data, shape))
            problems += failures(shapes[-1])
        compared, found = check_mixed_batches(store)
        mixed, problems = mixed + compared, problems + found
        del store
        print(f"shapes {indices[0]}-{indices[-1]}: {len(problems)} problems so far, "
              f"{time.perf_counter() - start:.0f} s", flush=True)

    counts = {name: sum(s["sets"].get(name, {}).get("packets", 0) for s in shapes) for name in SETS}
    if counts != manifest["counts"]["packets"]:
        problems.append(f"packet counts {counts} differ from the manifest's {manifest['counts']['packets']}")
    if any("ood" in s["sets"] and s["split"] != "test" for s in shapes):
        problems.append("out-of-distribution packets on a shape that is not a test shape")
    if len(fingerprint["shapes"]) != config.n_shapes:
        problems.append("a shape has no fingerprint")

    def worst(key: str, sets=SETS) -> float:
        return max(s["sets"][name][key] for s in shapes for name in s["sets"] if name in sets)

    def least(key: str, sets=SETS) -> float:
        return min(s["sets"][name][key] for s in shapes for name in s["sets"] if name in sets)

    in_distribution = ("train", "val", "test")
    summary = {
        "data": str(args.data), "identity": identity, "shapes": len(shapes), "packets": counts,
        "families": {f: sum(s["family"] == f for s in shapes) for f in n_max}, "n_max": n_max,
        "store_gib": nbytes / 2**30, "total_packets": packets, "packets_in_mixed_batches": mixed,
        "training_geo_scale": training_geo_scale(args.data), "geo_scale_used": preprocess.geo_scale,
        "geo_feat_max": worst("geo_feat_max"), "geo_error_max": worst("geo_error"),
        "label_error_max": worst("label_error"), "input_error_max": worst("input_error"),
        "padding_max": worst("padding"), "orthonormality_max": worst("orthonormality"),
        "evals_range": [least("evals_min"), worst("evals_max")],
        "input_rms_range": [least("input_rms"), worst("input_rms")],
        "oracle_difference_max": worst("oracle_difference"),
        "oracle_max": {"in_distribution": worst("oracle_max", in_distribution),
                       "ood": worst("oracle_max", ("ood",)) if counts["ood"] else None},
        "oracle_last_time_range": {
            "in_distribution": [least("oracle_last_time_max", in_distribution), worst("oracle_last_time_max", in_distribution)],
            "ood": [least("oracle_last_time_max", ("ood",)), worst("oracle_last_time_max", ("ood",))]
            if counts["ood"] else None},
        "tolerances": {"label": LABEL_TOLERANCE, "orthonormality": ORTHONORMALITY_TOLERANCE,
                       "oracle_difference": ORACLE_TOLERANCE, "oracle_limit": ORACLE_LIMIT, "geo": GEO_TOLERANCE,
                       "evals": EVAL_RANGE},
        "torch": torch.__version__, "numpy": np.__version__, "code_sha256": code,
        "seconds": time.perf_counter() - start,
        "problems": problems}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "shapes": shapes}, indent=1))
    for key, value in summary.items():
        if key != "problems":
            print(f"{key}: {value}")
    print(f"{len(problems)} problems" + "".join(f"\n  {p}" for p in problems))
    print(f"wrote {out}")
    if not problems:        # the basis that passed, to compare with the one another machine computes
        reference = out.with_name(f"glno_shapes_fingerprint_{args.data.name}.json")
        reference.write_text(json.dumps(fingerprint))
        print(f"wrote {reference}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
