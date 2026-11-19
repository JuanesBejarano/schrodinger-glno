"""Real meshes for the trained GLNO: public triangle meshes (a cow, a bunny) brought to the
resolution of the stage-3 shapes, with packets and Crank–Nicolson references drawn by the stage-3 rules.

Each mesh of MESHES is read from --raw (downloaded once, see README.md) and prepared:

1. pymeshlab: duplicates and non-manifold parts removed, holes closed (the bunny's scan is open
   underneath) and isotropic explicit remeshing to ``--target-edge`` in the units of area 4π, without
   feature edges: the result is a uniform mesh of the smoothed surface, about 10 000 vertices.
2. Area 4π and ``shapes.delaunay_flip``, as ``shapes.build_shape``.
3. The few triangles that still miss the angle gates are relaxed: their vertices move towards the mean
   of their neighbours along the tangent plane (a fraction of an edge), and the mesh is flipped again.

The mesh must then pass ``shapes.check_gates`` with the dataset's ``min_wavenumber`` and genus 0, or
nothing is written. There is no curvature gate: these surfaces have no formula for it.

What is written in --out/<name> is a shape folder of ``shapes_dataset`` (mesh.npz, inputs.npy,
outputs.npy, norms.npy, params.npy, diagnostics.npy, ood_*.npy, shape.json), built by its code: the
lumped eigenpairs, packets by its rules b and c (rule a is for tori) in the ranges of --config, the
config.json of the dataset the network was trained on, and Crank–Nicolson at dt and dt/2. The set
``test`` has k0 and σ in the training ranges and ``ood`` the out-of-distribution ones. There is no
manifest: the folders are not a stage-3 dataset, and compare_glno_meshes.py reads them, not ``ShapeStore``.

Rule c is the one change: the norm a packet may leave outside the eigenvectors (1e-5 in the dataset,
6e-5 out of distribution) is multiplied by ``--spectral-slack``, 10 by default. On these surfaces a
packet leaves 3e-5 to 2e-4 of its norm outside the 700 modes, against 1e-8 on a sphere of the dataset:
with the dataset's limit 1 in 5 draws passes on the cow, and 1 in 100 on the bunny, all of
them with k0 < 2. With 10 the missing part is at most 1 % of the packet (the square root of 1e-4), and
compare_glno_meshes.py reports what it costs as the spectral oracle.

BLAS runs on one thread, set before NumPy loads, as in generate_shapes_dataset.py. A complete folder is
left as it is; delete it to build it again.

    python scripts/generate_meshes_dataset.py               # about 6 min per mesh; --only bunny for one
"""
import os

if __name__ == "__main__":          # must happen before NumPy loads; importing this module changes nothing
    for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_variable] = "1"

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from schrodinger_glno.geodesic import HeatGeodesics
from schrodinger_glno.mesh import face_areas, vertex_normals
from schrodinger_glno.shapes import _normalisation, check_gates, delaunay_flip, mesh_quality
from schrodinger_glno.shapes_dataset import (CONFIG_FILE, OOD_STREAM, PACKET_STREAM, SHAPE_FILE, ShapesConfig,
                                             _shape_mesh, _write_packets)
from schrodinger_glno.solver import build_cn_matrices

ROOT = Path(__file__).resolve().parents[1]
# ``index`` seeds the packets as a shape index does in the dataset, whose shapes are numbered from 0.
MESHES = {"spot": {"file": "spot/spot.obj", "label": "vaca Spot", "index": 1001},
          "bunny": {"file": "bunny/stanford-bunny.obj", "label": "conejo de Stanford", "index": 1002}}
SOURCE = "https://github.com/alecjacobson/common-3d-test-models"


def corner_angles(verts: NDArray, faces: NDArray) -> NDArray:
    """(F, 3) angles of every triangle, in degrees."""
    corners, angles = verts[faces], []
    for k in range(3):
        u, w = corners[:, (k + 1) % 3] - corners[:, k], corners[:, (k + 2) % 3] - corners[:, k]
        angles.append(np.arctan2(np.linalg.norm(np.cross(u, w), axis=1), np.einsum("fd,fd->f", u, w)))
    return np.degrees(np.stack(angles, axis=1))


def boundary_edges(faces: NDArray, n_verts: int) -> int:
    """Edges in one face only: 0 on a closed mesh."""
    start, end = faces.ravel().astype(np.int64), faces[:, [1, 2, 0]].ravel().astype(np.int64)
    _, count = np.unique(np.minimum(start, end) * n_verts + np.maximum(start, end), return_counts=True)
    return int((count == 1).sum())


def remesh(path: Path, target_edge: float, iterations: int) -> tuple[NDArray, NDArray, dict]:
    """A closed, uniform mesh of the surface in ``path``, with edges near ``target_edge`` at area 4π."""
    import pymeshlab                                    # only this script needs it
    meshes = pymeshlab.MeshSet()
    meshes.load_new_mesh(str(path))
    meshes.meshing_remove_duplicate_vertices()
    meshes.meshing_remove_duplicate_faces()
    meshes.meshing_remove_unreferenced_vertices()
    meshes.meshing_repair_non_manifold_edges()
    meshes.meshing_repair_non_manifold_vertices()
    raw = meshes.current_mesh()
    info = {"raw_verts": raw.vertex_number(), "raw_faces": raw.face_number(),
            "raw_boundary_edges": boundary_edges(raw.face_matrix(), raw.vertex_number())}
    meshes.meshing_close_holes(maxholesize=100000, selfintersection=False)
    closed = meshes.current_mesh()
    scale = np.sqrt(4 * np.pi / face_areas(closed.vertex_matrix(), closed.face_matrix()).sum())
    meshes.meshing_isotropic_explicit_remeshing(iterations=iterations, targetlen=pymeshlab.PureValue(target_edge / scale),
                                                featuredeg=180, checksurfdist=False)
    result = meshes.current_mesh()
    return result.vertex_matrix().astype(np.float64), result.face_matrix().astype(np.int64), info


def relax(verts: NDArray, faces: NDArray, min_angle: float, max_angle: float, margin: float = 1.0,
          rounds: int = 100) -> tuple[NDArray, NDArray, int, float]:
    """Move the vertices of the triangles within ``margin`` of the angle gates, then flip, until none is left.

    Each round moves them halfway to the mean of their neighbours, along the tangent plane. Returns the
    mesh, the rounds it took and the largest move of a vertex in a round.
    """
    verts, largest = verts.copy(), 0.0
    for done in range(rounds):
        angles = corner_angles(verts, faces)
        bad = np.flatnonzero((angles.min(axis=1) < min_angle + margin) | (angles.max(axis=1) > max_angle - margin))
        if not len(bad):
            return verts, faces, done, largest
        move = np.unique(faces[bad])
        start, end = faces.ravel(), faces[:, [1, 2, 0]].ravel()
        neighbours = sp.csr_matrix((np.ones(len(start)), (start, end)), shape=(len(verts), len(verts)))
        mean = (neighbours @ verts) / np.asarray(neighbours.sum(axis=1))
        normals = vertex_normals(verts, faces)[move]
        shift = 0.5 * (mean[move] - verts[move])
        shift -= np.einsum("vd,vd->v", shift, normals)[:, None] * normals
        verts[move] += shift
        largest = max(largest, float(np.linalg.norm(shift, axis=1).max()))
        faces, _ = delaunay_flip(verts, faces)
    return verts, faces, rounds, largest


def prepare(path: Path, config: ShapesConfig, target_edge: float, iterations: int) -> tuple[NDArray, NDArray, dict]:
    """The mesh of ``path`` at area 4π with its ``mesh_quality``, or SystemExit when it misses the gates."""
    verts, faces, info = remesh(path, target_edge, iterations)
    scale, offset = _normalisation(verts, faces)
    verts = scale * (verts - offset)
    faces, flips = delaunay_flip(verts, faces)
    verts, faces, rounds, largest = relax(verts, faces, 30.0, 120.0)          # the angle gates of check_gates
    scale, offset = _normalisation(verts, faces)
    verts = scale * (verts - offset)
    quality = mesh_quality(verts, faces) | info | {"target_edge": target_edge, "remesh_iterations": iterations,
                                                    "flips": flips, "relax_rounds": rounds, "relax_largest_move": largest}
    passed, reasons = check_gates(quality, 0, min_wavenumber=config.min_wavenumber)
    if not passed:
        raise SystemExit(f"{path}: the prepared mesh misses the gates: {reasons}")
    return verts, faces, quality


def write(config: ShapesConfig, out: Path, name: str, verts: NDArray, faces: NDArray, quality: dict,
          counts: tuple[int, int]) -> dict:
    """Eigenpairs, packets and Crank–Nicolson states of one mesh in ``out/name``, as ``shapes_dataset.write_shape``."""
    entry, target, folder = MESHES[name], out / name, out / f"{name}.partial"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    start = time.perf_counter()
    mesh = _shape_mesh(config, SimpleNamespace(verts=verts, faces=faces, quality=quality))
    np.savez(folder / "mesh.npz", verts=verts, faces=faces, mass=mesh["mass"], evals=mesh["evals"],
             evecs=mesh["evecs"].astype(np.float32), geo_feat=np.linalg.norm(verts - verts.mean(axis=0), axis=1),
             t_out=config.output_times())
    eigen_seconds = time.perf_counter() - start
    free = np.zeros(len(verts))                                    # V = 0
    solvers = (build_cn_matrices(mesh["L"], mesh["blended"], free, config.dt),
               build_cn_matrices(mesh["L"], mesh["blended"], free, config.dt / 2))
    mesh["geodesics"], mesh["sections"] = HeatGeodesics(verts, faces), None
    packets = {set_name: _write_packets(config, mesh, solvers, folder, entry["index"], set_name, prefix, n, stream)
               for set_name, prefix, n, stream in (("test", "", counts[0], PACKET_STREAM),
                                                   ("ood", "ood_", counts[1], OOD_STREAM))}
    record = {"id": name, "label": entry["label"], "source": f"{SOURCE} ({Path(entry['file']).name})",
              "family": "mesh", "genus": 0, "seed": [config.seed, entry["index"]], "quality": quality,
              "max_spectral_deficit": {"test": 1 - config.min_spectral_fraction,
                                       "ood": 1 - config.min_ood_spectral_fraction},
              "n_verts": len(verts), "k_allowed": mesh["k_allowed"], "max_eigenvalue": float(mesh["evals"][-1]),
              "packets": packets,
              "timings": {"eigenpairs": round(eigen_seconds, 1), "total": round(time.perf_counter() - start, 1)}}
    (folder / SHAPE_FILE).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    os.replace(folder, target)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "meshes_raw")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "meshes")
    parser.add_argument("--config", type=Path, default=ROOT / "data" / "shapes" / CONFIG_FILE)
    parser.add_argument("--only", nargs="+", choices=sorted(MESHES), help="Meshes to build (default: all)")
    parser.add_argument("--packets", type=int, default=20, help="Packets in the training ranges, per mesh")
    parser.add_argument("--ood-packets", type=int, default=10, help="Out-of-distribution packets, per mesh")
    parser.add_argument("--spectral-slack", type=float, default=10.0,
                        help="Factor on the norm rule c lets a packet leave outside the eigenvectors (1: the dataset's)")
    parser.add_argument("--target-edge", type=float, default=0.036, help="Edge asked of the remesher, at area 4π")
    parser.add_argument("--remesh-iterations", type=int, default=20)
    args = parser.parse_args(argv)
    config = ShapesConfig(**json.loads(args.config.read_text())["config"])
    config = replace(config, min_spectral_fraction=1 - args.spectral_slack * (1 - config.min_spectral_fraction),
                     min_ood_spectral_fraction=1 - args.spectral_slack * (1 - config.min_ood_spectral_fraction))
    for name in args.only or MESHES:
        if (args.out / name).is_dir():
            print(f"{name}: already in {args.out / name}", flush=True)
            continue
        verts, faces, quality = prepare(args.raw / MESHES[name]["file"], config, args.target_edge,
                                        args.remesh_iterations)
        print(f"{name}: {quality['raw_verts']} -> {quality['n_verts']} vertices, angles {quality['min_angle']:.1f}°-"
              f"{quality['max_angle']:.1f}°, mean edge {quality['mean_edge']:.4f}, k_allowed "
              f"{quality['k_allowed']:.2f}, {quality['raw_boundary_edges']} boundary edges closed, "
              f"{quality['relax_rounds']} relax rounds", flush=True)
        record = write(config, args.out, name, verts, faces, quality, (args.packets, args.ood_packets))
        for set_name, stats in record["packets"].items():
            print(f"{name} {set_name}: {stats['packets']} packets of {stats['draws']} draws (rejected "
                  f"{stats['rejected']}), CN estimate up to {stats['max_cn_error_estimate'] * 100:.2f} %", flush=True)
        print(f"{name}: {record['timings']['total']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
