"""Datasets of Gaussian packets on a fixed surface (V = 0), solved with Crank–Nicolson.

The surface is an icosphere (stage 1) or a torus of revolution (stage 2), chosen by
``SurfaceConfig.surface``; everything else is shared.

Directory layout, following ``schrodinger_fno.dataset``::

    mesh.npz                 verts, faces, mass (lumped), evals, evecs, geo_feat, t_out
    <split>/inputs.npy       (n, V, 2)                  float32  Re, Im of ψ0
    <split>/outputs.npy      (n, V, output_steps, 2)    float32  saved states, t = 0 included
    <split>/norms.npy        (n, output_steps)          float64  ψᴴ M ψ of each saved state
                                                                 (M = the solver's mass)
    <split>/params.npy       (n, 4)                     float64  center, phi, k0, sigma
    <split>/diagnostics.npy  (n, 2)                     float64  spectral fraction in the
                                                                 k_eig GLNO modes, CN phase-error estimate
    manifest.json            settings, sampling report, SHA-256 of every file and an identity hash

``DIAGNOSTIC_COLUMNS``:

- ``spectral_fraction``: ``‖Φᵀ M_lumped ψ0‖² / ‖ψ0‖²`` with the k_eig lumped eigenvectors Φ
  saved in ``mesh.npz``, the part of ψ0 the GLNO basis can represent.
- ``cn_phase_error_estimate``: ``T dt² / 12 · (Σ |c_j|² E_j⁶)^½`` over the first k_eig
  eigenpairs of the solver's mass (``E_j = λ_j / 2``), the leading term of CN's phase
  error. It leaves out the mesh-scale content above those modes, whose CN phase is wrong
  for any practical dt, so it is a lower bound of the CN error. When k_eig covers the
  packet's spectrum it is close: 0.6–1.0 times the measured error on the production
  torus and sphere (the gap, a floor of ~2e-4, shows on slow packets). On coarse test
  meshes with k_eig = 60 it can be 10–100× too small.

The sampling report in the manifest gives, per split, the empirical ``acceptance``
of the ``(k0, sigma)`` draws and the exact ``resolved_fraction`` of the requested
rectangle (``sampling.resolved_fraction``); below 1, the split only covers the
packets with ``k0 + 3/(2σ) <= allowed_wavenumber``.

Bit-for-bit reproducibility: the identity covers ``mesh.npz``, and ARPACK returns a
rotated basis inside degenerate eigenspaces (the sphere's 2ℓ+1 per level) when the
number of BLAS threads changes. A regenerated dataset has the same identity only with
single-threaded BLAS (``OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1``, set
before NumPy loads; ``scripts/generate_surface_dataset.py`` does it) and the same NumPy,
SciPy and BLAS builds. Otherwise the eigenbasis spans the same spaces, the outputs agree
to ~1e-8 and the GLNO's spectral filter, a function of λ, is the same operator in any
basis of a degenerate eigenspace; only the hashes differ.

The time stepper uses the mass chosen in ``SurfaceConfig.mass`` (blended by
default). ``mesh.npz`` stores the lumped mass and its eigenpairs because those are
what the GLNO uses, as in the paper's code. The manifest uses the 1D schema, so
``schrodinger_fno.dataset.dataset_identity`` verifies it. ``geo_feat`` is the GLNO
repo's distance to the centroid; on a sphere it is constant.

The dataset is written to ``<directory>.partial`` and renamed at the end, so an
interrupted run never leaves a directory that looks complete.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Callable
import warnings
import numpy as np
from numpy.typing import NDArray
from schrodinger_glno.data_utils import DatasetConfig
from schrodinger_glno.data_utils import _digest, _manifest_identity, psi_to_channels
from .config import SurfaceConfig
from .initial_conditions import gaussian_packet
from .mesh import (blended_mass, cotan_laplacian, icosphere, laplacian_eigenpairs, mean_edge_length,
                   vertex_areas, vertex_normals)
from .sampling import PARAM_COLUMNS, allowed_wavenumber, direction_from_angle, resolved_fraction, sample_packets
from .solver import build_cn_matrices, solve_CN

DIAGNOSTIC_COLUMNS = ("spectral_fraction", "cn_phase_error_estimate")


def surface_vertices_faces(config: SurfaceConfig) -> tuple[NDArray, NDArray]:
    """The mesh of the configured surface."""
    if config.surface == "sphere":
        return icosphere(config.subdivisions, config.radius)
    from .mesh import torus                     # stage 2
    return torus(config.target_edge, config.major_radius, config.minor_radius)


def surface_mesh(config: SurfaceConfig) -> dict:
    """The surface mesh, its operators, the GLNO eigenpairs (lumped mass) and the solver's mass."""
    verts, faces = surface_vertices_faces(config)
    L, mass = cotan_laplacian(verts, faces), vertex_areas(verts, faces)
    evals, evecs = laplacian_eigenpairs(L, mass, config.k_eig)
    if config.mass == "blended":
        solver_mass = blended_mass(verts, faces)
        solver_evals, solver_evecs = laplacian_eigenpairs(L, solver_mass, config.k_eig)
    else:
        solver_mass, solver_evals, solver_evecs = mass, evals, evecs
    return {"verts": verts, "faces": faces, "L": L, "mass": mass,
            "normals": vertex_normals(verts, faces), "evals": evals, "evecs": evecs,
            "solver_mass": solver_mass, "solver_evals": solver_evals, "solver_evecs": solver_evecs,
            "mean_edge": mean_edge_length(verts, faces)}


def split_parameters(config: SurfaceConfig, counts: DatasetConfig, mass: NDArray,
                     k_allowed: float) -> tuple[dict[str, NDArray], dict[str, dict]]:
    """Draw every split from one seeded stream: disjoint and reproducible.

    Also returns, per split, the fraction of ``(k0, sigma)`` draws accepted, the exact
    fraction of the requested rectangle the mesh resolves and the ranges actually sampled.
    """
    rng = np.random.default_rng(counts.seed)
    total = counts.train + counts.val + counts.test
    stats = {}
    params = sample_packets(total, mass, rng, config.k0_range, config.sigma_range, k_allowed, stats)
    a, b = counts.train, counts.train + counts.val
    splits = {"train": params[:a], "val": params[a:b], "test": params[b:]}
    acceptance = {name: stats for name in splits}
    if counts.ood:
        ood_stats = {}
        splits["ood"] = sample_packets(counts.ood, mass, rng, config.ood_k0_range,
                                       config.sigma_range, k_allowed, ood_stats)
        acceptance["ood"] = ood_stats
    report = {}
    for name, rows in splits.items():
        tried = acceptance[name]["tried"]
        k0_range = config.ood_k0_range if name == "ood" else config.k0_range
        report[name] = {"acceptance": acceptance[name]["accepted"] / tried if tried else 1.0,
                        "resolved_fraction": resolved_fraction(k0_range, config.sigma_range, k_allowed),
                        "k0": [float(rows[:, 2].min()), float(rows[:, 2].max())] if len(rows) else None,
                        "sigma": [float(rows[:, 3].min()), float(rows[:, 3].max())] if len(rows) else None}
    return splits, report


def one_sample(row: NDArray, mesh: dict, config: SurfaceConfig, LU: object,
               U1: object) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Initial state, saved trajectory, conserved norms and diagnostics for one parameter row."""
    center, phi, k0, sigma = int(row[0]), *row[1:]
    verts, normals, M = mesh["verts"], mesh["normals"], mesh["solver_mass"]
    direction = direction_from_angle(normals[center], phi)
    psi0 = gaussian_packet(verts, M, normals, center, k0, sigma, direction)
    traj = solve_CN(psi0, LU, U1, config.steps, config.output_indices())
    if np.ndim(M) == 1:
        norms = np.sum(M[:, None] * np.abs(traj)**2, axis=0)
    else:
        norms = np.real(np.einsum("vt,vt->t", traj.conj(), M @ traj))
    lumped = mesh["mass"]
    glno_coeffs = mesh["evecs"].T @ (lumped * psi0)            # how much of ψ0 the GLNO basis captures
    fraction = np.sum(np.abs(glno_coeffs)**2) / np.sum(lumped * np.abs(psi0)**2)
    solver_coeffs = mesh["solver_evecs"].T @ (M @ psi0 if np.ndim(M) == 2 else M * psi0)
    energies = mesh["solver_evals"] / 2
    estimate = config.final_time * config.dt**2 / 12 * np.sqrt(np.sum(np.abs(solver_coeffs)**2 * energies**6))
    return psi_to_channels(psi0), psi_to_channels(traj), norms, np.array([fraction, estimate])


def generate_surface_dataset(config: SurfaceConfig, counts: DatasetConfig, directory: Path,
                             progress: Callable[[str], None] | None = None) -> dict:
    """Write the dataset through memmaps, one trajectory at a time; return the manifest.

    This function does not limit the BLAS threads: run it with single-threaded BLAS to
    reproduce an identity bit for bit (see the module docstring).
    """
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"Destination is not empty: {directory}")
    partial = directory.with_name(directory.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"{partial} is left from an interrupted run; delete it and try again")
    say = progress or (lambda message: None)
    mesh = surface_mesh(config)
    k_allowed = allowed_wavenumber(mesh["mean_edge"], config.min_points_per_wavelength)
    splits, sampling = split_parameters(config, counts, mesh["mass"], k_allowed)
    for name, info in sampling.items():
        if info["resolved_fraction"] < 1:
            message = (f"{name}: only {info['resolved_fraction']:.1%} of the requested (k0, sigma) rectangle is resolved "
                       f"by this mesh; sampled k0 in {info['k0']}, sigma in {info['sigma']}")
            warnings.warn(message)
            say("warning: " + message)
    partial.mkdir(parents=True)
    verts = mesh["verts"]
    np.savez(partial / "mesh.npz", verts=verts, faces=mesh["faces"], mass=mesh["mass"],
             evals=mesh["evals"], evecs=mesh["evecs"],
             geo_feat=np.linalg.norm(verts - verts.mean(axis=0), axis=1), t_out=config.output_times())
    LU, U1 = build_cn_matrices(mesh["L"], mesh["solver_mass"], np.zeros(len(verts)), config.dt)
    n_vertices, n_out = len(verts), config.output_steps
    for name, params in splits.items():
        split_dir = partial / name
        split_dir.mkdir()
        n = len(params)
        inputs = np.lib.format.open_memmap(split_dir / "inputs.npy", mode="w+", dtype=np.float32,
                                           shape=(n, n_vertices, 2))
        outputs = np.lib.format.open_memmap(split_dir / "outputs.npy", mode="w+", dtype=np.float32,
                                            shape=(n, n_vertices, n_out, 2))
        norms = np.zeros((n, n_out))
        diagnostics = np.zeros((n, len(DIAGNOSTIC_COLUMNS)))
        for i, row in enumerate(params):
            inputs[i], outputs[i], norms[i], diagnostics[i] = one_sample(row, mesh, config, LU, U1)
            say(f"{name}: {i + 1}/{n}")
        inputs.flush(); outputs.flush()
        del inputs, outputs
        np.save(split_dir / "norms.npy", norms)
        np.save(split_dir / "params.npy", params)
        np.save(split_dir / "diagnostics.npy", diagnostics)
    files = [partial / "mesh.npz", *(p for name in splits for p in sorted((partial / name).glob("*.npy")))]
    manifest = {"format_version": 2, "preset": config.surface, "physics": config.physics_dict(),
                "counts": asdict(counts), "time_convention": "states include t=0",
                "splits": list(splits),
                "sha256": {str(p.relative_to(partial)): _digest(p) for p in files},
                "param_columns": list(PARAM_COLUMNS), "diagnostic_columns": list(DIAGNOSTIC_COLUMNS),
                "sampling": {"allowed_wavenumber": k_allowed, **sampling},
                "mesh": {"vertices": n_vertices, "faces": len(mesh["faces"]),
                         "mean_edge_length": mesh["mean_edge"], "area": float(mesh["mass"].sum()),
                         "max_eigenvalue": float(mesh["evals"][-1])}}
    manifest["identity"] = _manifest_identity(manifest)
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if directory.exists():
        directory.rmdir()                     # empty, checked above
    os.replace(partial, directory)
    return manifest


# Stage-1 names, kept for existing code
sphere_mesh = surface_mesh
generate_sphere_dataset = generate_surface_dataset
