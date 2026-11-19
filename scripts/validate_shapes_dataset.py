"""Check a stage-3 shapes dataset (``schrodinger_glno.shapes_dataset``) shape by shape; estimate its data error.

1. **Files:** the manifest identity, config.json and the SHA-256 of every data file (only those of
   the --shapes, when given); shape.json, which is not hashed (it holds the timings), against its
   manifest entry (``shapes_dataset._shape_entry``). The other checks read the manifest's copy of
   the record (deformation, quality, ...), which the identity covers.
2. **Shapes and meshes:** the entry against the config (id, family, split, round torus, seed, vertex
   count, k_allowed); the arrays of mesh.npz (sizes, ``t_out`` the config's saved times,
   ``geo_feat``); ``shapes.mesh_quality`` recomputed from the stored verts and faces and compared
   with the record, the shape's own numbers (smallest radius of curvature of a sphere, tube numbers
   of a torus) recomputed from its deformation, the gates (``shapes.check_gates``) on both, the
   stored vertices against the surface of the deformation (--refine puts new vertices on it) and the
   lumped mass. A round torus: ``exact_torus`` recomputed from its deformation and quality, its
   centre within 1e-4 of the origin and the vertices on the torus of radii s·R0, s·r0 (§8.9).
3. **Packets:** ψᴴMψ (blended M) recomputed from the float32 outputs against norms.npy, 1 at t = 0
   and its drift; the state at t = 0 is the input; the parameters lie in their set's ranges. The
   first --evolve packets of each set are evolved again by Crank–Nicolson from their stored input and
   compared with the stored outputs, which ψᴴMψ cannot do: a phase error or a misplaced saved time
   keeps it (2-3 s per packet at production size).
4. **ψ0 and spectral fraction:** the first --spectral packets of each set are rebuilt from their
   parameters and compared with the input, and their spectral fraction in the stored float32
   eigenvectors is compared with the stored one (float64 eigenvectors). Also the M-norms and the
   Rayleigh quotients of the float32 eigenvectors against 1 and the stored eigenvalues.
5. **--exact:** every packet of the round tori against ``reference.torus_exact_solution`` with the
   radii s·R0, s·r0 and the centre of ``exact_torus`` (§8.9). The reference starts from the chordal
   packet and the data from the packet cut at 7σ-9σ. The continuous evolution is unitary, so at every
   time the error against the exact solution of the cut packet differs from the printed one by at
   most ``‖ψ_cut - ψ_chordal‖ ≤ √r + r``, r the removed fraction (``0 ≤ w ≤ 1`` gives
   ``(1 - w)² ≤ 1 - w²``); the error at t = 0, which also holds the mesh's normalisation, is printed too.
6. **--refine N:** for N shapes of each family (deformed ones, spread over the dataset; a round torus
   too when --shapes names it), Richardson's estimate of the error of every packet in space and time
   (§6, ``richardson_errors``): the stored trajectory c (dt), the same packet at dt/2 on the same
   mesh, c½, and at dt on the mesh refined once on the same surface (``fine_mesh``), f, whose first
   vertices are the stored ones. Both are second order, so ``truth ≈ c - 4/3 (c - f) - 4/3 (c - c½)``
   and the error is ``max_t ‖4/3 (c - f) + 4/3 (c - c½)‖ / ‖c‖`` on the stored vertices. On the fine
   mesh the packet is the same function of x (centre vertex, direction t0 from φ and the stored mesh's
   normal, k0, σ), cut with that mesh's geodesic distance and normalised in its blended mass. The
   stored ``cn_error_estimate`` is recomputed from c and c½. Measured: 12 s per packet on a production
   sphere, 20 s on a torus (0.9 GB).
7. **Summary** per family and per packet set: errors (median, maximum), diagnostic ranges, draws and
   rejections of shapes and packets.

Errors are relative, in the lumped-mass norm on the stored vertices, the largest over the saved
times. With the default --evolve 1, checks 1-4 take 2.3 s per production sphere and 3.5 s per torus,
plus 2-3 s for the OOD set of a test shape: about 10 min for 200 shapes, besides hashing. Prints a report,
writes it as JSON with --output and exits with status 1 if a check fails: those of 1-4 and, in 6,
the stored ``cn_error_estimate`` not reproduced. The errors of 5 and 6 are estimates, without a
verdict. A shape's line gives the verdict of 1-4; a failure in 6 is printed below it.
"""
import os

if __name__ == "__main__":          # must happen before NumPy loads; importing this module changes nothing
    for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_variable] = "1"

import argparse
from collections import Counter
import json
from math import isclose, isfinite
from pathlib import Path
import time
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from schrodinger_glno.data_utils import _digest
from schrodinger_glno.geodesic import HeatGeodesics, cut_gaussian_packet
from schrodinger_glno.initial_conditions import max_wavenumber, spectral_fraction
from schrodinger_glno.mesh import blended_mass, cotan_laplacian, vertex_areas, vertex_normals
from schrodinger_glno.reference import torus_exact_solution
from schrodinger_glno.sampling import allowed_wavenumber, direction_from_angle
from schrodinger_glno.shapes import (AREA, GENUS, SphereDeformation, TorusDeformation, _base_torus, check_gates,
                                     mesh_quality, refine_shape, smallest_curvature_radius, sphere_radius,
                                     torus_surface, tube_quantities)
from schrodinger_glno.shapes_dataset import (CONFIG_FILE, DIAGNOSTIC_COLUMNS, FAMILIES, MANIFEST_FILE, SHAPE_FILE,
                                             ShapesConfig, _reason_kind, _shape_entry, round_torus_geometry,
                                             shapes_dataset_identity)
from schrodinger_glno.solver import build_cn_matrices, solve_CN

DEFORMATIONS = {"sphere": SphereDeformation, "torus": TorusDeformation}
COLUMN = {name: k for k, name in enumerate(DIAGNOSTIC_COLUMNS)}
# Largest values that pass. A sound dataset is far below: rounding (1e-15), float32 rounding (1e-7); the centre
# of a round torus is the area centroid of its mesh after the flips (measured 1e-5 at edge 0.12, 5e-7 at 0.03).
TOLERANCE = {"surface": 1e-9, "mass": 1e-12, "geo_feat": 1e-12, "offset": 1e-4, "norm": 1e-10,
             "recomputed_norm": 1e-6, "evolve": 1e-6, "input": 1e-6, "spectral": 1e-6, "basis_norm": 1e-5,
             "rayleigh": 1e-6, "cn_estimate": 1e-6}
GROUP_NAMES = {"sphere": "esferas", "torus": "toros", "train": "train", "val": "val", "test": "test", "ood": "ood",
               "all": "todo"}


def m_norms(u: NDArray, mass: NDArray) -> NDArray:
    """Lumped-mass norms of the columns of ``u`` (V, T)."""
    return np.sqrt(np.sum(mass[:, None] * np.abs(u)**2, axis=0))


def as_complex(channels: NDArray) -> NDArray:
    """Re and Im channels (last axis, float32) as complex128."""
    return channels[..., 0].astype(float) + 1j * channels[..., 1].astype(float)


def _same(a, b, rtol: float = 1e-10, atol: float = 1e-13) -> bool:
    """Whether two JSON-like values agree, numbers to ``rtol`` and ``atol``."""
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(
            _same(a[key], b[key], rtol, atol) for key in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) and all(
            _same(x, y, rtol, atol) for x, y in zip(a, b))
    if a is None or b is None or isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a == b and type(a) is type(b)
    return isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)


def _exceeds(value: float, limit: float) -> bool:
    """``value > limit``, true also for NaN."""
    return not value <= limit


def _strict(value):
    """``value`` with non-finite floats as None, for strict JSON."""
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def check_files(directory: Path, manifest: dict, ids: set[str]) -> dict:
    """Manifest identity, config.json and the SHA-256 of the data files of the shapes ``ids``."""
    problems = []
    try:
        identity = shapes_dataset_identity(directory, verify=False)
    except (KeyError, ValueError) as error:
        identity = None
        problems.append(f"manifest.json: {error}")
    if json.loads((directory / CONFIG_FILE).read_text()).get("config") != manifest.get("config"):
        problems.append("config.json: no coincide con la configuración del manifiesto")
    files = {name: digest for name, digest in manifest["sha256"].items() if name.split("/")[0] in ids}
    for name, digest in files.items():
        if not (directory / name).is_file():
            problems.append(f"{name}: falta")
        elif _digest(directory / name) != digest:
            problems.append(f"{name}: su SHA-256 difiere del manifiesto")
    return {"identity": identity, "files": len(files), "problems": problems}


def check_record(entry: dict, record: dict, hashes: dict) -> list[str]:
    """Problems of shape.json against its manifest ``entry`` and the manifest's ``hashes`` of its files.

    The entry is ``shapes_dataset._shape_entry`` of the record, so every field the identity covers is
    compared; a difference in ``quality`` or ``packets`` is named by its key.
    """
    try:
        own = _shape_entry(record)
    except (KeyError, TypeError, AttributeError) as error:
        return [f"shape.json: incompleto ({error!r})"]
    problems = []
    for key in list(entry) + [key for key in own if key not in entry]:
        mine, theirs = own.get(key), entry.get(key)
        if key in ("quality", "packets") and isinstance(mine, dict) and isinstance(theirs, dict):
            problems += [f"shape.json: {key}[{name}] difiere del manifiesto"
                         for name in list(theirs) + [name for name in mine if name not in theirs]
                         if name not in mine or name not in theirs or mine[name] != theirs[name]]
        elif key not in own or key not in entry or mine != theirs:
            problems.append(f"shape.json: {key} difiere del manifiesto")
    if record.get("sha256") != hashes:
        problems.append("shape.json: sha256 difiere del manifiesto")
    return problems


def surface_residual(deformation: SphereDeformation | TorusDeformation, verts: NDArray, quality: dict) -> float:
    """Largest distance, in mesh units, from a stored vertex to the surface of ``deformation``.

    The mesh is ``scale · (x - offset)`` with x on the surface (``shapes.build_shape``). On a sphere the
    distance is taken along the ray from the origin; on a torus to x(u_i, v_i) of vertex i of the base
    torus, whose order build_shape keeps.
    """
    scale = quality["scale"]
    x = verts / scale + np.asarray(quality["offset"], float)
    if isinstance(deformation, SphereDeformation):
        rho = np.linalg.norm(x, axis=1)
        return float(scale * np.abs(rho - sphere_radius(deformation, x / rho[:, None])).max())
    _, u, v = _base_torus(deformation, quality["target_edge"])
    if len(u) != len(verts):
        return float("inf")
    return float(scale * np.linalg.norm(torus_surface(deformation, u, v) - x, axis=1).max())


def check_entry(config: ShapesConfig, record: dict, mesh: dict) -> list[str]:
    """Problems of a shape's record against the config, and of the arrays of its mesh.npz."""
    index, verts = record["index"], mesh["verts"]
    expected = {"id": config.shape_id(index), "family": config.family(index), "split": config.split(index),
                "round": config.is_round(index), "seed": [config.seed, index], "n_verts": len(verts),
                "k_allowed": allowed_wavenumber(record["quality"]["mean_edge"], config.min_points_per_wavelength)}
    problems = [f"forma: {key} no es el de la configuración" for key, value in expected.items()
                if not _same(value, record.get(key))]
    sizes = {"faces": (len(mesh["faces"]), 3), "mass": (len(verts),), "evals": (config.k_eig,),
             "evecs": (len(verts), config.k_eig), "geo_feat": (len(verts),), "t_out": (config.output_steps,)}
    wrong = [key for key, size in sizes.items() if np.shape(mesh[key]) != size]
    if wrong:
        return problems + [f"mesh.npz: forma inesperada de {', '.join(wrong)}"]
    if not np.array_equal(mesh["t_out"], config.output_times()):
        problems.append("mesh.npz: t_out no son los tiempos guardados de la configuración")
    if _exceeds(np.abs(mesh["geo_feat"] - np.linalg.norm(verts - verts.mean(axis=0), axis=1)).max(),
                TOLERANCE["geo_feat"]):
        problems.append("mesh.npz: geo_feat no es la distancia al vértice medio")
    return problems


def check_mesh(config: ShapesConfig, record: dict, verts: NDArray, faces: NDArray,
               mass: NDArray) -> tuple[dict, list[str]]:
    """Quality and gates of a shape, recomputed from its mesh and deformation, against its record."""
    family, stored = record["family"], record["quality"]
    deformation = DEFORMATIONS[family](**record["deformation"])
    options = config.shape_options(record["index"])
    quality = mesh_quality(verts, faces)
    if family == "sphere":
        numbers = {"min_curvature_radius": smallest_curvature_radius(deformation, stored["scale"])}
        settings = {"subdivisions": config.sphere_subdivisions, "relax": config.sphere_relax}
    else:
        numbers = tube_quantities(deformation, stored["scale"], options["sigmas"])
        settings = {"target_edge": config.torus_target_edge}
    recomputed = quality | numbers | settings
    differ = [key for key, value in recomputed.items() if not _same(value, stored.get(key))]
    _, reasons = check_gates(quality | numbers, GENUS[family], numbers if family == "torus" else None,
                             **options["gates"])
    result = {key: quality[key] for key in ("min_angle", "max_angle", "n_negative", "edge_p95_ratio", "area")}
    result |= {key: numbers[key] for key in ("min_curvature_radius", "r_min", "fold", "hole") if key in numbers}
    result |= {"surface_residual": surface_residual(deformation, verts, stored),
               "mass_error": float(np.abs(vertex_areas(verts, faces) - mass).max() / mass.max())}
    problems = [f"malla: {key} recalculado difiere del registro" for key in differ]
    problems += [f"malla: no pasa la compuerta: {reason}" for reason in reasons]
    if not isclose(quality["area"], AREA, rel_tol=1e-9):
        problems.append(f"malla: área {quality['area']:.9f} ≠ 4π")
    if _exceeds(result["surface_residual"], TOLERANCE["surface"]):
        problems.append(f"malla: vértices a {result['surface_residual']:.1e} de la superficie de su deformación")
    if _exceeds(result["mass_error"], TOLERANCE["mass"]):
        problems.append("malla: la masa concentrada guardada no es vertex_areas")
    if record["round"]:
        geometry = record["exact_torus"]
        R, r = geometry["major_radius"], geometry["minor_radius"]
        x = verts - np.asarray(geometry["center"], float)
        result["exact_torus"] = {"radii": [R, r], "offset": float(np.linalg.norm(geometry["center"])),
                                 "fit": float(np.abs(np.hypot(np.hypot(x[:, 0], x[:, 1]) - R, x[:, 2]) - r).max())}
        if not _same(round_torus_geometry(deformation, stored), geometry):
            problems.append("toro de revolución: exact_torus recalculado de su deformación difiere del registro")
        if _exceeds(result["exact_torus"]["offset"], TOLERANCE["offset"]):
            problems.append(f"toro de revolución: centro a {result['exact_torus']['offset']:.1e} del origen")
        if _exceeds(result["exact_torus"]["fit"], TOLERANCE["surface"]):
            problems.append(f"toro de revolución: vértices a {result['exact_torus']['fit']:.1e} del toro s·R0, s·r0")
    elif record["exact_torus"] is not None:
        problems.append("forma: exact_torus en una forma que no es un toro de revolución")
    return result, problems


def load_packets(config: ShapesConfig, folder: Path, index: int, n_verts: int) -> tuple[dict, list[str]]:
    """Arrays of every packet set of a shape (outputs as complex128 states), and problems with their shapes."""
    data, problems = {}, []
    for name, prefix, n, _ in config.packet_sets(index):
        arrays = {key: np.load(folder / f"{prefix}{key}.npy") for key in
                  ("inputs", "outputs", "norms", "params", "diagnostics")}
        expected = {"inputs": (n, n_verts, 2), "outputs": (n, n_verts, config.output_steps, 2),
                    "norms": (n, config.output_steps), "params": (n, 4), "diagnostics": (n, len(DIAGNOSTIC_COLUMNS))}
        wrong = [key for key, shape in expected.items() if arrays[key].shape != shape]
        if wrong:
            problems.append(f"paquetes {name}: forma inesperada de {', '.join(wrong)}")
            continue
        data[name] = arrays | {"states": as_complex(arrays["outputs"]), "psi0": as_complex(arrays["inputs"])}
    return data, problems


def check_packets(config: ShapesConfig, record: dict, name: str, arrays: dict,
                  blended) -> tuple[dict, list[str]]:
    """Norms, t = 0 and parameters of one packet set."""
    norms, params, diagnostics = arrays["norms"], arrays["params"], arrays["diagnostics"]
    recomputed = np.stack([np.real(np.sum(state.conj() * (blended @ state), axis=0)) for state in arrays["states"]])
    result = {"norm_start": float(np.abs(norms[:, 0] - 1).max()),
              "norm_drift": float(np.abs(norms - norms[:, :1]).max()),
              "norm_recomputed": float(np.abs(recomputed - norms).max())}
    problems = [f"paquetes {name}: {label} {result[key]:.1e}" for key, label, limit in
                (("norm_start", "|ψᴴMψ - 1| en t = 0 de", "norm"), ("norm_drift", "deriva de ψᴴMψ de", "norm"),
                 ("norm_recomputed", "ψᴴMψ recalculada de outputs difiere de norms.npy en", "recomputed_norm"))
                if _exceeds(result[key], TOLERANCE[limit])]
    if not np.array_equal(arrays["outputs"][:, :, 0], arrays["inputs"]):
        problems.append(f"paquetes {name}: el estado en t = 0 no es el de entrada")
    k0_range, sigma_range, min_fraction = config.packet_rules(name)
    center, _, k0, sigma = params.T
    valid = ((center == np.round(center)) & (0 <= center) & (center < record["n_verts"])
             & (k0_range[0] <= k0) & (k0 <= k0_range[1]) & (sigma_range[0] <= sigma) & (sigma <= sigma_range[1])
             & (max_wavenumber(k0, sigma) <= record["k_allowed"] * (1 + 1e-12)))
    if not valid.all():
        problems.append(f"paquetes {name}: parámetros fuera de rango en {np.flatnonzero(~valid).tolist()}")
    if not (np.all(np.isfinite(diagnostics)) and diagnostics[:, COLUMN["spectral_fraction"]].min() >= min_fraction):
        problems.append(f"paquetes {name}: diagnósticos no finitos o fracción espectral bajo {min_fraction}")
    return result, problems


def rebuild_packet(config: ShapesConfig, mesh: dict, row: NDArray, distance: NDArray) -> tuple[NDArray, float]:
    """The cut packet of a params row on ``mesh`` (verts, blended, normals), as the generator builds it:
    ψ0 and the removed fraction."""
    center, phi, k0, sigma = int(row[0]), *row[1:]
    normals = mesh["normals"]
    return cut_gaussian_packet(mesh["verts"], mesh["blended"], normals, center, k0, sigma,
                               direction_from_angle(normals[center], phi), distance, config.cutoff_inner,
                               config.cutoff_outer)


def basis_errors(L, mass: NDArray, evals: NDArray, evecs: NDArray, chunk: int = 100) -> dict:
    """M-norms and Rayleigh quotients of the stored float32 eigenvectors against 1 and the stored eigenvalues."""
    norms, rayleigh = [], []
    for start in range(0, evecs.shape[1], chunk):            # in float64, ``chunk`` columns at a time
        phi = evecs[:, start:start + chunk].astype(float)
        norms.append(np.sum(mass[:, None] * phi**2, axis=0))
        rayleigh.append(np.sum(phi * (L @ phi), axis=0) / norms[-1])
    norms, rayleigh = np.concatenate(norms), np.concatenate(rayleigh)
    return {"norm": float(np.abs(norms - 1).max()), "rayleigh": float(np.abs(rayleigh - evals).max() / evals[-1])}


def exact_comparison(record: dict, data: dict, verts: NDArray, mass: NDArray, normals: NDArray,
                     times: NDArray) -> dict:
    """Round torus: per (set, packet), the errors against ``torus_exact_solution`` about its ``exact_torus``."""
    geometry = record["exact_torus"]
    R, r = geometry["major_radius"], geometry["minor_radius"]
    x = verts - np.asarray(geometry["center"], float)
    rows = {}
    for name, arrays in data.items():
        for i, (row, states) in enumerate(zip(arrays["params"], arrays["states"])):
            center, phi, k0, sigma = int(row[0]), *row[1:]
            exact = torus_exact_solution(x, x[center], direction_from_angle(normals[center], phi), k0, sigma, times,
                                         R, r)
            errors = m_norms(states - exact, mass) / m_norms(states, mass)
            removed = max(float(arrays["diagnostics"][i, COLUMN["removed_fraction"]]), 0.0)
            rows[name, i] = {"error": float(errors.max()), "start": float(errors[0]),
                             "cut": float(np.sqrt(removed) + removed)}
    return rows


def fine_mesh(record: dict, mesh: dict) -> dict:
    """The stored mesh refined once on the surface of its deformation (``shapes.refine_shape``), with its operators.

    The stored vertices are its first rows and keep their normals, so a packet rebuilt on it from the
    same params row (``rebuild_packet``) is the same function of x: only its geodesic cutoff and its
    blended normalisation are the fine mesh's own.
    """
    deformation = DEFORMATIONS[record["family"]](**record["deformation"])
    verts, faces = refine_shape(deformation, mesh["verts"], mesh["faces"], record["quality"])
    normals = vertex_normals(verts, faces)
    normals[:len(mesh["verts"])] = mesh["normals"]
    return {"verts": verts, "faces": faces, "L": cotan_laplacian(verts, faces), "blended": blended_mass(verts, faces),
            "normals": normals, "geodesics": HeatGeodesics(verts, faces)}


def richardson_errors(c: NDArray, f: NDArray, c_half: NDArray, mass: NDArray) -> dict:
    """Richardson's relative errors (largest over the saved times) of the stored trajectory ``c`` (V, T).

    ``f`` is the fine mesh's trajectory on the stored vertices and ``c_half`` the stored mesh's at dt/2.
    Both are second order, so ``4/3 (c - f)`` is the error of the mesh and ``4/3 (c - c½)`` that of
    the time step; ``error`` is that of their sum. ``cn_estimate`` is the generator's ``cn_error_estimate``.
    """
    size = m_norms(c, mass)
    space, step = 4 / 3 * (c - f), 4 / 3 * (c - c_half)
    return {"error": float(np.max(m_norms(space + step, mass) / size)),
            "space": float(np.max(m_norms(space, mass) / size)),
            "time": float(np.max(m_norms(step, mass) / size)),
            "cn_estimate": float(4 / 3 * np.max(m_norms(c - c_half, mass) / m_norms(c_half, mass)))}


def refinement_estimates(config: ShapesConfig, record: dict, data: dict, mesh: dict, geodesics: HeatGeodesics,
                         rows: dict, say: Callable[[str], None]) -> dict:
    """Richardson's estimate of the error of every packet of a shape (module docstring, item 6), into ``rows``."""
    n = len(mesh["verts"])
    fine = fine_mesh(record, mesh)
    steps, indices = config.steps, config.output_indices()
    half = build_cn_matrices(mesh["L"], mesh["blended"], np.zeros(n), config.dt / 2)
    fine_cn = build_cn_matrices(fine["L"], fine["blended"], np.zeros(len(fine["verts"])), config.dt)
    say(f"      refinamiento: malla fina de {len(fine['verts'])} vértices; "
        "error = espacio (4/3 (c - f)) + tiempo (4/3 (c - c½))")
    for name, arrays in data.items():
        for i, (row, c) in enumerate(zip(arrays["params"], arrays["states"])):
            center = int(row[0])
            psi0, _ = rebuild_packet(config, mesh, row, geodesics.distance(center))
            c_half = solve_CN(psi0, *half, 2 * steps - 1, 2 * indices)
            psi_fine, _ = rebuild_packet(config, fine, row, fine["geodesics"].distance(center))
            f = solve_CN(psi_fine, *fine_cn, steps, indices)[:n]
            estimate = richardson_errors(c, f, c_half, mesh["mass"])
            target = rows[name, i]
            target["refine"] = estimate
            ratio = (f"; refinamiento/exacta {estimate['error'] / target['exact']['error']:.2f}"
                     if "exact" in target else "")
            say(f"      refinamiento {name} {i}: k0 = {target['k0']:.2f}, σ = {target['sigma']:.3f}: error "
                f"{estimate['error']:.2%} (espacio {estimate['space']:.2%}, tiempo {estimate['time']:.2%}; CN "
                f"guardado {target['cn_error_estimate']:.2%}){ratio}")
    return {"fine_verts": len(fine["verts"])}


def check_shape(config: ShapesConfig, directory: Path, entry: dict, hashes: dict, *, spectral: int, evolve: int,
                exact: bool, refine: bool, say: Callable[[str], None]) -> dict:
    """Every check of one shape, printed as it goes; ``entry`` is its manifest entry and ``hashes`` the
    manifest's SHA-256 of its files."""
    start = time.perf_counter()
    folder = directory / entry["id"]
    record = json.loads((folder / SHAPE_FILE).read_text())
    problems = check_record(entry, record, hashes)
    record |= entry                     # from here on the manifest's copy, which its identity covers
    with np.load(folder / "mesh.npz") as stored:
        mesh = {key: stored[key] for key in stored.files}
    verts, faces, mass = mesh["verts"], mesh["faces"], mesh["mass"]
    result = {key: record[key] for key in ("id", "index", "family", "split", "round", "n_verts")}
    problems += check_entry(config, record, mesh)
    result["mesh"], found = check_mesh(config, record, verts, faces, mass)
    problems += found
    mesh |= {"L": cotan_laplacian(verts, faces), "blended": blended_mass(verts, faces),
             "normals": vertex_normals(verts, faces)}
    result["basis"] = basis_errors(mesh["L"], mass, mesh["evals"], mesh["evecs"])
    problems += [f"base float32: {label} {result['basis'][key]:.1e}" for key, label, limit in
                 (("norm", "|φᵀMφ - 1| =", "basis_norm"), ("rayleigh", "error relativo de Rayleigh", "rayleigh"))
                 if _exceeds(result["basis"][key], TOLERANCE[limit])]
    data, found = load_packets(config, folder, record["index"], len(verts))
    problems += found
    result["sets"], rows = {}, []
    for name, arrays in data.items():
        result["sets"][name], found = check_packets(config, record, name, arrays, mesh["blended"])
        problems += found
        for i, (row, diagnostics) in enumerate(zip(arrays["params"], arrays["diagnostics"])):
            rows.append({"set": name, "packet": i, "k0": float(row[2]), "sigma": float(row[3])}
                        | {column: float(value) for column, value in zip(DIAGNOSTIC_COLUMNS, diagnostics)}
                        | {"norm_drift": float(np.abs(arrays["norms"][i] - arrays["norms"][i, 0]).max())})
    by_key = {(row["set"], row["packet"]): row for row in rows}
    geodesics = HeatGeodesics(verts, faces) if spectral > 0 or refine else None
    cn = build_cn_matrices(mesh["L"], mesh["blended"], np.zeros(len(verts)), config.dt) if evolve > 0 else None
    for name, arrays in data.items():
        for i in range(min(evolve, len(arrays["params"]))):
            evolved = solve_CN(arrays["psi0"][i], *cn, config.steps, config.output_indices())
            error = float(np.max(m_norms(arrays["states"][i] - evolved, mass) / m_norms(evolved, mass)))
            by_key[name, i]["evolve_error"] = error
            if _exceeds(error, TOLERANCE["evolve"]):
                problems.append(f"paquetes {name} {i}: outputs difiere de Crank–Nicolson desde inputs en {error:.1e}")
        for i in range(min(spectral, len(arrays["params"]))):
            row = arrays["params"][i]
            psi0, removed = rebuild_packet(config, mesh, row, geodesics.distance(int(row[0])))
            fraction = spectral_fraction(psi0, mass, mesh["evecs"])
            check = {"input_error": float(np.abs(psi0 - arrays["psi0"][i]).max() / np.abs(psi0).max()),
                     "fraction": fraction, "difference": fraction - by_key[name, i]["spectral_fraction"],
                     "removed_difference": removed - by_key[name, i]["removed_fraction"]}
            by_key[name, i]["spectral"] = check
            if _exceeds(check["input_error"], TOLERANCE["input"]):
                problems.append(f"paquetes {name} {i}: ψ0 reconstruido de params difiere de la entrada "
                                f"({check['input_error']:.1e})")
            if _exceeds(abs(check["difference"]), TOLERANCE["spectral"]):
                problems.append(f"paquetes {name} {i}: fracción espectral (float32) difiere de la guardada "
                                f"en {check['difference']:.1e}")
    result |= {"packets": rows, "problems": problems,
               "shape_stats": {"attempts": record["attempts"], "rejections": record["rejections"],
                               "draws": {name: record["packets"][name]["draws"] for name in data},
                               "rejected": {name: record["packets"][name]["rejected"] for name in data}}}
    say(_shape_line(result, time.perf_counter() - start))
    for problem in problems:
        say(f"      FALLA {problem}")
    if exact and record["round"]:
        geometry = result["mesh"]["exact_torus"]
        say(f"      solución exacta: radios s·R0 = {geometry['radii'][0]:.4f}, s·r0 = {geometry['radii'][1]:.4f}; "
            f"centro a {geometry['offset']:.1e} del origen; vértices a {geometry['fit']:.0e} del toro")
        for key, value in exact_comparison(record, data, verts, mass, mesh["normals"], mesh["t_out"]).items():
            by_key[key]["exact"] = value
            say(f"      exacta {key[0]} {key[1]}: k0 = {by_key[key]['k0']:.2f}, σ = {by_key[key]['sigma']:.3f}: error "
                f"{value['error']:.2%} (en t = 0 {value['start']:.2%}; el corte cambia ψ0 en ≤ {value['cut']:.1e})")
    if refine:
        result["refinement"] = refinement_estimates(config, record, data, mesh, geodesics, by_key, say)
        for key, row in by_key.items():
            if _exceeds(abs(row["refine"]["cn_estimate"] - row["cn_error_estimate"]), TOLERANCE["cn_estimate"]):
                problems.append(f"paquetes {key[0]} {key[1]}: cn_error_estimate no se reproduce "
                                f"({row['refine']['cn_estimate']:.3e} contra {row['cn_error_estimate']:.3e})")
                say(f"      FALLA {problems[-1]}")
    return result


def _stats(values: list[float]) -> dict:
    return {"n": len(values), "median": float(np.median(values)) if values else None,
            "max": float(np.max(values)) if values else None}


def summarize(results: list[dict]) -> dict:
    """Errors, diagnostic ranges and draws per family, per packet set and over all checked packets."""
    groups = {}
    for key in ("sphere", "torus", "train", "val", "test", "ood", "all"):
        members = [(result, row) for result in results for row in result.get("packets", [])
                   if key in ("all", result["family"], row["set"])]
        if not members:
            continue
        shapes = {result["id"]: result for result, _ in members}
        sets = {(result["id"], row["set"]): result["shape_stats"] for result, row in members}
        diagnostics = np.array([[row[column] for column in DIAGNOSTIC_COLUMNS] for _, row in members])
        rejected = Counter()
        for (_, name), stats in sets.items():
            rejected.update(stats["rejected"][name])
        reasons = Counter(_reason_kind(reason) for result in shapes.values()
                          for draw in result["shape_stats"]["rejections"] for reason in draw)
        groups[key] = {
            "shapes": len(shapes), "packets": len(members),
            "shape_draws": sum(result["shape_stats"]["attempts"] for result in shapes.values()),
            "shape_rejections": dict(reasons),
            "packet_draws": sum(stats["draws"][name] for (_, name), stats in sets.items()),
            "packet_rejections": {reason: rejected[reason] for reason in ("thin", "leak", "spectral")},
            "diagnostics": {column: [float(low), float(high)] for column, low, high in
                            zip(DIAGNOSTIC_COLUMNS, diagnostics.min(axis=0), diagnostics.max(axis=0))},
            "cn_error_estimate": _stats(list(diagnostics[:, COLUMN["cn_error_estimate"]])),
            "norm_drift": max(row["norm_drift"] for _, row in members),
            "refine": _stats([row["refine"]["error"] for _, row in members if "refine" in row]),
            "exact": _stats([row["exact"]["error"] for _, row in members if "exact" in row])}
    return groups


def _select(entries: list[dict], names: list[str] | None) -> list[dict]:
    """Manifest entries of the shapes ``names`` (ids such as 003_torus, or indices), all if None."""
    if not names:
        return entries
    known = {name: entry for entry in entries for name in (entry["id"], str(entry["index"]), f"{entry['index']:03d}")}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(f"shapes not in the dataset: {unknown}")
    return sorted({known[name]["index"]: known[name] for name in names}.values(), key=lambda entry: entry["index"])


def _refined(entries: list[dict], n: int, explicit: bool) -> set[str]:
    """Ids of ``n`` shapes per family, spread evenly over ``entries``; round tori only if ``explicit``."""
    chosen = set()
    for family in FAMILIES:
        pool = [entry["id"] for entry in entries if entry["family"] == family and (explicit or not entry["round"])]
        if pool and n > 0:
            chosen |= {pool[k] for k in np.unique(np.linspace(0, len(pool) - 1, n).round().astype(int))}
    return chosen


def _shape_line(result: dict, seconds: float) -> str:
    """One line with the checks 1-4 of a shape and their verdict."""
    sets = result["sets"].values()
    spectral = [row["spectral"] for row in result["packets"] if "spectral" in row]
    evolved = [row["evolve_error"] for row in result["packets"] if "evolve_error" in row]
    family = "toro de revolución" if result["round"] else {"sphere": "esfera", "torus": "toro"}[result["family"]]
    line = (f"{result['id']:11s} {result['split']:5s} {family:18s} {result['n_verts']:6d} v  "
            f"{'ok   ' if not result['problems'] else 'FALLA'}  ángulos {result['mesh']['min_angle']:.1f}°-"
            f"{result['mesh']['max_angle']:.1f}°  superficie {result['mesh']['surface_residual']:.0e}")
    if sets:
        line += (f"  deriva {max(s['norm_drift'] for s in sets):.0e}"
                 f"  ψᴴMψ(outputs) {max(s['norm_recomputed'] for s in sets):.0e}")
    if evolved:
        line += f"  CN(inputs) {max(evolved):.0e}"
    if spectral:
        line += (f"  ψ0 {max(s['input_error'] for s in spectral):.0e}"
                 f"  Δfrac {max(abs(s['difference']) for s in spectral):.0e}")
    return line + f"  base {result['basis']['norm']:.0e}  ({seconds:.1f} s)"


def _print_summary(groups: dict) -> None:
    print("\nResumen (errores relativos en la norma de la masa concentrada, máximo en el tiempo):")
    for key, group in groups.items():
        rejected = group["packet_rejections"]
        line = (f"  {GROUP_NAMES[key]}: {group['shapes']} formas ({group['shape_draws']} sorteos), "
                f"{group['packets']} paquetes ({group['packet_draws']} sorteos; rechazados: delgado "
                f"{rejected['thin']}, fuga {rejected['leak']}, espectral {rejected['spectral']})")
        diagnostics = group["diagnostics"]
        cn = group["cn_error_estimate"]
        line += (f"\n      fracción espectral ≥ {diagnostics['spectral_fraction'][0]:.7f}; CN estimado: mediana "
                 f"{cn['median']:.2%}, máx. {cn['max']:.2%}; quitado por el corte ≤ "
                 f"{diagnostics['removed_fraction'][1]:.1e}; fuga ≤ {diagnostics['leak_fraction'][1]:.1e}; "
                 f"deriva ≤ {group['norm_drift']:.0e}")
        for name, label in (("refine", "refinamiento"), ("exact", "solución exacta")):
            if group[name]["n"]:
                line += (f"\n      {label}: {group[name]['n']} paquetes, mediana {group[name]['median']:.2%}, "
                         f"máx. {group[name]['max']:.2%}")
        if group["shape_rejections"]:
            line += "\n      formas rechazadas por: " + ", ".join(
                f"{reason} ({count})" for reason, count in Counter(group["shape_rejections"]).most_common())
        print(line)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/shapes"))
    p.add_argument("--shapes", nargs="+", help="Only these shapes (ids such as 003_torus, or indices)")
    p.add_argument("--evolve", type=int, default=1,
                   help="Packets per set evolved again from their input and compared with the outputs (default 1)")
    p.add_argument("--spectral", type=int, default=2,
                   help="Packets per set rebuilt from params, with their spectral fraction recomputed (default 2)")
    p.add_argument("--exact", action="store_true", help="Compare the round tori with their exact solution")
    p.add_argument("--refine", type=int, default=0, metavar="N",
                   help="Richardson estimate on N shapes per family (slow: 12-20 s per packet)")
    p.add_argument("--output", type=Path, help="Optional JSON report")
    args = p.parse_args(argv)
    directory = args.data_dir
    if not (directory / MANIFEST_FILE).is_file():
        raise SystemExit(f"{directory / MANIFEST_FILE} does not exist: the dataset is not complete")
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    config = ShapesConfig(**manifest["config"])
    selected = _select(manifest["shapes"], args.shapes)
    refined = _refined(selected, args.refine, explicit=bool(args.shapes))
    files = check_files(directory, manifest, {entry["id"] for entry in selected})
    print(f"dataset {str(files['identity'])[:16]}  {len(selected)} de {len(manifest['shapes'])} formas  "
          f"{files['files']} archivos verificados con SHA-256"
          + ("" if not files["problems"] else "\n  FALLA " + "\n  FALLA ".join(files["problems"])), flush=True)

    def say(message: str) -> None:
        print(message, flush=True)

    results = []
    for entry in selected:
        start = time.perf_counter()
        hashes = {name.split("/", 1)[1]: digest for name, digest in manifest["sha256"].items()
                  if name.split("/")[0] == entry["id"]}
        try:
            result = check_shape(config, directory, entry, hashes, spectral=args.spectral, evolve=args.evolve,
                                 exact=args.exact, refine=entry["id"] in refined, say=say)
        except Exception as error:                      # an unreadable file fails this shape, not the run
            result = {key: entry[key] for key in ("id", "index", "family", "split", "round", "n_verts")}
            result["problems"] = [f"error: {error!r}"]
            say(f"{entry['id']}  FALLA  {result['problems'][0]}")
        result["seconds"] = round(time.perf_counter() - start, 2)
        results.append(result)
    groups = summarize(results)
    _print_summary(groups)
    problems = files["problems"] + [f"{result['id']}: {problem}"
                                    for result in results for problem in result["problems"]]
    verdict = "Todo en orden" if not problems else f"FALLA: {len(problems)} problema{'s' * (len(problems) != 1)}"
    print(f"\n{verdict}; {len(results)} formas revisadas" + (f", {len(refined)} refinadas" if refined else "") + ".")
    report = {"data_dir": str(directory), "identity": files["identity"],
              "options": {"shapes": args.shapes, "evolve": args.evolve, "spectral": args.spectral, "exact": args.exact,
                          "refine": args.refine},
              "tolerances": TOLERANCE, "files": files, "shapes": results, "summary": groups,
              "problems": problems, "passed": not problems}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_strict(report), indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main()["passed"] else 1)
