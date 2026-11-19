"""Stage-3 datasets: Gaussian packets on many deformed spheres and tori, one directory per shape.

Every shape is its own sample of geometry (Generacion_datos_etapa3.md, §5 and §8): shape ``i``
is a sphere when i is even and a torus when it is odd, and every ``round_torus_every``-th
torus (``ShapesConfig.is_round``) is a torus of revolution with a random aspect ratio, whose
exact solution (``reference.torus_exact_solution``, with the radii and centre recorded as
``exact_torus`` by ``round_torus_geometry``) calibrates the data error. Splits go by index,
train first, then val and test, so both families are balanced in each and no shape is shared
between splits. Shape ``i``
depends only on ``(seed, i)``: ``np.random.default_rng([seed, i, stream])`` with one stream for
the shape (``shapes.make_shape``), one for its packets and one for its out-of-distribution
packets, so the data do not depend on the number of workers or on the order they finish in.

Per shape: the mesh of ``shapes.make_shape`` (area 4π), the cotangent Laplacian, the lumped
mass with its ``k_eig`` eigenpairs (the GLNO basis) and the blended mass of Crank–Nicolson,
factored at ``dt`` and at ``dt/2``. Each packet draws ``(center, phi, k0, sigma)`` as
``sampling.sample_packets`` does (centre ∝ lumped area, φ uniform, (k0, σ) uniform on the part
of the rectangle the mesh resolves) and is drawn again, with the reason counted, when

a. on a torus, the cross-section through the centre is thinner than ``min_tube_sigmas · σ``
   (``tube_section_radii``): in a thinner tube the packet wraps around it (§3, §4);
b. the chordal packet puts more than ``max_leak`` of its lumped norm beyond the geodesic
   distance ``leak_radius · σ`` from the centre: it reaches across the hole of a torus (§4);
c. the packet cut at 7σ–9σ (``geodesic.cut_gaussian_packet``) keeps less than
   ``min_spectral_fraction`` (``min_ood_spectral_fraction`` out of distribution) of its norm in
   the ``k_eig`` lumped eigenvectors, computed in float64 (§8.9: the packet is drawn again, not
   the shape).

The cut packet, normalised in the blended mass, is then evolved by Crank–Nicolson at ``dt``
(stored) and at ``dt/2`` (for the error estimate, §8.9), with the saved times of
``SurfaceConfig``: ``output_steps`` states including t = 0.

Layout of ``directory``::

    config.json                   the ShapesConfig; a run with another one is refused
    <id>/mesh.npz                 verts, faces, mass (lumped), evals, evecs, geo_feat, t_out
    <id>/inputs.npy               (n, V, 2)                  float32  Re, Im of ψ0
    <id>/outputs.npy              (n, V, output_steps, 2)    float32  saved states, t = 0 included
    <id>/norms.npy                (n, output_steps)          float64  ψᴴ M ψ, M the blended mass
    <id>/params.npy               (n, 4)                     float64  center, phi, k0, sigma
    <id>/diagnostics.npy          (n, 7)                     float64  DIAGNOSTIC_COLUMNS
    <id>/ood_*.npy                the same for the out-of-distribution packets (test shapes only)
    <id>/shape.json               deformation, quality, draws, packet statistics, file hashes, timings
    manifest.json                 settings, the shapes (shape.json without its timings and packet
                                  ranges), SHA-256 of every data file and an identity hash

``<id>`` is ``f"{i:03d}_{family}"``; ``n`` is ``packets_per_shape`` (``ood_packets_per_shape``
for ``ood_*``). ``mesh.npz`` has the keys of the stage-1 and stage-2 datasets: ``evals`` are
float64, but ``evecs`` are stored as float32 to halve the largest file of a shape (700 modes:
44 MB per torus instead of 87); rounding them changed the spectral fractions of a production
sphere and torus by less than 1e-8, far below the gates. ``geo_feat`` is the distance to the mean vertex, as before.

``DIAGNOSTIC_COLUMNS``:

- ``spectral_fraction``: of the cut ψ0 in the float64 lumped eigenbasis (rule c).
- ``cn_error_estimate``: ``(4/3) max_t ‖ψ_dt - ψ_dt/2‖ / ‖ψ_dt/2‖`` in the lumped-mass norm,
  Richardson's estimate of the time-stepping error of the stored trajectory (CN is second order).
- ``removed_fraction``: the part of the chordal packet's norm the geodesic cutoff removed.
- ``leak_fraction``: rule b's fraction beyond ``leak_radius · σ``.
- ``thin_draws``, ``leak_draws``, ``spectral_draws``: draws rejected by rules a, b and c before
  this packet was kept.

Writes resume: a shape is built in ``<id>.partial`` and renamed when complete, so an
interrupted run never leaves a complete-looking shape; a new run with the same config skips
the complete shapes and clears stale ``.partial`` directories. A run holds an exclusive
``flock`` on ``directory`` from start to end, so a second run on it (another ``families``,
say) is refused instead of clearing the ``.partial`` directories the first one is writing.
``manifest.json`` is written only when every shape exists. Its identity (as in
``schrodinger_fno.dataset``) hashes the settings, the shapes (seed, deformation, quality and
packet statistics of each) and the SHA-256 of every data file; ``shape.json`` is not hashed,
since it holds the timings, but ``shapes_dataset_identity(verify=True)`` checks it against
the manifest.

Shapes are built in spawned worker processes with single-threaded BLAS: as in stages 1 and 2,
ARPACK rotates the eigenbasis inside degenerate eigenspaces (the round tori have them) when
the thread count changes. With the same NumPy, SciPy and BLAS builds, 1 or N workers and a
resumed run give the same identity. Spawned workers import the caller's main module, so a
script that calls ``generate_shapes_dataset`` needs the ``if __name__ == "__main__":`` guard
and must not set the BLAS variables at import time (a worker that sees them changed refuses to
build). A worker that dies (killed, out of memory) stops the run with an error; run it again to
resume.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
import fcntl
from hashlib import sha256
import json
from math import isfinite
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import time
from typing import Callable, Iterable
import numpy as np
from numpy.typing import NDArray
from schrodinger_glno.data_utils import _digest, psi_to_channels
from .geodesic import HeatGeodesics, cut_gaussian_packet
from .initial_conditions import spectral_fraction
from .mesh import blended_mass, cotan_laplacian, laplacian_eigenpairs, vertex_areas, vertex_normals
from .sampling import PARAM_COLUMNS, allowed_wavenumber, direction_from_angle, resolved_fraction, sample_packets
from .shapes import AREA, TorusDeformation, _base_torus, make_shape, torus_surface, tube_log_radius
from .solver import build_cn_matrices, solve_CN

FORMAT_VERSION = 1
FAMILIES = ("sphere", "torus")
SPLITS = ("train", "val", "test")
SHAPE_STREAM, PACKET_STREAM, OOD_STREAM = 0, 1, 2
DIAGNOSTIC_COLUMNS = ("spectral_fraction", "cn_error_estimate", "removed_fraction", "leak_fraction",
                      "thin_draws", "leak_draws", "spectral_draws")
# Points per wavelength of the k_allowed in shapes.mesh_quality, which make_shape's min_wavenumber gate reads.
GATE_POINTS_PER_WAVELENGTH = 10
# shape.json entries the manifest copies (and its identity hashes), besides the packet statistics.
RECORD_KEYS = ("index", "id", "family", "split", "round", "seed", "exact_torus", "n_verts", "k_allowed", "attempts",
               "deformation", "quality")
PACKET_KEYS = ("packets", "draws", "rejected", "acceptance", "resolved_fraction")
BLAS_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
CONFIG_FILE, MANIFEST_FILE, SHAPE_FILE = "config.json", "manifest.json", "shape.json"
_IDENTITY_KEYS = ("format_version", "config", "counts", "time_convention", "param_columns", "diagnostic_columns",
                  "shapes", "sha256")
_INT_FIELDS = ("seed", "train_shapes", "val_shapes", "test_shapes", "packets_per_shape", "ood_packets_per_shape",
               "round_torus_every", "sphere_subdivisions", "sphere_l_max", "max_shape_draws", "output_steps", "k_eig",
               "max_packet_draws")
_FLOAT_FIELDS = ("sphere_slope", "torus_target_edge", "torus_a_max", "torus_b_max", "min_wavenumber",
                 "min_curvature_radius", "area", "final_time", "dt", "min_points_per_wavelength", "cutoff_inner",
                 "cutoff_outer", "min_tube_sigmas", "leak_radius", "max_leak", "min_spectral_fraction",
                 "min_ood_spectral_fraction")
_DEFORMATION_RANGES = ("sphere_tau_range", "torus_alpha_range", "torus_tau_range")
_PACKET_RANGES = ("k0_range", "sigma_range", "ood_k0_range", "ood_sigma_range")


@dataclass(frozen=True)
class ShapesConfig:
    """Settings of a stage-3 dataset; the defaults are the numbers adopted in §8.9.

    Shapes: ``train_shapes + val_shapes + test_shapes``, alternating sphere / torus; torus
    number j (= i // 2) is round when j is a multiple of ``round_torus_every`` (0: none).
    The sphere and torus fields go to ``shapes.make_shape`` and its samplers; round tori keep
    ``torus_alpha_range`` with no other deformation. Gates are the ``shapes.check_gates``
    defaults with ``min_wavenumber`` (k0 + 3/(2σ) = 8 + 3/0.36 ≈ 16.33 at the out-of-distribution corner),
    the wavenumber the mesh must resolve at ``min_points_per_wavelength`` like the packets, and
    ``min_curvature_radius`` (spheres only). ``area`` is fixed by ``shapes.build_shape``
    and only recorded. Time settings follow ``SurfaceConfig``. Packet rules: see the module docstring.
    """
    seed: int = 0
    train_shapes: int = 160
    val_shapes: int = 20
    test_shapes: int = 20
    packets_per_shape: int = 5
    ood_packets_per_shape: int = 5
    round_torus_every: int = 4
    sphere_subdivisions: int = 5
    sphere_relax: bool = True
    sphere_l_max: int = 4
    sphere_slope: float = 2.0
    sphere_tau_range: tuple[float, float] = (0.05, 0.15)
    torus_target_edge: float = 0.0305
    torus_alpha_range: tuple[float, float] = (0.45, 0.55)
    torus_a_max: float = 0.04
    torus_b_max: float = 0.06
    torus_tau_range: tuple[float, float] = (0.0, 0.075)
    min_wavenumber: float = 16.33
    min_curvature_radius: float = 0.18
    max_shape_draws: int = 100
    area: float = AREA
    final_time: float = 0.5
    dt: float = 0.001
    output_steps: int = 11
    k_eig: int = 700
    min_points_per_wavelength: float = 10.0
    k0_range: tuple[float, float] = (1.0, 6.0)
    sigma_range: tuple[float, float] = (0.15, 0.2)
    ood_k0_range: tuple[float, float] = (6.0, 8.0)
    ood_sigma_range: tuple[float, float] = (0.18, 0.2)
    cutoff_inner: float = 7.0
    cutoff_outer: float = 9.0
    min_tube_sigmas: float = 1.75
    leak_radius: float = 4.0
    max_leak: float = 5e-3
    min_spectral_fraction: float = 1 - 1e-5
    min_ood_spectral_fraction: float = 1 - 6e-5
    max_packet_draws: int = 200

    def __post_init__(self) -> None:
        # Accept JSON lists and NumPy scalars, but reject non-integral "integers" such as 5.5.
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or float(value) != int(value):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.sphere_relax, (bool, np.bool_)):
            raise ValueError("sphere_relax must be true or false")
        object.__setattr__(self, "sphere_relax", bool(self.sphere_relax))
        for name in _FLOAT_FIELDS:
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in _DEFORMATION_RANGES + _PACKET_RANGES:
            low, high = (float(x) for x in getattr(self, name))
            if not (isfinite(low) and isfinite(high) and 0 <= low <= high):
                raise ValueError(f"{name} must satisfy 0 <= low <= high")
            if name in _PACKET_RANGES and not low < high:
                raise ValueError(f"{name} must satisfy low < high")
            object.__setattr__(self, name, (low, high))
        if self.seed < 0 or min(self.train_shapes, self.val_shapes, self.test_shapes) < 0 or self.n_shapes < 1:
            raise ValueError("seed and shape counts must be >= 0, with at least one shape")
        if self.packets_per_shape < 1 or self.ood_packets_per_shape < 0 or self.round_torus_every < 0:
            raise ValueError("need packets_per_shape >= 1, ood_packets_per_shape >= 0 and round_torus_every >= 0")
        if self.sphere_subdivisions < 0 or self.sphere_l_max < 2:
            raise ValueError("need sphere_subdivisions >= 0 and sphere_l_max >= 2")
        if not (0 < self.torus_alpha_range[0] and self.torus_alpha_range[1] < 1):
            raise ValueError("torus_alpha_range must lie in (0, 1)")
        if self.torus_target_edge <= 0 or min(self.torus_a_max, self.torus_b_max) < 0:
            raise ValueError("need torus_target_edge > 0 and torus_a_max, torus_b_max >= 0")
        if min(self.min_wavenumber, self.min_curvature_radius) < 0:
            raise ValueError("min_wavenumber and min_curvature_radius must be >= 0")
        if self.max_shape_draws < 1 or self.max_packet_draws < 1:
            raise ValueError("max_shape_draws and max_packet_draws must be positive")
        if self.area != AREA:
            raise ValueError("shapes.build_shape scales every shape to area 4π")
        if self.final_time <= 0 or self.dt <= 0:
            raise ValueError("final_time and dt must be positive")
        n = self.final_time / self.dt
        if abs(n - round(n)) > 1e-9 * max(1.0, n) or round(n) < 1:
            raise ValueError("final_time must be a whole number of dt steps")
        if self.output_steps < 2 or (self.steps - 1) % (self.output_steps - 1):
            raise ValueError("output_steps - 1 must divide the number of time steps")
        if self.k_eig < 1 or self.min_points_per_wavelength <= 0:
            raise ValueError("k_eig and min_points_per_wavelength must be positive")
        if self.sigma_range[0] <= 0 or self.ood_sigma_range[0] <= 0:
            raise ValueError("sigma must be positive")
        if self.ood_k0_range[0] < self.k0_range[1]:
            raise ValueError("ood_k0_range must start at or above the training k0 range")
        if not 0 <= self.cutoff_inner < self.cutoff_outer:
            raise ValueError("need 0 <= cutoff_inner < cutoff_outer")
        if self.min_tube_sigmas < 0 or self.leak_radius <= 0 or not 0 < self.max_leak <= 1:
            raise ValueError("need min_tube_sigmas >= 0, leak_radius > 0 and 0 < max_leak <= 1")
        if not (0 < self.min_spectral_fraction <= 1 and 0 < self.min_ood_spectral_fraction <= 1):
            raise ValueError("spectral fractions must be in (0, 1]")

    @property
    def n_shapes(self) -> int:
        return self.train_shapes + self.val_shapes + self.test_shapes

    @property
    def steps(self) -> int:
        return int(round(self.final_time / self.dt)) + 1

    def output_indices(self) -> NDArray:
        return np.linspace(0, self.steps - 1, self.output_steps).round().astype(int)

    def output_times(self) -> NDArray:
        return self.output_indices() * self.dt

    def _check_index(self, index: int) -> None:
        if not 0 <= index < self.n_shapes:
            raise ValueError(f"shape index {index} is outside 0..{self.n_shapes - 1}")

    def family(self, index: int) -> str:
        self._check_index(index)
        return FAMILIES[index % 2]

    def is_round(self, index: int) -> bool:
        every = self.round_torus_every
        return self.family(index) == "torus" and every > 0 and (index // 2) % every == 0

    def split(self, index: int) -> str:
        self._check_index(index)
        if index < self.train_shapes:
            return "train"
        return "val" if index < self.train_shapes + self.val_shapes else "test"

    def shape_id(self, index: int) -> str:
        return f"{index:03d}_{self.family(index)}"

    def shape_options(self, index: int) -> dict:
        """Keyword arguments of ``shapes.make_shape`` for shape ``index`` (without its family and rng).

        The gap gate of a torus (``shapes.tube_quantities``) is evaluated at the smallest and the largest σ
        of any packet. The wavenumber gate reads ``shapes.mesh_quality``'s k_allowed, at 10 points per
        wavelength, so ``min_wavenumber`` is rescaled to ``min_points_per_wavelength`` (a factor 1 by default).
        """
        sigmas = (min(self.sigma_range[0], self.ood_sigma_range[0]), max(self.sigma_range[1], self.ood_sigma_range[1]))
        wavenumber = self.min_wavenumber * (self.min_points_per_wavelength / GATE_POINTS_PER_WAVELENGTH)
        options = {"max_attempts": self.max_shape_draws, "sigmas": sigmas,
                   "gates": {"min_wavenumber": wavenumber, "min_curvature_radius": self.min_curvature_radius}}
        if self.family(index) == "sphere":
            return options | {"subdivisions": self.sphere_subdivisions, "relax": self.sphere_relax,
                              "l_max": self.sphere_l_max, "slope": self.sphere_slope,
                              "tau_range": self.sphere_tau_range}
        options |= {"target_edge": self.torus_target_edge, "alpha_range": self.torus_alpha_range}
        if self.is_round(index):
            return options | {"a_max": 0.0, "b_max": 0.0, "tau_range": (0.0, 0.0)}
        return options | {"a_max": self.torus_a_max, "b_max": self.torus_b_max, "tau_range": self.torus_tau_range}

    def packet_sets(self, index: int) -> list[tuple[str, str, int, int]]:
        """``(name, file prefix, packets, rng stream)`` of each packet set of shape ``index``."""
        split = self.split(index)
        sets = [(split, "", self.packets_per_shape, PACKET_STREAM)]
        if split == "test" and self.ood_packets_per_shape:
            sets.append(("ood", "ood_", self.ood_packets_per_shape, OOD_STREAM))
        return sets

    def packet_rules(self, name: str) -> tuple[tuple[float, float], tuple[float, float], float]:
        """``(k0_range, sigma_range, min_spectral_fraction)`` of a packet set."""
        if name == "ood":
            return self.ood_k0_range, self.ood_sigma_range, self.min_ood_spectral_fraction
        return self.k0_range, self.sigma_range, self.min_spectral_fraction


def tube_section_radii(deformation: TorusDeformation, quality: dict, verts: NDArray, n_v: int = 128,
                       chunk: int = 1024) -> NDArray:
    """Smallest tube radius of the cross-section through each vertex of a torus shape, in the units of ``verts``.

    ``shapes.build_shape`` puts vertex i at x(u_i, v_i), with (u_i, v_i) the angles of vertex i of the
    round base torus (``shapes._base_torus``), and ``delaunay_flip`` only changes faces, so u_i is read
    from that torus; the vertices are checked against ``scale · (x(u_i, v_i) - offset)``. The radius
    is ``scale · r0 · e^{min_v t(u_i, v)}`` with the minimum over ``n_v`` angles v (t has |k| ≤ 2 in v).
    """
    _, u, v = _base_torus(deformation, quality["target_edge"])
    scale = quality["scale"]
    expected = scale * (torus_surface(deformation, u, v) - np.asarray(quality["offset"], float))
    if expected.shape != np.shape(verts) or np.abs(expected - verts).max() > 1e-9:
        raise RuntimeError("the torus vertices are not x(u, v) of the base torus, in its order")
    grid = 2 * np.pi * np.arange(n_v) / n_v
    t_min = np.concatenate([tube_log_radius(deformation, part[:, None], grid[None, :]).min(axis=1)
                            for part in np.array_split(u, max(1, -(-len(u) // chunk)))])
    return scale * deformation.minor_radius * np.exp(t_min)


def round_torus_geometry(deformation: TorusDeformation, quality: dict) -> dict:
    """Radii and centre of a round torus shape in the units of its mesh, for ``reference.torus_exact_solution``.

    The mesh is ``scale · (x - offset)`` with x on the torus (R0, r0) about the origin, so its radii are
    ``R0 · scale`` and ``r0 · scale`` (§8.9: not the nominal ones) and its centre is ``-scale · offset``,
    to be subtracted from the vertices and the packet centre before calling the reference.
    """
    if any(deformation.a + deformation.b + deformation.c):
        raise ValueError("not a torus of revolution")
    scale = quality["scale"]
    return {"major_radius": deformation.major_radius * scale, "minor_radius": deformation.minor_radius * scale,
            "center": [-scale * x for x in quality["offset"]]}


def _shape_mesh(config: ShapesConfig, shape) -> dict:
    """Operators, GLNO eigenpairs (lumped mass, float64) and the solver's blended mass of one shape."""
    verts, faces = shape.verts, shape.faces
    L, lumped = cotan_laplacian(verts, faces), vertex_areas(verts, faces)
    evals, evecs = laplacian_eigenpairs(L, lumped, config.k_eig)
    return {"verts": verts, "faces": faces, "L": L, "mass": lumped, "blended": blended_mass(verts, faces),
            "normals": vertex_normals(verts, faces), "evals": evals, "evecs": evecs,
            "k_allowed": allowed_wavenumber(shape.quality["mean_edge"], config.min_points_per_wavelength)}


def _draw_packet(config: ShapesConfig, mesh: dict, rng: np.random.Generator, name: str,
                 stats: dict) -> tuple[NDArray, NDArray, NDArray]:
    """One packet of set ``name`` that passes rules a-c: its row, the cut ψ0 and its first diagnostics.

    Every rejected draw is counted in ``stats`` and in the returned diagnostics; the whole row
    is drawn again, since each rule depends on the centre and σ.
    """
    k0_range, sigma_range, min_fraction = config.packet_rules(name)
    verts, lumped, sections = mesh["verts"], mesh["mass"], mesh["sections"]
    rejected = {"thin": 0, "leak": 0, "spectral": 0}
    for _ in range(config.max_packet_draws):
        stats["draws"] += 1
        drawn = {}
        row = sample_packets(1, lumped, rng, k0_range, sigma_range, mesh["k_allowed"], drawn)[0]
        stats["kd_tried"] += drawn["tried"]
        stats["kd_accepted"] += drawn["accepted"]
        center, phi, k0, sigma = int(row[0]), *row[1:]
        if sections is not None and sections[center] < config.min_tube_sigmas * sigma:
            rejected["thin"] += 1
            continue
        distance = mesh["geodesics"].distance(center)
        # |ψ|² of the chordal packet, which does not depend on k0 or φ
        density = lumped * np.exp(-np.sum((verts - verts[center])**2, axis=1) / (2 * sigma**2))
        leak = float(density[distance > config.leak_radius * sigma].sum() / density.sum())
        if leak > config.max_leak:
            rejected["leak"] += 1
            continue
        direction = direction_from_angle(mesh["normals"][center], phi)
        psi0, removed = cut_gaussian_packet(verts, mesh["blended"], mesh["normals"], center, k0, sigma, direction,
                                            distance, config.cutoff_inner, config.cutoff_outer)
        fraction = spectral_fraction(psi0, lumped, mesh["evecs"])
        if fraction < min_fraction:
            rejected["spectral"] += 1
            continue
        for reason, count in rejected.items():
            stats["rejected"][reason] += count
        return row, psi0, np.array([fraction, removed, leak, rejected["thin"], rejected["leak"], rejected["spectral"]])
    raise RuntimeError(f"no {name} packet passed the rules in {config.max_packet_draws} draws: {rejected}")


def _cn_error_estimate(coarse: NDArray, fine: NDArray, mass: NDArray) -> float:
    """``(4/3) max_t ‖ψ_dt - ψ_dt/2‖_M / ‖ψ_dt/2‖_M`` (lumped M): Richardson for a second-order scheme."""
    difference = np.sqrt(np.sum(mass[:, None] * np.abs(coarse - fine)**2, axis=0))
    return float(4 / 3 * np.max(difference / np.sqrt(np.sum(mass[:, None] * np.abs(fine)**2, axis=0))))


def _write_packets(config: ShapesConfig, mesh: dict, solvers: tuple, folder: Path, index: int, name: str,
                   prefix: str, n: int, stream: int) -> dict:
    """Draw, evolve and save one packet set of a shape through memmaps; return its statistics."""
    rng = np.random.default_rng([config.seed, index, stream])
    (LU, U1), (LU_half, U1_half) = solvers
    steps, indices = config.steps, config.output_indices()
    n_vertices, n_out = len(mesh["verts"]), config.output_steps
    inputs = np.lib.format.open_memmap(folder / f"{prefix}inputs.npy", mode="w+", dtype=np.float32,
                                       shape=(n, n_vertices, 2))
    outputs = np.lib.format.open_memmap(folder / f"{prefix}outputs.npy", mode="w+", dtype=np.float32,
                                        shape=(n, n_vertices, n_out, 2))
    params, norms = np.zeros((n, len(PARAM_COLUMNS))), np.zeros((n, n_out))
    diagnostics = np.zeros((n, len(DIAGNOSTIC_COLUMNS)))
    stats = {"draws": 0, "rejected": {"thin": 0, "leak": 0, "spectral": 0}, "kd_tried": 0, "kd_accepted": 0}
    for i in range(n):
        params[i], psi0, (fraction, *checks) = _draw_packet(config, mesh, rng, name, stats)
        trajectory = solve_CN(psi0, LU, U1, steps, indices)
        half_steps = solve_CN(psi0, LU_half, U1_half, 2 * steps - 1, 2 * indices)
        norms[i] = np.real(np.einsum("vt,vt->t", trajectory.conj(), mesh["blended"] @ trajectory))
        diagnostics[i] = [fraction, _cn_error_estimate(trajectory, half_steps, mesh["mass"]), *checks]
        inputs[i], outputs[i] = psi_to_channels(psi0), psi_to_channels(trajectory)
    inputs.flush(); outputs.flush()
    del inputs, outputs
    np.save(folder / f"{prefix}norms.npy", norms)
    np.save(folder / f"{prefix}params.npy", params)
    np.save(folder / f"{prefix}diagnostics.npy", diagnostics)
    k0_range, sigma_range, _ = config.packet_rules(name)
    column = dict(zip(DIAGNOSTIC_COLUMNS, diagnostics.T))
    return stats | {"packets": n, "acceptance": stats["kd_accepted"] / stats["kd_tried"],
                    "resolved_fraction": resolved_fraction(k0_range, sigma_range, mesh["k_allowed"]),
                    "k0": [float(params[:, 2].min()), float(params[:, 2].max())],
                    "sigma": [float(params[:, 3].min()), float(params[:, 3].max())],
                    "min_spectral_fraction": float(column["spectral_fraction"].min()),
                    "max_cn_error_estimate": float(column["cn_error_estimate"].max()),
                    "max_removed_fraction": float(column["removed_fraction"].max()),
                    "max_leak_fraction": float(column["leak_fraction"].max()),
                    "max_norm_drift": float(np.abs(norms - norms[:, :1]).max())}


def write_shape(config: ShapesConfig, directory: Path, index: int) -> dict:
    """Build shape ``index`` with its packets in ``directory/<id>`` and return its ``shape.json`` record.

    Works in ``<id>.partial`` (cleared first) and renames it when every file is written. A
    complete ``<id>`` is left as it is and its stored record returned. Neither limits the BLAS
    threads nor locks ``directory``: ``generate_shapes_dataset`` does both and calls it in its workers.
    """
    directory = Path(directory)
    shape_id, family, split = config.shape_id(index), config.family(index), config.split(index)
    target, folder = directory / shape_id, directory / f"{shape_id}.partial"
    if target.is_dir():
        return json.loads((target / SHAPE_FILE).read_text())
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    clock = {"start": time.perf_counter()}

    def lap(name: str) -> None:
        now = time.perf_counter()
        clock[name], clock["last"] = now - clock.get("last", clock["start"]), now

    shape = make_shape(family, np.random.default_rng([config.seed, index, SHAPE_STREAM]), **config.shape_options(index))
    lap("shape")
    mesh = _shape_mesh(config, shape)
    verts = mesh["verts"]
    np.savez(folder / "mesh.npz", verts=verts, faces=mesh["faces"], mass=mesh["mass"], evals=mesh["evals"],
             evecs=mesh["evecs"].astype(np.float32), geo_feat=np.linalg.norm(verts - verts.mean(axis=0), axis=1),
             t_out=config.output_times())
    lap("eigenpairs")
    free = np.zeros(len(verts))                                    # V = 0
    solvers = (build_cn_matrices(mesh["L"], mesh["blended"], free, config.dt),
               build_cn_matrices(mesh["L"], mesh["blended"], free, config.dt / 2))
    mesh["geodesics"] = HeatGeodesics(verts, mesh["faces"])
    mesh["sections"] = tube_section_radii(shape.deformation, shape.quality, verts) if family == "torus" else None
    lap("factorizations")
    packets = {name: _write_packets(config, mesh, solvers, folder, index, name, prefix, n, stream)
               for name, prefix, n, stream in config.packet_sets(index)}
    lap("packets")
    hashes = {path.name: _digest(path) for path in sorted(folder.iterdir())}
    lap("hashes")
    record = {"format_version": FORMAT_VERSION, "index": index, "id": shape_id, "family": family, "split": split,
              "round": config.is_round(index), "seed": [config.seed, index],
              "exact_torus": round_torus_geometry(shape.deformation, shape.quality) if config.is_round(index) else None,
              "deformation": asdict(shape.deformation), "quality": shape.quality, "attempts": shape.attempts,
              "rejections": [list(reasons) for reasons in shape.rejections], "n_verts": len(verts),
              "k_allowed": mesh["k_allowed"], "max_eigenvalue": float(mesh["evals"][-1]), "packets": packets,
              "sha256": hashes,
              "timings": {key: round(value, 3) for key, value in clock.items() if key not in ("start", "last")}
              | {"total": round(time.perf_counter() - clock["start"], 3)}}
    (folder / SHAPE_FILE).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    os.replace(folder, target)
    return record


def _shape_task(config: ShapesConfig, directory: str, index: int) -> dict:
    """Worker entry point: ``write_shape``, with the shape's id in any error.

    Refuses to build when the BLAS variables are not the "1" ``generate_shapes_dataset`` set
    before spawning: the caller's main module, which every spawned worker imports, changed them.
    """
    changed = {name: os.environ.get(name) for name in BLAS_VARIABLES if os.environ.get(name) != "1"}
    if changed:
        raise RuntimeError(f"shape {config.shape_id(index)}: the worker's BLAS is not single-threaded ({changed}); "
                           "the caller's main module must not set these variables at import time")
    try:
        record = write_shape(config, Path(directory), index)
    except Exception as error:
        raise RuntimeError(f"shape {config.shape_id(index)}: {error!r}") from error
    return {key: record[key] for key in ("id", "split", "attempts", "n_verts", "timings")}


def _lock(directory: Path) -> int:
    """An exclusive ``flock`` on ``directory`` (its descriptor); it ends when that is closed or the process exits."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise BlockingIOError(f"{directory} is in use by another run; run the families one after the other, "
                              "not at the same time") from None
    return descriptor


def _claim(directory: Path, config: ShapesConfig) -> None:
    """Store the config in an empty ``directory`` (which exists), or check that it matches the one stored there."""
    path = directory / CONFIG_FILE
    if path.exists():
        stored = json.loads(path.read_text())
        try:
            same = stored.get("format_version") == FORMAT_VERSION and ShapesConfig(**stored["config"]) == config
        except (TypeError, ValueError, KeyError):
            same = False
        if not same:
            raise FileExistsError(f"{directory} holds a shapes dataset made with a different config; "
                                  "use another directory to make a new one")
        return
    if any(directory.iterdir()):
        raise FileExistsError(f"Destination is not empty: {directory}")
    _write_json(path, {"format_version": FORMAT_VERSION, "config": asdict(config)})


def _write_json(path: Path, data: dict) -> None:
    """Strict JSON, written to a temporary file and renamed, so it is never seen half-written."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _identity(manifest: dict) -> str:
    payload = {key: manifest[key] for key in _IDENTITY_KEYS}
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _shape_entry(record: dict) -> dict:
    """The manifest entry of a shape: its ``shape.json`` record without the timings, hashes and packet ranges.

    The deformation and the quality (with ``scale``, ``offset`` and the mesh options) are what rebuild
    and refine the shape, so the identity covers them.
    """
    return {key: record[key] for key in RECORD_KEYS} | {
        "packets": {name: {key: stats[key] for key in PACKET_KEYS} for name, stats in record["packets"].items()}}


def _manifest(config: ShapesConfig, directory: Path) -> dict:
    """The manifest, from the ``shape.json`` of every shape."""
    shapes, hashes = [], {}
    for index in range(config.n_shapes):
        record = json.loads((directory / config.shape_id(index) / SHAPE_FILE).read_text())
        shapes.append(_shape_entry(record))
        hashes |= {f"{record['id']}/{name}": digest for name, digest in record["sha256"].items()}
    splits = [config.split(i) for i in range(config.n_shapes)]
    packets = {split: splits.count(split) * config.packets_per_shape for split in SPLITS}
    counts = {"shapes": {split: splits.count(split) for split in SPLITS},
              "families": {f: sum(config.family(i) == f for i in range(config.n_shapes)) for f in FAMILIES},
              "round_tori": sum(config.is_round(i) for i in range(config.n_shapes)),
              "packets": packets | {"ood": splits.count("test") * config.ood_packets_per_shape}}
    manifest = {"format_version": FORMAT_VERSION, "config": asdict(config), "counts": counts,
                "time_convention": "states include t=0", "param_columns": list(PARAM_COLUMNS),
                "diagnostic_columns": list(DIAGNOSTIC_COLUMNS), "shapes": shapes, "sha256": hashes}
    manifest["identity"] = _identity(manifest)
    return manifest


def shapes_dataset_identity(directory: Path, *, verify: bool = True) -> str:
    """Return the manifest identity; ``verify`` also checks every data file against its SHA-256 and every
    ``shape.json`` (not hashed: it holds the timings) against its manifest entry and hashes."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    if manifest.get("format_version") != FORMAT_VERSION or manifest["identity"] != _identity(manifest):
        raise ValueError("Dataset manifest identity is invalid")
    if verify:
        for relative, expected in manifest["sha256"].items():
            if _digest(directory / relative) != expected:
                raise ValueError(f"Dataset file differs from manifest: {relative}")
        for entry in manifest["shapes"]:
            record = json.loads((directory / entry["id"] / SHAPE_FILE).read_text())
            hashes = {name: digest for name, digest in manifest["sha256"].items()
                      if name.split("/", 1)[0] == entry["id"]}
            try:
                same = _shape_entry(record) == entry and {
                    f"{entry['id']}/{name}": digest for name, digest in record["sha256"].items()} == hashes
            except (KeyError, TypeError, AttributeError):
                same = False
            if not same:
                raise ValueError(f"Dataset file differs from manifest: {entry['id']}/{SHAPE_FILE}")
    return manifest["identity"]


def _set_environment(values: dict) -> dict:
    """Set (or, for None, remove) environment variables; return their previous values."""
    previous = {name: os.environ.get(name) for name in values}
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    return previous


def generate_shapes_dataset(config: ShapesConfig, directory: Path, *, workers: int = 1,
                            families: Iterable[str] | None = None,
                            progress: Callable[[str], None] | None = None) -> dict | None:
    """Build the missing shapes of ``families`` (default: all) with ``workers`` processes; return the manifest.

    The manifest is written, and returned, only when every shape of the dataset exists;
    otherwise this returns None. A directory made with another config is refused
    (``FileExistsError``), as is a non-empty directory that is not a shapes dataset and one that
    another run holds (``BlockingIOError``). Workers are spawned processes with single-threaded
    BLAS, even for ``workers=1``; each one builds whole shapes (``write_shape``) and ``progress``
    gets one line per finished shape, with its timings. A failed shape stops the run once the
    shapes in progress are complete; a worker that dies stops it at once (``RuntimeError``).
    """
    directory = Path(directory)
    families = FAMILIES if families is None else tuple(families)
    if not set(families) <= set(FAMILIES) or not families:
        raise ValueError(f"families must be some of {FAMILIES}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    directory.mkdir(parents=True, exist_ok=True)
    lock = _lock(directory)
    try:
        return _generate(config, directory, workers, families, progress or (lambda message: None))
    finally:
        os.close(lock)


def _generate(config: ShapesConfig, directory: Path, workers: int, families: tuple[str, ...],
              say: Callable[[str], None]) -> dict | None:
    """``generate_shapes_dataset`` in a locked ``directory``."""
    _claim(directory, config)
    for stale in sorted(directory.glob("*.partial")):
        shutil.rmtree(stale)
        say(f"cleared {stale.name}, left by an interrupted run")
    missing = [i for i in range(config.n_shapes) if not (directory / config.shape_id(i)).is_dir()]
    pending = [i for i in missing if config.family(i) in families]
    if missing:
        (directory / MANIFEST_FILE).unlink(missing_ok=True)
    say(f"{config.n_shapes - len(missing)} of {config.n_shapes} shapes complete; " + (
        f"building {len(pending)} ({', '.join(families)}) with {min(workers, len(pending))} workers" if pending
        else "nothing to build"))
    if pending:
        start = time.perf_counter()
        previous = _set_environment(dict.fromkeys(BLAS_VARIABLES, "1"))     # inherited by the spawned workers
        try:
            # Not multiprocessing.Pool: it replaces a killed worker and waits forever for its shape.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(min(workers, len(pending)), mp_context=context) as pool:
                futures = [pool.submit(_shape_task, config, str(directory), index) for index in pending]
                try:
                    for done, future in enumerate(as_completed(futures), 1):
                        result = future.result()
                        t = result["timings"]
                        left = (time.perf_counter() - start) / done * (len(pending) - done) / 60
                        say(f"{result['id']} ({result['split']}, {result['n_verts']} vertices): {t['total']:.1f} s = "
                            f"shape {t['shape']:.1f} s ({result['attempts']} draws) + "
                            f"eigenpairs {t['eigenpairs']:.1f} s + factorizations {t['factorizations']:.1f} s + "
                            f"packets {t['packets']:.1f} s; {done}/{len(pending)} built, ~{left:.1f} min left")
                except BrokenProcessPool as error:
                    raise RuntimeError("a worker process died (killed, or out of memory?); "
                                       "run again to resume") from error
                except BaseException:
                    pool.shutdown(cancel_futures=True)      # finish the shapes in progress, start no other
                    raise
        finally:
            _set_environment(previous)
    if any(not (directory / config.shape_id(i)).is_dir() for i in missing):
        say(f"{sum(not (directory / config.shape_id(i)).is_dir() for i in missing)} shapes are still missing; "
            "the manifest is written once every shape exists")
        return None
    manifest = _manifest(config, directory)
    _write_json(directory / MANIFEST_FILE, manifest)
    say(f"manifest written: identity {manifest['identity'][:16]}")
    return manifest


def _reason_kind(reason: str) -> str:
    """A gate failure without its numbers, e.g. ``"min angle #° < #°"``."""
    return re.sub(r"(?<![A-Za-z\d])-?\d+(\.\d+)?(e[-+]?\d+)?", "#", reason)


def dataset_summary(directory: Path) -> dict:
    """Counts, rejection rates of shapes and packets, diagnostic ranges and timings of the complete shapes."""
    directory = Path(directory)
    config = ShapesConfig(**json.loads((directory / CONFIG_FILE).read_text())["config"])
    records = [json.loads((directory / config.shape_id(i) / SHAPE_FILE).read_text())
               for i in range(config.n_shapes) if (directory / config.shape_id(i)).is_dir()]
    shapes = {family: {"built": 0, "draws": 0, "reasons": {}, "seconds": 0.0} for family in FAMILIES}
    packets, ranges, drift = {}, {}, {}
    for record in records:
        entry = shapes[record["family"]]
        entry["built"] += 1
        entry["draws"] += record["attempts"]
        entry["seconds"] += record["timings"]["total"]
        for reasons in record["rejections"]:
            for reason in reasons:
                kind = _reason_kind(reason)
                entry["reasons"][kind] = entry["reasons"].get(kind, 0) + 1
        for name, prefix, _, _ in config.packet_sets(record["index"]):
            stats = record["packets"][name]
            total = packets.setdefault(name, {"packets": 0, "draws": 0, "thin": 0, "leak": 0, "spectral": 0})
            total["packets"] += stats["packets"]
            total["draws"] += stats["draws"]
            for reason, count in stats["rejected"].items():
                total[reason] += count
            columns = np.load(directory / record["id"] / f"{prefix}diagnostics.npy")
            low, high = columns.min(axis=0), columns.max(axis=0)
            if name in ranges:
                low, high = np.minimum(low, ranges[name][0]), np.maximum(high, ranges[name][1])
            ranges[name] = (low, high)
            norms = np.load(directory / record["id"] / f"{prefix}norms.npy")
            drift[name] = max(drift.get(name, 0.0), float(np.abs(norms - norms[:, :1]).max()))
    for entry in shapes.values():
        entry["rejection_rate"] = 1 - entry["built"] / entry["draws"] if entry["draws"] else None
        entry["mean_seconds"] = entry.pop("seconds") / entry["built"] if entry["built"] else None
    for name, total in packets.items():
        total["rejection_rate"] = 1 - total["packets"] / total["draws"]
        total["diagnostics"] = {column: [float(low), float(high)]
                                for column, low, high in zip(DIAGNOSTIC_COLUMNS, *ranges[name])}
        total["max_norm_drift"] = drift[name]
    summary = {"shapes": {"total": config.n_shapes, "built": len(records),
                          "by_split": {s: sum(r["split"] == s for r in records) for s in SPLITS},
                          "round_tori": sum(r["round"] for r in records), **shapes},
               "packets": packets}
    if (directory / MANIFEST_FILE).exists():
        summary["identity"] = json.loads((directory / MANIFEST_FILE).read_text())["identity"]
    return summary
