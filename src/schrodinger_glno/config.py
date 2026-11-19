"""Settings for the datasets: Gaussian packets on a fixed surface with V = 0.

``surface`` selects the domain:

- ``"sphere"``: an icosphere with ``subdivisions`` and ``radius`` (stage 1);
- ``"torus"``: a torus of revolution with ``major_radius`` and ``minor_radius``,
  meshed with near-equilateral triangles of edge ≈ ``target_edge`` (stage 2).
  ``torus_config()`` gives its default packet ranges and ``k_eig = 800``
  (``SurfaceConfig(surface="torus")`` keeps the sphere's ranges and ``k_eig = 400``).

``steps`` counts states including t = 0, as in ``schrodinger_fno.config``; the
saved states are evenly spaced and include both t = 0 and ``final_time``.

``mass`` chooses the mass matrix of the time stepper: ``"blended"`` (average of
the lumped and consistent matrices, the default) keeps the phase of travelling
packets accurate; ``"lumped"`` is the diagonal mass the GLNO uses for its
eigenbasis, whose eigenvalues are too low (2 % at ℓ = 15 on the sphere), which
shifts the phase of fast packets by up to ~20 % at t = 0.5. The eigenpairs saved
for the GLNO always use the lumped mass, as in the paper's code. On the icosphere
the blended mass is almost exact (+0.02 % at λ = 240); on the torus mesh its
eigenvalues are still too high by ~0.1 % in the resolved band (second order in the
edge), a phase lead about as large as the phase lag of Crank–Nicolson: see ``torus_config``.
"""
from dataclasses import asdict, dataclass
from math import isfinite
import numpy as np

SURFACES = ("sphere", "torus")
MASS_CHOICES = ("blended", "lumped")
_INT_FIELDS = ("subdivisions", "output_steps", "k_eig")
_FLOAT_FIELDS = ("radius", "major_radius", "minor_radius", "target_edge", "final_time", "dt",
                 "min_points_per_wavelength")
_RANGE_FIELDS = ("k0_range", "sigma_range", "ood_k0_range")
_GEOMETRY_FIELDS = {"sphere": ("subdivisions", "radius"),
                    "torus": ("surface", "major_radius", "minor_radius", "target_edge")}


@dataclass(frozen=True)
class SurfaceConfig:
    surface: str = "sphere"
    subdivisions: int = 5            # sphere
    radius: float = 1.0              # sphere
    major_radius: float = 1.0        # torus
    minor_radius: float = 0.5        # torus
    target_edge: float = 0.03        # torus: 25 346 vertices
    k0_range: tuple[float, float] = (2.0, 8.0)
    sigma_range: tuple[float, float] = (0.2, 0.35)
    ood_k0_range: tuple[float, float] = (8.0, 10.0)
    final_time: float = 0.5
    dt: float = 0.001
    output_steps: int = 11
    k_eig: int = 400
    min_points_per_wavelength: float = 10.0
    mass: str = "blended"

    def __post_init__(self) -> None:
        # Accept JSON lists and NumPy scalars, but reject non-integral "integers" such as 5.5.
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or float(value) != int(value):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        for name in _FLOAT_FIELDS:
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in _RANGE_FIELDS:
            low, high = getattr(self, name)
            object.__setattr__(self, name, (float(low), float(high)))
        if self.surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}")
        if self.mass not in MASS_CHOICES:
            raise ValueError(f"mass must be one of {MASS_CHOICES}")
        if self.surface == "sphere" and (self.subdivisions < 0 or not isfinite(self.radius) or self.radius <= 0):
            raise ValueError("subdivisions must be >= 0 and radius positive")
        if self.surface == "torus":
            if not (isfinite(self.major_radius) and 0 < self.minor_radius < self.major_radius):
                raise ValueError("the torus needs major_radius > minor_radius > 0")
            if not 0 < self.target_edge < self.minor_radius:
                raise ValueError("target_edge must be positive and smaller than minor_radius")
        for name in _RANGE_FIELDS:
            low, high = getattr(self, name)
            if not (isfinite(low) and isfinite(high) and 0 <= low < high):
                raise ValueError(f"{name} must satisfy 0 <= low < high")
        if self.sigma_range[0] <= 0:
            raise ValueError("sigma must be positive")
        if self.ood_k0_range[0] < self.k0_range[1]:
            raise ValueError("ood_k0_range must start at or above the training k0 range")
        if not (isfinite(self.final_time) and isfinite(self.dt)) or self.final_time <= 0 or self.dt <= 0:
            raise ValueError("final_time and dt must be finite and positive")
        n = self.final_time / self.dt
        if abs(n - round(n)) > 1e-9 * max(1.0, n) or round(n) < 1:
            raise ValueError("final_time must be a whole number of dt steps")
        if self.output_steps < 2 or (self.steps - 1) % (self.output_steps - 1):
            raise ValueError("output_steps - 1 must divide the number of time steps")
        if self.k_eig < 1 or self.min_points_per_wavelength <= 0:
            raise ValueError("k_eig and min_points_per_wavelength must be positive")

    @property
    def steps(self) -> int:
        return int(round(self.final_time / self.dt)) + 1

    def output_indices(self) -> np.ndarray:
        return np.linspace(0, self.steps - 1, self.output_steps).round().astype(int)

    def output_times(self) -> np.ndarray:
        return self.output_indices() * self.dt

    def physics_dict(self) -> dict:
        """Fields stored in the manifest: the shared ones plus only this surface's geometry.

        For the sphere this is exactly the stage-1 manifest (no ``surface`` key), so a
        regenerated sphere dataset keeps the same identity.
        """
        others = {f for s, fields in _GEOMETRY_FIELDS.items() if s != self.surface for f in fields}
        keep = set(_GEOMETRY_FIELDS[self.surface])
        return {k: v for k, v in asdict(self).items() if k in keep or k not in others}


SphereConfig = SurfaceConfig    # stage-1 name, kept for existing code and manifests


def torus_config(**overrides) -> SurfaceConfig:
    """Default stage-2 settings: torus R = 1, r = 0.5, edge 0.03 (25 346 vertices).

    Chosen from a study against the exact torus solution (``reference.torus_exact_solution``):

    - σ ≤ 0.25 starts the packet well inside the tube (circumference 2πr = π); the
      chordal Gaussian puts at most ~1e-3 of its norm across the tube. By t = 0.5 every
      packet has spread over a large part of the torus, as it should.
    - Resolution, ≥ 10 points per wavelength: k0 + 3/(2σ) ≤ 20.55 on this mesh. The
      training rectangle (k0 ≤ 6, σ ≥ 0.15) needs 16.0 and the OOD rectangle (k0 ≤ 8)
      needs 18.0, so both are sampled whole.
    - Data error, Crank–Nicolson with dt = 0.001 and the blended mass against the
      continuous torus, maximum over the saved times, over 144 packets per corner (12
      places around the tube, bottom included, × 12 directions): up to 1.2 % at the
      training corner (k0 = 6, σ = 0.15; median 1.0 %) and up to 2.6 % at the OOD
      corner (k0 = 8, σ = 0.15; median 1.65 %; ≤ 1.5 % for σ ≥ 0.18). The worst packets
      start on the outer equator and move along u. Away from the corners it is much
      smaller: over the 1100 packets of the generated dataset, median 0.31 % and maximum
      1.04 % in the training rectangle, maximum 1.80 % in the OOD one.
    - That error has two parts of similar size at the corners. The blended eigenvalues
      are ~0.1 % too high in the resolved band (second order in the edge), a phase lead;
      CN's phase lag (relative, ∝ dt²λ²) is about as large at dt = 0.001 and partly
      cancels it, by an amount that depends on where the packet starts and moves. Over
      8 packets per corner (outer equator, top, inner equator, bottom; along u or v) the
      worst errors are 1.2 % / 2.5 % at the training / OOD corner with dt = 0.001,
      against 1.7 % / 2.7 % with dt = 0.0005 and 1.9 % / 3.1 % for dt → 0 (the mesh
      error alone), so dt = 0.001 gives the smallest worst case. For dt → 0 the worst
      packets start at the bottom of the tube and move along u; for dt = 0.001 they
      start on the outer equator and move along u, where the mesh error is small and
      leaves CN's lag uncancelled. Packets moving along v at the OOD corner would be more
      accurate with dt = 0.0005 (at most 1.6 % instead of 2.2 %). The edge 0.038 mesh
      (15 704 vertices) gave up to 2.4 % at the training corner, which is why it was
      refined.
    - k_eig = 800: the torus has 1.6× the sphere's area, so it needs more modes for the
      same wavenumbers. 800 lumped eigenfunctions keep ≥ 1 - 1e-5 of the norm of every
      training packet and ≥ 1 - 6e-5 at the OOD corner (400 would lose up to 0.4 % and
      1.7 %).
    - Cost: GLNO training needs ~3.3 GiB of GPU memory per sample (~13 GiB with batches
      of 4: an L4, A100 or G4 on Colab; a T4 is too tight).
    """
    if overrides.get("surface", "torus") != "torus":
        raise ValueError("torus_config builds torus settings; use SurfaceConfig for the sphere")
    sphere_only = set(_GEOMETRY_FIELDS["sphere"]) & set(overrides)
    if sphere_only:
        raise ValueError(f"{sorted(sphere_only)} are sphere settings, not torus ones")
    fields = {"surface": "torus", "k0_range": (1.0, 6.0), "sigma_range": (0.15, 0.25),
              "ood_k0_range": (6.0, 8.0), "k_eig": 800}
    fields.update(overrides)
    return SurfaceConfig(**fields)
