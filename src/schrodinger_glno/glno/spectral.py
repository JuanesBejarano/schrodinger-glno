"""What the GLNO sees of one mesh in the stage-3 dataset, computed when the data are loaded.

The stored eigenpairs ``(Φ, λ)`` solve ``L φ = λ M_l φ`` with the lumped mass ``M_l``,
but Crank–Nicolson made the data with the blended mass ``M_b = (M_l + M_c)/2``. Evolving
in ``(Φ, λ)`` misses the data by 1.4–27 % at t = 0.5. ``ritz_polar_basis`` fixes this
inside the stored 700 modes, without a new eigensolve:

1. Rayleigh–Ritz with the blended mass: ``A y = μ B y`` with ``A = Φᵀ L Φ`` and
   ``B = Φᵀ M_b Φ``, so ``yᵀ B y = I`` and ``μ`` are the frequencies CN actually used.
2. The orthogonal polar factor ``U = y (yᵀy)^(-1/2)``, which equals ``B^(1/2) y``.
   ``E = Φ U`` stays orthonormal in the lumped mass, ``Eᵀ diag(m) E = UᵀU = I``, so the
   network's diagonal-mass ``to_basis``/``from_basis`` stay exact projections, and each
   column ``k`` of ``E`` carries ``μ_k``. Evolving in ``(E, μ)`` is within 0.28 % (median)
   and 1.35 % (max) of the data, which is the data's own time-step error.

Because ``(5/8) M_l ⪯ M_b ⪯ M_l`` element by element and ``Φᵀ M_l Φ = I``, the
eigenvalues of ``B`` lie in ``[5/8, 1]`` and the singular values of ``y`` in
``[1, √(8/5)]`` (measured on the 200 shapes: ``[1 - 5e-8, 1.055]``).

The signs of the eigenvectors that LAPACK returns may change with the driver or the
number of BLAS threads, so the column signs of ``U`` are fixed (largest entry positive).
On the dataset the largest entry beats the second by at least 1.8e-5, and ``U`` agrees
across drivers and thread counts to 4e-10, or to ~1e-6 on round tori, whose nearly
equal eigenvalue pairs are anyway rotated at random in training (``augment.py``).

NumPy and SciPy only, float64 inside. See Plan_GLNO_etapa3.md, sections 1.1–1.5.
"""
from typing import NamedTuple
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from numpy.typing import NDArray
from ..mesh import blended_mass, cotan_laplacian, vertex_areas

RITZ_VERSION = 1
SINGULAR_VALUE_BOUNDS = (1.0, float(np.sqrt(8 / 5)))


class RitzBasis(NamedTuple):
    mu: NDArray                  # (K,) float64, ascending, >= 0: blended-mass Ritz values
    U: NDArray                   # (K, K) float64, orthogonal, canonical column signs
    E: NDArray                   # (N, K) float32, Φ U
    singular_values: NDArray     # (K,) float64, of y, descending


def canonical_signs(U: NDArray) -> NDArray:
    """Flip each column so that its entry of largest magnitude is positive."""
    rows = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[rows, np.arange(U.shape[1])])
    signs[signs == 0] = 1
    return U * signs


def ritz_polar_basis(verts: NDArray, faces: NDArray, evecs: NDArray, *,
                     L: sp.spmatrix | None = None, blended: sp.spmatrix | None = None,
                     driver: str | None = None) -> RitzBasis:
    """Blended-mass Ritz values and the lumped-orthonormal polar basis inside ``span(evecs)``.

    ``evecs`` (N, K) must be orthonormal in the lumped mass ``vertex_areas(verts, faces)``,
    as stored in each shape's ``mesh.npz``. ``L`` and ``blended`` default to the operators
    the Crank–Nicolson solver used (``cotan_laplacian`` and ``blended_mass``). ``driver``
    is passed to ``scipy.linalg.eigh`` (tests only).
    """
    phi = np.asarray(evecs, dtype=np.float64)
    n, k = phi.shape
    if len(verts) != n or not 1 <= k < n:
        raise ValueError("evecs must be (number of vertices, K) with 1 <= K < number of vertices")
    L = cotan_laplacian(verts, faces) if L is None else L
    blended = blended_mass(verts, faces) if blended is None else blended
    lumped = vertex_areas(verts, faces)
    gram = phi.T @ (lumped[:, None] * phi)
    # 1e-6 keeps this check ahead of the singular-value one (s_min ≈ 1 - error/2); float32 storage gives 1e-7
    if not np.allclose(gram, np.eye(k), rtol=0, atol=1e-6):
        raise ValueError(f"evecs are not orthonormal in the lumped mass (max error "
                         f"{np.abs(gram - np.eye(k)).max():.1e})")
    A = phi.T @ (L @ phi)
    B = phi.T @ (blended @ phi)
    A, B = (A + A.T) / 2, (B + B.T) / 2
    mu, y = scipy.linalg.eigh(A, B, driver=driver)
    W, s, Vt = np.linalg.svd(y)
    U = canonical_signs(W @ Vt)
    if np.abs(U.T @ U - np.eye(k)).max() > 1e-10:
        raise ValueError("the polar factor is not orthogonal")
    low, high = SINGULAR_VALUE_BOUNDS
    if s.min() < low - 1e-6 or s.max() > high + 1e-6:
        raise ValueError(f"singular values of y outside [1, √(8/5)]: [{s.min():.6f}, {s.max():.6f}]")
    if np.any(np.diff(mu) < 0):
        raise ValueError("Ritz values are not ascending")
    return RitzBasis(np.maximum(mu, 0.0), U, (phi @ U).astype(np.float32), s)


def eigen_clusters(mu: NDArray, rel_gap: float) -> NDArray:
    """Label groups of consecutive modes whose relative gap is below ``rel_gap``.

    The gap between modes k and k+1 is ``(μ_{k+1} - μ_k) / ((μ_{k+1} + μ_k)/2)``; a new
    group starts where it is ``>= rel_gap``. Mode 0 (the constant) is always alone. Each
    mode gets the index of the first mode of its group, so labels are non-decreasing and
    ``labels[k] == k`` marks the start of a group.

    Groups chain: a group can span more than ``rel_gap`` from its first to its last mode
    (on the dataset, up to ~3× at 1e-4).
    """
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 1 or len(mu) < 1 or not np.isfinite(mu).all() or np.any(np.diff(mu) < 0):
        raise ValueError("mu must be a non-empty, finite, ascending vector")
    if not rel_gap >= 0:
        raise ValueError("rel_gap must be >= 0")
    mean = (mu[1:] + mu[:-1]) / 2
    gap = np.divide(np.diff(mu), mean, out=np.zeros(len(mu) - 1), where=mean > 0)
    starts = np.concatenate([[True], gap >= rel_gap])
    if len(mu) > 1:
        starts[1] = True
    first = np.flatnonzero(starts)
    return first[np.cumsum(starts) - 1].astype(np.int64)


def centroid(verts: NDArray, mass: NDArray) -> NDArray:
    """Area-weighted centroid ``Σ m x / Σ m``."""
    mass = np.asarray(mass, dtype=np.float64)
    return (mass[:, None] * verts).sum(axis=0) / mass.sum()


def geo_feature(verts: NDArray, mass: NDArray, scale: float) -> NDArray:
    """``‖x - c‖ / scale`` per vertex (float32), ``c`` the area centroid; invariant to rigid motions."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return (np.linalg.norm(verts - centroid(verts, mass), axis=1) / scale).astype(np.float32)


def multiplier(mu: NDArray, steps: NDArray, dt: float, scheme: str) -> NDArray:
    """(K, T) factor applied to each mode after ``steps`` time steps of size ``dt``.

    ``exact``: ``e^{-iμt/2}`` with ``t = steps·dt``, the solution of ``i M ψ' = (L/2) ψ``.
    ``cn``: Crank–Nicolson, ``r(μ)^n`` with ``r = (1 - iμΔt/4)/(1 + iμΔt/4)``, written as
    ``e^{-2in·atan(μΔt/4)}`` (``|r| = 1``), from ``(M + iΔt A/2) ψ^{n+1} = (M - iΔt A/2) ψ^n``
    with ``A = L/2``.
    """
    mu = np.asarray(mu, dtype=np.float64)
    steps = np.asarray(steps)
    if not np.issubdtype(steps.dtype, np.integer) or np.any(steps < 0):
        raise ValueError("steps must be non-negative integers")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if scheme == "exact":
        phase = 0.5 * np.outer(mu, steps * dt)
    elif scheme == "cn":
        phase = 2 * np.outer(np.arctan(mu * dt / 4), steps)
    else:
        raise ValueError("scheme must be 'exact' or 'cn'")
    return np.exp(-1j * phase)


def propagate(E: NDArray, mu: NDArray, mass: NDArray, psi0: NDArray, steps: NDArray, dt: float,
              scheme: str = "cn") -> NDArray:
    """(N, T) evolution of ``psi0`` inside ``span(E)``: ``E (f ⊙ Eᵀ(m ψ0))``, ``f`` from ``multiplier``.

    ``E`` must be orthonormal in the lumped mass ``mass``.
    """
    E = np.asarray(E, dtype=np.float64)
    coeffs = E.T @ (np.asarray(mass, dtype=np.float64) * psi0)
    return E @ (coeffs[:, None] * multiplier(mu, steps, dt, scheme))


def mass_relative_error(pred: NDArray, target: NDArray, mass: NDArray) -> NDArray:
    """``‖pred - target‖_M / ‖target‖_M`` per column of (N, T) arrays, with the lumped mass."""
    mass = np.asarray(mass, dtype=np.float64)[:, None]
    return np.sqrt(np.sum(mass * np.abs(pred - target) ** 2, axis=0) / np.sum(mass * np.abs(target) ** 2, axis=0))


def spectral_oracle_error(E: NDArray, mu: NDArray, mass: NDArray, psi0: NDArray, outputs: NDArray,
                          steps: NDArray, dt: float) -> NDArray:
    """Error of the Crank–Nicolson multiplier applied in ``(E, μ)`` against the stored outputs.

    The labels are CN states, and in this basis CN is one factor per mode that the
    network could learn exactly, so this is the error of the best answer of that form
    (the "achievable error"). It is not a strict lower bound: the ``e^{-σg}`` weighting
    and the per-vertex layers can leave ``span(E)``. ``outputs`` is (N, T) complex at
    ``steps``; returns (T,).
    """
    return mass_relative_error(propagate(E, mu, mass, psi0, steps, dt, "cn"), outputs, mass)
