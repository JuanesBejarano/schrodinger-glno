"""Random Gaussian-packet parameters on a mesh.

Each packet is ``(center, phi, k0, sigma)``: a vertex chosen with probability
proportional to its area (uniform over the surface, not over vertices), an
angle ``phi`` for the direction in the tangent plane, and ``(k0, sigma)`` drawn
uniformly and kept only if the mesh resolves the packet.
"""
import numpy as np
from numpy.typing import NDArray
from .initial_conditions import max_wavenumber, tangent_direction

PARAM_COLUMNS = ("center", "phi", "k0", "sigma")


def tangent_basis(normal: NDArray) -> tuple[NDArray, NDArray]:
    """Orthonormal ``(e1, e2)`` spanning the plane orthogonal to ``normal``."""
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]  # the axis least aligned with the normal
    e1 = tangent_direction(normal, axis)
    return e1, np.cross(normal, e1)


def direction_from_angle(normal: NDArray, phi: float) -> NDArray:
    """Unit tangent vector ``cos(phi) e1 + sin(phi) e2`` at a point with this ``normal``."""
    e1, e2 = tangent_basis(normal)
    return np.cos(phi) * e1 + np.sin(phi) * e2


def allowed_wavenumber(mean_edge: float, min_points_per_wavelength: float) -> float:
    """Largest wavenumber with at least ``min_points_per_wavelength`` edges per wavelength."""
    return 2 * np.pi / (min_points_per_wavelength * mean_edge)


def resolved_fraction(k0_range: tuple[float, float], sigma_range: tuple[float, float], k_allowed: float,
                      n_std: float = 3.0) -> float:
    """Exact area fraction of ``k0_range x sigma_range`` where ``k0 + n_std / (2 sigma) <= k_allowed``.

    It is the probability that ``sample_packets`` accepts a uniform draw; the
    empirical acceptance of a few dozen draws is a noisy estimate of it.
    """
    (k_lo, k_hi), (s_lo, s_hi) = k0_range, sigma_range
    c = n_std / 2

    def integral(a: float, b: float) -> float:          # ∫ (k_allowed - k_lo - c/σ) dσ over [a, b]
        return (k_allowed - k_lo) * (b - a) - c * np.log(b / a) if b > a else 0.0

    # For each σ the accepted k0 run from k_lo to min(k_hi, k_allowed - c/σ): none below s1, all above s2.
    s1 = c / (k_allowed - k_lo) if k_allowed > k_lo else np.inf
    s2 = c / (k_allowed - k_hi) if k_allowed > k_hi else np.inf
    a, b = min(max(s1, s_lo), s_hi), min(max(s2, s_lo), s_hi)
    area = integral(a, b) + (k_hi - k_lo) * (s_hi - b)
    return float(area / ((k_hi - k_lo) * (s_hi - s_lo)))


def sample_packets(n: int, mass: NDArray, rng: np.random.Generator, k0_range: tuple[float, float],
                   sigma_range: tuple[float, float], k_allowed: float, stats: dict | None = None) -> NDArray:
    """Return ``(n, 4)`` rows ``(center, phi, k0, sigma)`` that the mesh resolves.

    ``(k0, sigma)`` is uniform on the part of ``k0_range x sigma_range`` where
    ``k0 + 3 / (2 sigma) <= k_allowed``. If ``stats`` is given, it receives the
    number of ``(k0, sigma)`` pairs ``tried`` and ``accepted``: an acceptance below 1
    means part of the requested rectangle was left out.
    """
    if n < 0 or np.any(mass <= 0):
        raise ValueError("n must be >= 0 and mass positive")
    if max_wavenumber(k0_range[0], sigma_range[1]) > k_allowed:
        raise ValueError("no (k0, sigma) in these ranges is resolved by the mesh")
    centers = rng.choice(len(mass), size=n, p=mass / mass.sum())
    phi = rng.uniform(0, 2 * np.pi, size=n)
    k0, sigma = np.empty(n), np.empty(n)
    filled, rounds, tried, accepted = 0, 0, 0, 0
    while filled < n:
        rounds += 1
        if rounds > 1000:
            raise ValueError("almost no (k0, sigma) in these ranges is resolved by the mesh")
        k_try = rng.uniform(*k0_range, size=2 * (n - filled) + 8)
        s_try = rng.uniform(*sigma_range, size=k_try.size)
        resolved = k_try + 3 / (2 * s_try) <= k_allowed
        tried += k_try.size
        accepted += int(resolved.sum())
        ok = np.flatnonzero(resolved)[: n - filled]
        k0[filled:filled + ok.size], sigma[filled:filled + ok.size] = k_try[ok], s_try[ok]
        filled += ok.size
    if stats is not None:
        stats.update(tried=tried, accepted=accepted)
    return np.column_stack([centers, phi, k0, sigma])
