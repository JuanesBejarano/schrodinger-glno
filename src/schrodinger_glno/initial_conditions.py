"""Gaussian wave packets discretized on a triangle mesh.

The continuous packet is evaluated at each vertex (one complex value per vertex).
That is the P1 finite-element representation Crank–Nicolson uses, and the same
per-vertex format the GLNO datasets store as ``input``. The vector is then
normalized with the mass matrix used by the solver: ``ψᴴ M ψ`` approximates
``∫ |ψ|² dμ`` and is the quantity Crank–Nicolson conserves exactly. The plain sum
``Σ |ψ_i|²`` weights every vertex by 1 instead of by its area: normalizing with it
would leave ``∫ |ψ|² dμ`` close to the mean vertex area rather than 1, a factor
that also depends on the local vertex density on non-uniform meshes.

``mass`` may be one area per vertex (lumped) or a sparse matrix such as
``mesh.blended_mass``.
"""
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from .mesh import as_mass_matrix, mean_edge_length


def mass_norm(psi: NDArray, mass: NDArray | sp.spmatrix) -> float:
    """Discrete L² norm on the surface: ``sqrt(ψᴴ M ψ)`` (``sqrt(Σ m_i |ψ_i|²)`` if lumped)."""
    if sp.issparse(mass):
        return float(np.sqrt(np.real(np.vdot(psi, mass @ psi))))
    return float(np.sqrt(np.sum(mass * np.abs(psi)**2)))


def tangent_direction(normal: NDArray, direction: NDArray) -> NDArray:
    """Unit vector of ``direction`` projected onto the tangent plane with this ``normal``."""
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    direction = np.asarray(direction, float)
    tangent = direction - (direction @ normal) * normal
    size = np.linalg.norm(tangent)
    if size < 1e-8 * max(np.linalg.norm(direction), 1e-300):
        raise ValueError("direction is (almost) normal to the surface; choose another one")
    return tangent / size


def gaussian_packet(verts: NDArray, mass: NDArray | sp.spmatrix, normals: NDArray, center: int,
                    k0: float, sigma: float, direction: NDArray) -> NDArray:
    """Gaussian packet centred at vertex ``center``, moving along ``direction``.

    ``ψ(x) ∝ exp(-|x - x0|² / (4σ²)) · exp(i k0 t0·(x - x0))``, where ``t0`` is
    ``direction`` projected onto the tangent plane at ``x0``. As in the 1D
    ``packet="paper"`` convention, σ is the standard deviation of ``|ψ|²`` and
    the momentum spread is ``1 / (2σ)``. Distances are chordal (straight lines in
    R³): smooth everywhere and accurate while σ is small compared with the
    curvature radius. Returns the vertex values normalized with ``mass``.
    """
    if not 0 <= center < len(verts):
        raise ValueError("center must be a vertex index")
    if sigma <= 0 or not np.isfinite(k0):
        raise ValueError("sigma must be positive and k0 finite")
    M = as_mass_matrix(mass)
    if M.shape != (len(verts), len(verts)) or np.any(M.diagonal() <= 0):
        raise ValueError("mass must hold one positive area per vertex or be a (V, V) mass matrix")
    offset = verts - verts[center]
    t0 = tangent_direction(normals[center], direction)
    psi = np.exp(-np.sum(offset**2, axis=1) / (4 * sigma**2)) * np.exp(1j * k0 * (offset @ t0))
    return psi / mass_norm(psi, mass)


def max_wavenumber(k0: float, sigma: float, n_std: float = 3.0) -> float:
    """Largest wavenumber with appreciable weight: ``|k0| + n_std / (2σ)``."""
    return abs(k0) + n_std / (2 * sigma)


def points_per_wavelength(verts: NDArray, faces: NDArray, k0: float, sigma: float,
                          n_std: float = 3.0) -> float:
    """Mean mesh edges per shortest wavelength of the packet; aim for about 10 or more.

    This only says whether the mesh represents ψ0 well. How well the phase is kept
    while the packet travels depends on the mass matrix (see ``mesh.blended_mass``).
    """
    return 2 * np.pi / (max_wavenumber(k0, sigma, n_std) * mean_edge_length(verts, faces))


def spectral_fraction(psi: NDArray, mass: NDArray | sp.spmatrix, evecs: NDArray) -> float:
    """Fraction of ``‖ψ‖²`` captured by the eigenvectors ``evecs``, M-orthonormal for this ``mass``."""
    coeffs = evecs.T @ (as_mass_matrix(mass) @ psi)
    return float(np.sum(np.abs(coeffs)**2) / mass_norm(psi, mass)**2)
