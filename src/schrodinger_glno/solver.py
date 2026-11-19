"""Crank–Nicolson on a triangle mesh.

The semi-discrete equation is ``i M dψ/dt = A ψ`` with ``A = ½ L + M diag(V)``.
Crank–Nicolson becomes ``(M + i dt/2 A) ψ^{n+1} = (M - i dt/2 A) ψ^n``, which is
the 1D scheme with the identity replaced by ``M``. Time stepping reuses
``schrodinger_fno.solver.solve_CN`` unchanged.

``mass`` may be one area per vertex (lumped, a diagonal ``M``) or a sparse matrix
such as ``mesh.blended_mass``. The scheme conserves ``ψᴴ M ψ`` for that ``M``.
"""
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import SuperLU, splu
from schrodinger_glno.time_stepping import solve_CN, solve_CN_full
from .mesh import as_mass_matrix

__all__ = ["hamiltonian_matrix", "build_cn_matrices", "energy", "exact_evolution", "spectral_evolution",
           "solve_CN", "solve_CN_full"]


def _checked_mass(L: sp.spmatrix, mass: NDArray | sp.spmatrix) -> sp.csr_matrix:
    n = L.shape[0]
    M = as_mass_matrix(mass)
    if L.shape != (n, n) or M.shape != (n, n):
        raise ValueError("L and the mass must be (V, V) or one value per vertex")
    if np.any(M.diagonal() <= 0):
        raise ValueError("mass must be positive")
    return M


def hamiltonian_matrix(L: sp.spmatrix, mass: NDArray | sp.spmatrix, V: NDArray) -> sp.csr_matrix:
    """``A = ½ L + M diag(V)``: real, symmetric for diagonal M, the weak form of ``-½Δ + V``."""
    M = _checked_mass(L, mass)
    if np.shape(V) != (L.shape[0],) or not np.isfinite(V).all():
        raise ValueError("V must hold one finite value per vertex")
    return (0.5 * L + M @ sp.diags(np.asarray(V, float))).tocsr()


def build_cn_matrices(L: sp.spmatrix, mass: NDArray | sp.spmatrix, V: NDArray, dt: float) -> tuple[SuperLU, sp.spmatrix]:
    """Factor ``M + i dt/2 A`` once and return it with ``M - i dt/2 A``."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    A = hamiltonian_matrix(L, mass, V)
    M = _checked_mass(L, mass)
    U2 = (M + (0.5j * dt) * A).tocsc()
    U1 = (M - (0.5j * dt) * A).tocsr()
    return splu(U2), U1


def energy(psi: NDArray, L: sp.spmatrix, mass: NDArray | sp.spmatrix, V: NDArray) -> float:
    """``⟨ψ, Hψ⟩ / ⟨ψ, ψ⟩`` in the mass inner product (``ψᴴ A ψ / ψᴴ M ψ``)."""
    A = hamiltonian_matrix(L, mass, V)
    M = _checked_mass(L, mass)
    return float(np.real(np.vdot(psi, A @ psi)) / np.real(np.vdot(psi, M @ psi)))


def exact_evolution(L: sp.spmatrix, mass: NDArray | sp.spmatrix, V: NDArray, psi0: NDArray,
                    times: NDArray) -> NDArray:
    """Exact solution of ``i M dψ/dt = A ψ`` at ``times`` (evenly spaced from 0): ``e^{-it M⁻¹A} ψ0``.

    No time-stepping error, so it isolates the error of Crank–Nicolson. For a
    diagonal mass ``M⁻¹A`` is sparse; otherwise ``M⁻¹`` is applied with a sparse LU,
    without forming the dense matrix. Returns (V, len(times)).
    """
    from scipy.sparse.linalg import LinearOperator, expm_multiply
    times = np.asarray(times, float)
    if times[0] != 0 or (len(times) > 2 and not np.allclose(np.diff(times), times[1] - times[0])):
        raise ValueError("times must start at 0 and be evenly spaced")
    A = hamiltonian_matrix(L, mass, V)
    if not sp.issparse(mass):
        generator = (-1j * sp.diags(1 / np.asarray(mass, float)) @ A).tocsr()
        trace = None
    else:
        lu = splu(_checked_mass(L, mass).tocsc())
        At = A.T.tocsr()
        solve = lambda b: lu.solve(np.ascontiguousarray(b.real)) + 1j * lu.solve(np.ascontiguousarray(b.imag))
        solve_t = lambda b: (lu.solve(np.ascontiguousarray(b.real), trans="T")
                             + 1j * lu.solve(np.ascontiguousarray(b.imag), trans="T"))
        n = A.shape[0]
        # B = -i M⁻¹A and its adjoint Bᴴ = i Aᵀ M⁻ᵀ (A and M are real); trace(B) only sets a shift.
        generator = LinearOperator((n, n), dtype=complex,
                                   matvec=lambda x: -1j * solve(A @ np.ravel(x)),
                                   rmatvec=lambda y: 1j * (At @ solve_t(np.ravel(y))))
        trace = 0.0
    if len(times) == 1:
        return psi0[:, None].astype(complex)
    return expm_multiply(generator, psi0.astype(complex), start=0, stop=times[-1], num=len(times),
                         endpoint=True, traceA=trace).T


def spectral_evolution(psi0: NDArray, mass: NDArray | sp.spmatrix, energies: NDArray, evecs: NDArray,
                       times: NDArray) -> NDArray:
    """Exact evolution inside the span of ``evecs``: ``Φ e^{-iEt} Φᵀ M ψ0``.

    ``evecs`` must be M-orthonormal eigenvectors of ``A φ = E M φ`` for the same ``M``.
    With V = 0 these are the Laplacian eigenpairs and ``E = λ / 2``. Returns (V, len(times)).
    """
    coeffs = evecs.T @ (as_mass_matrix(mass) @ psi0)
    phases = np.exp(-1j * np.outer(energies, np.asarray(times, float)))
    return evecs @ (coeffs[:, None] * phases)
