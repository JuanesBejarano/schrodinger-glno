"""Crank–Nicolson time stepping for factored surface operators."""
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as spa
from scipy.sparse.linalg import SuperLU

def solve_CN(psi0: NDArray, LU: SuperLU, U1: spa.spmatrix, N_steps: int, t_out_idx: NDArray) -> NDArray:
    """Save requested states, numbered 0 through N_steps-1."""
    idx = np.asarray(t_out_idx)
    if N_steps < 1 or idx.ndim != 1 or idx.size < 1 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("steps and output indices must be nonempty integers")
    if idx[0] < 0 or idx[-1] >= N_steps or np.any(np.diff(idx) <= 0):
        raise ValueError("output indices must be unique, increasing and within the simulated range")
    if np.asarray(psi0).shape != (U1.shape[0],):
        raise ValueError("psi0 length must match the CN matrices")
    psi = psi0.copy()
    snapshots = np.zeros((len(psi0), len(idx)), dtype=complex)
    out_count = 0
    for n in range(N_steps):
        if out_count < len(idx) and n == idx[out_count]:
            snapshots[:, out_count] = psi
            out_count += 1
        if n < N_steps - 1:
            psi = LU.solve(U1.dot(psi))
    return snapshots

def solve_CN_full(psi0: NDArray, LU: SuperLU, U1: spa.spmatrix, N_steps: int) -> NDArray:
    """Return every state, including the initial state."""
    return solve_CN(psi0, LU, U1, N_steps, np.arange(N_steps))
