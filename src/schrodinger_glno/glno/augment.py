"""Random gauge of the eigenbasis while training: sign flips and rotations inside eigenvalue groups.

The eigenvectors of a mesh are fixed only up to sign, and inside a group of nearly equal
eigenvalues only up to a rotation: the solver's choice is arbitrary. The periodic branch
of the GLNO layer, ``Φ diag(H(λ)) Φᵀ M``, does not see that choice (exactly for signs,
up to ``H'·gap`` for rotations), but the aperiodic branch (``layers.py``,
``output_residue2``) does. Training with a random gauge teaches the network not to depend
on it.

For every sample the basis ``E`` becomes ``E Q``, with ``Q`` block diagonal: one Haar-random
orthogonal block per group from ``spectral.eigen_clusters`` (a group of one mode gets
``±1``, i.e. a sign flip). Eigenvalues, mass, geo feature, input and labels are unchanged.
Inside a group the eigenvalues differ by less than the threshold τ, so the rotated basis
gives each mixed mode a slightly wrong frequency; with τ = 1e-4 that adds up to ~0.08 % to
the labels' mismatch at t = 0.5 over the stage-3 dataset (Plan_GLNO_etapa3.md §1.3), and
0.012-0.016 % on the step-4 gate meshes (000_sphere, 003_torus).

The random numbers come from a CPU generator seeded by ``(seed, epoch, step)``, so a
resumed run and another GPU model draw the same gauges.
"""
import numpy as np
import torch

__all__ = ["gauge_generator", "group_starts", "block_haar", "tf32_matmul_enabled", "augment_batch"]


def gauge_generator(seed: int, epoch: int, step: int) -> torch.Generator:
    """CPU generator for the gauge of one training step."""
    state = np.random.SeedSequence([seed, epoch, step]).generate_state(1, dtype=np.uint64)[0]
    return torch.Generator(device="cpu").manual_seed(int(state))


def group_starts(labels: np.ndarray) -> np.ndarray:
    """First mode of each group, from labels made by ``spectral.eigen_clusters``; checks they are contiguous."""
    labels = np.asarray(labels)
    k = np.arange(len(labels))
    if labels.ndim != 1 or len(labels) == 0 or not np.issubdtype(labels.dtype, np.integer) or labels[0] != 0 \
            or np.any(np.diff(labels) < 0) or np.any(labels > k) or np.any(labels[labels] != labels):
        raise ValueError("group labels must be non-decreasing first-mode indices, as eigen_clusters returns")
    return np.flatnonzero(labels == k)


def block_haar(clusters: torch.Tensor | np.ndarray, generator: torch.Generator, *, device=None,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(B, K, K) block-diagonal orthogonal matrices, one Haar-random block per eigenvalue group.

    ``clusters`` is (B, K), the labels of ``eigen_clusters`` for each sample (keep them on
    the CPU: reading GPU labels forces a synchronisation). Entries off the blocks are
    exactly 0; a group of one mode gets ``±1``. The Gaussian entries are drawn in a fixed
    order (sample, then group, row by row). All blocks of one size go through one batched
    QR with the sign correction of Mezzadri (2007), which makes each block exactly Haar on
    O(s). Everything is built on the CPU and only the finished ``Q`` is moved: thousands of
    tiny QRs on a GPU cost ~90 ms per step against ~4 ms here, and the result does not
    depend on the device.
    """
    if torch.is_tensor(clusters):
        labels = clusters.detach().cpu().numpy()
        device = clusters.device if device is None else device
    else:
        labels = np.asarray(clusters)
    if labels.ndim != 2:
        raise ValueError("clusters must be (B, K)")
    batch, k = labels.shape
    starts = [group_starts(row) for row in labels]
    sample = np.concatenate([np.full(len(s), b) for b, s in enumerate(starts)])
    start = np.concatenate(starts)
    size = np.concatenate([np.diff(np.append(s, k)) for s in starts])
    squares = size * size
    offsets = np.cumsum(squares) - squares
    draws = torch.randn(int(squares.sum()), generator=generator, dtype=torch.float64)
    index, values = [], []                     # only the block entries go to the device, not the dense Q
    for n in np.unique(size):
        chosen = size == n
        gauss = draws[torch.from_numpy(offsets[chosen][:, None] + np.arange(n * n))].reshape(-1, n, n)
        q, r = torch.linalg.qr(gauss)
        signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        signs[signs == 0] = 1
        values.append((q * signs.unsqueeze(-2)).reshape(-1))
        i = np.arange(n)
        rows = start[chosen][:, None, None] + i[:, None]
        cols = start[chosen][:, None, None] + i
        index.append(np.stack(np.broadcast_arrays(sample[chosen][:, None, None], rows, cols)).reshape(3, -1))
    index = torch.from_numpy(np.concatenate(index, axis=1))
    values = torch.cat(values).to(dtype)
    Q = torch.zeros(batch, k, k, dtype=dtype, device=device)
    if device is not None:
        index, values = index.to(device), values.to(device)
    Q[index[0], index[1], index[2]] = values
    return Q


def tf32_matmul_enabled() -> bool:
    """Whether float32 CUDA matmuls run in TF32, with the new (``fp32_precision``) or legacy API.

    In PyTorch >= 2.9, reading the legacy ``allow_tf32`` after the new API was used raises.
    """
    precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if precision is not None:
        return precision == "tf32"
    return torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest"


def augment_batch(batch: dict, generator: torch.Generator) -> dict:
    """A new batch whose ``evecs`` are ``evecs @ Q``; nothing in ``batch`` is modified.

    ``batch`` needs ``evecs`` (B, N, K) and ``clusters`` (B, K). Rows of zero padding stay
    exactly zero. On the GPU, TF32 matmuls must be off: they would round the rotated basis
    to ~1e-3 and break its orthonormality.
    """
    evecs = batch["evecs"]
    if evecs.is_cuda and tf32_matmul_enabled():
        raise RuntimeError("TF32 matmuls are on; turn them off with "
                           "torch.backends.cuda.matmul.fp32_precision = 'ieee' (PyTorch >= 2.9) or "
                           "torch.backends.cuda.matmul.allow_tf32 = False")
    if tuple(batch["clusters"].shape) != (evecs.shape[0], evecs.shape[2]):
        raise ValueError("clusters must be (B, K) for evecs (B, N, K)")
    Q = block_haar(batch["clusters"], generator, device=evecs.device, dtype=evecs.dtype)
    return {**batch, "evecs": torch.bmm(evecs, Q)}
