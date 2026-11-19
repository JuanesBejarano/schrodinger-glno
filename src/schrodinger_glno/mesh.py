"""Triangle-mesh operators with the same definitions the GLNO repository uses.

``cotan_laplacian`` and ``vertex_areas`` reproduce ``potpourri3d.cotan_laplacian``
and ``potpourri3d.vertex_areas`` (used in ``GLNO-repo/utils/geometry/operator.py``),
so the solver and the network see the same discretization.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from numpy.typing import NDArray


def _check_mesh(verts: NDArray, faces: NDArray) -> None:
    if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("verts must be (V, 3) and faces (F, 3)")
    if faces.min() < 0 or faces.max() >= len(verts):
        raise ValueError("faces reference vertices that do not exist")


def icosphere(subdivisions: int, radius: float = 1.0) -> tuple[NDArray, NDArray]:
    """Icosahedron refined ``subdivisions`` times and projected onto the sphere.

    It has ``10 * 4**subdivisions + 2`` vertices and outward-oriented faces.
    """
    if subdivisions < 0 or radius <= 0:
        raise ValueError("subdivisions must be >= 0 and radius > 0")
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
                      (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    faces = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
                      (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
                      (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    for _ in range(subdivisions):
        # Each face edge (01, 12, 20) gets one shared midpoint; each face splits into four.
        edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        mid = verts[unique].mean(axis=1)
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        ab, bc, ca = (len(verts) + inverse.reshape(-1, 3)).T
        a, b, c = faces.T
        verts = np.vstack([verts, mid])
        faces = np.concatenate([np.stack(f, axis=1) for f in
                                ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))])
    return radius * verts, faces


def cotan_laplacian(verts: NDArray, faces: NDArray, denom_eps: float = 1e-10) -> sp.csr_matrix:
    """Cotangent Laplacian: symmetric, positive semidefinite, the weak form of ``-Δ``.

    Edge ``ij`` has weight ``(cot α + cot β) / 2``, where α and β are the angles
    opposite to it. Rows sum to zero.
    """
    _check_mesh(verts, faces)
    corners = verts[faces]
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        u = corners[:, i] - corners[:, k]
        w = corners[:, j] - corners[:, k]
        cot = np.einsum("fd,fd->f", u, w) / (np.linalg.norm(np.cross(u, w), axis=1) + denom_eps)
        half = 0.5 * cot
        fi, fj = faces[:, i], faces[:, j]
        rows += [fi, fj, fi, fj]
        cols += [fj, fi, fi, fj]
        vals += [-half, -half, half, half]
    n = len(verts)
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n)).tocsr()


def face_areas(verts: NDArray, faces: NDArray) -> NDArray:
    _check_mesh(verts, faces)
    corners = verts[faces]
    return 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)


def vertex_areas(verts: NDArray, faces: NDArray) -> NDArray:
    """Lumped mass: one third of the area of the triangles around each vertex."""
    areas = face_areas(verts, faces)
    mass = np.zeros(len(verts))
    for k in range(3):
        np.add.at(mass, faces[:, k], areas / 3)
    return mass


def consistent_mass(verts: NDArray, faces: NDArray) -> sp.csr_matrix:
    """Consistent P1 mass matrix: ``M_ij = ∫ φ_i φ_j dA``, exact for piecewise-linear functions.

    On each triangle of area ``a`` the local matrix is ``a/12 · [[2,1,1],[1,2,1],[1,1,2]]``.
    Its rows sum to the lumped masses.
    """
    areas = face_areas(verts, faces)
    rows, cols, vals = [], [], []
    for i in range(3):
        for j in range(3):
            rows.append(faces[:, i])
            cols.append(faces[:, j])
            vals.append(areas * (2 if i == j else 1) / 12)
    n = len(verts)
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n)).tocsr()


def blended_mass(verts: NDArray, faces: NDArray) -> sp.csr_matrix:
    """Average of the lumped and consistent mass matrices, used by the time stepper.

    With P1 elements the lumped mass gives Laplacian eigenvalues that are too low and
    the consistent mass gives values that are too high, by almost the same amount; the
    average cancels most of it (Marfurt 1984; Christon 1999). On the 10 242-vertex
    icosphere the error at ℓ = 15 drops from -2.1 % to +0.02 %, so wave packets keep
    the right phase while they travel. The cancellation depends on the triangles: on
    the ``torus`` mesh it is only partial: for λ in 150–250, -1.7 % (lumped) becomes
    +0.13 % at edge 0.038, and -1.1 % becomes +0.09 % at edge 0.03 (second order in
    the edge), so there packets gain a small phase lead.
    """
    return (0.5 * (sp.diags(vertex_areas(verts, faces)) + consistent_mass(verts, faces))).tocsr()


def as_mass_matrix(mass: NDArray | sp.spmatrix) -> sp.csr_matrix:
    """A mass given as one area per vertex (lumped) or as a sparse matrix, as a sparse matrix."""
    return sp.csr_matrix(mass) if sp.issparse(mass) else sp.diags(np.asarray(mass, float)).tocsr()


def vertex_normals(verts: NDArray, faces: NDArray) -> NDArray:
    """Unit normals averaged from the incident faces, weighted by area."""
    _check_mesh(verts, faces)
    corners = verts[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def laplacian_eigenpairs(L: sp.spmatrix, mass: NDArray, k: int) -> tuple[NDArray, NDArray]:
    """The ``k`` smallest solutions of ``L φ = λ M φ``, sorted, with ``Φᵀ M Φ = I``.

    Shift-invert around a small negative value avoids factoring the singular ``L``;
    tiny negative eigenvalues from round-off are clipped to zero, as in the GLNO repo.
    ARPACK starts from a seeded vector: with a random start, repeated eigenvalues
    (the sphere has 2ℓ+1 per level) come back in a different basis on every run.
    ``mass`` is one area per vertex (lumped) or a sparse mass matrix.
    """
    n = L.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < number of vertices")
    v0 = np.random.default_rng(0).standard_normal(n)
    evals, evecs = sla.eigsh(L.tocsc(), k=k, M=as_mass_matrix(mass).tocsc(), sigma=-1e-3, v0=v0)
    order = np.argsort(evals)
    return np.clip(evals[order], 0, None), evecs[:, order]


def mean_edge_length(verts: NDArray, faces: NDArray) -> float:
    _check_mesh(verts, faces)
    edges = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    return float(np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1).mean())


def _torus_points(u: NDArray, v: NDArray, major_radius: float, minor_radius: float) -> NDArray:
    rho = major_radius + minor_radius * np.cos(v)
    return np.stack([rho * np.cos(u), rho * np.sin(u), minor_radius * np.sin(v)], axis=-1)


def _zip_rings(start_a: int, n_a: int, offset_a: float, start_b: int, n_b: int,
               offset_b: float) -> tuple[NDArray, NDArray, NDArray]:
    """Outward-oriented triangle strip between ring a and the next ring b (one step up in v).

    Point k of a ring sits at angle ``offset + 2πk/n``; edge k joins points k and k+1. Both
    rings are walked in angle order and each step closes the triangle on the ring edge whose
    midpoint comes first, which is the one with the shorter diagonal. Also returns, for every
    point of each ring, how many edges of the other ring it is the apex of (1 when staggered).
    """
    two_pi = 2 * np.pi
    mid_a = two_pi * (np.arange(n_a) + 0.5) / n_a
    mid_b = np.mod(offset_b - offset_a + two_pi * (np.arange(n_b) + 0.5) / n_b, two_pi)
    first_b = int(np.argmin(mid_b))
    on_a = np.argsort(np.concatenate([mid_a, np.roll(mid_b, -first_b)]), kind="stable") < n_a
    a = (np.cumsum(on_a) - on_a) % n_a                    # current point of each ring at each step
    b = (first_b + np.cumsum(~on_a) - ~on_a) % n_b
    faces = np.where(on_a[:, None],
                     np.stack([start_a + a, start_a + (a + 1) % n_a, start_b + b], axis=1),
                     np.stack([start_a + a, start_b + (b + 1) % n_b, start_b + b], axis=1))
    return faces, np.bincount(a[~on_a], minlength=n_a), np.bincount(b[on_a], minlength=n_b)


def _zip_ring_loop(counts: NDArray, starts: NDArray, first: int = 0) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Offsets and faces of a closed loop of rings, visited in the order first, first + 1, ...

    Each ring is staggered by half a step from the one before. Where the two counts differ, four
    sub-step shifts are tried and the one keeping the previous ring's valences closest to 6 is
    kept; the last ring also scores the strip that closes the loop. Returns the ring offsets, the
    faces, valence - 6 of every vertex and the visiting order.
    """
    n_v = len(counts)
    order = (first + np.arange(n_v)) % n_v
    offsets = np.zeros(n_v)

    def strip(a, b):
        return _zip_rings(starts[a], counts[a], offsets[a], starts[b], counts[b], offsets[b])

    # up[j] / down[j]: per point of ring j, the edges of the next / previous ring it is the apex of.
    # Its valence is 4 + up + down (6 when both are 1). down[first] is a guess until the last strip.
    faces, up, down = [], {}, {first: np.ones(counts[first], int)}
    for i in range(1, n_v):
        j, prev = order[i], order[i - 1]
        base = offsets[prev] + np.pi / counts[j]
        shifts = [0.0] if counts[j] == counts[prev] else 2 * np.pi * np.arange(4) / (4 * counts[j])
        best = None
        for shift in shifts:
            offsets[j] = base + shift
            new = [strip(prev, j)] + ([strip(j, first)] if i == n_v - 1 else [])
            cost = np.sum((new[0][1] + down[prev] - 2)**2)
            if i == n_v - 1:
                cost += np.sum((new[1][1] + new[0][2] - 2)**2) + np.sum((up[first] + new[1][2] - 2)**2)
            if best is None or cost < best[0]:
                best = (cost, shift, new)
        offsets[j] = base + best[1]
        for (strip_faces, extra_a, extra_b), (a, b) in zip(best[2], [(prev, j), (j, first)]):
            faces.append(strip_faces)
            up[a], down[b] = extra_a, extra_b
    excess = np.concatenate([up[j] + down[j] - 2 for j in range(n_v)])
    return offsets, np.concatenate(faces), excess, order


def _stagger_closing_strip(offsets: NDArray, counts: NDArray, order: NDArray) -> NDArray:
    """Ring offsets with the strip that closes the loop staggered like the other equal-count strips.

    Strips between rings of equal count get a half-step stagger, except the closing one, which
    gets whatever phase the others add up to (aligned rings would make a seam of right triangles).
    When its two counts are equal, its phase error is shared by all equal-count strips, as the
    same fraction of each one's step, by rotating the rings after each of them. Equal-count
    strips zip the same way at any phase in (0, 1), so the faces stay as they are. Each one ends
    at most ½·s_c/Σs of its step off the half step (s: the steps of all equal-count strips, s_c
    the closing one's), a quarter step when there are two of the same count.
    """
    first, last = order[0], order[-1]
    c = counts[order]
    equal_steps = np.concatenate([[0.0], np.where(c[1:] == c[:-1], 2 * np.pi / c[1:], 0.0)])
    if counts[last] != counts[first] or not equal_steps.any():
        return offsets
    step = 2 * np.pi / counts[first]
    error = np.mod(offsets[first] - offsets[last], step) - step / 2
    out = offsets.copy()
    out[order] += error / (step + equal_steps.sum()) * np.cumsum(equal_steps)
    return out


def torus(target_edge: float, major_radius: float = 1.0, minor_radius: float = 0.5) -> tuple[NDArray, NDArray]:
    """Torus of revolution with near-equilateral triangles of side ≈ ``target_edge``.

    The surface is ``((R + r cos v) cos u, (R + r cos v) sin u, r sin v)``. Vertices lie on
    rings of constant v, a geodesic distance ≈ target_edge·√3/2 apart; ring j has
    round(2π ρ_j / target_edge) points (ρ = R + r cos v), so triangles keep their size on the
    inner side instead of narrowing by (R - r)/(R + r) as on a (u, v) grid. Consecutive rings
    are staggered by half a step and zipped together; the strip that closes the loop gets the
    same stagger through a slight twist (``_stagger_closing_strip``). Where two counts differ,
    each extra point makes a 5-7 pair with two near-right triangles (cotan weight ≈ 0 on their
    diagonal), the best a ring mesh can do; ring offsets are picked among four sub-step shifts
    to keep two of them off the same point (valence 4 or 8); if that still happens where the
    loop closes (coarse meshes), the loop is redone starting from the inner side. With
    R - r ≥ 2·target_edge this gives valences 5-7, angles of 38-92.5° and vertex areas within a
    factor 1.5. Nearer the horn torus (a dozen triangles or fewer around the inner equator) the
    mesh is still a valid embedding but its quality degrades. Faces are outward-oriented as in
    ``icosphere``, and the result is deterministic.
    """
    R, r, h = float(major_radius), float(minor_radius), float(target_edge)
    if not (np.isfinite(R) and R > r > 0):
        raise ValueError("need major_radius > minor_radius > 0")
    if not 0 < h < r:
        raise ValueError("need 0 < target_edge < minor_radius")
    n_v = max(3, int(round(2 * np.pi * r / (h * np.sqrt(3) / 2))))
    v = 2 * np.pi * np.arange(n_v) / n_v
    counts = np.maximum(3, np.rint(2 * np.pi * (R + r * np.cos(v)) / h).astype(int))
    starts = np.concatenate([[0], np.cumsum(counts)])

    def score(excess):                   # vertices of valence 4 or 8 first, then Σ (valence - 6)²
        return int(np.sum(np.abs(excess) > 1)), int(np.sum(excess**2))

    best = _zip_ring_loop(counts, starts)
    for first in dict.fromkeys((n_v // 2, n_v // 4, 3 * n_v // 4)):      # inner side first
        if score(best[2])[0] == 0:
            break
        c = counts[(first + np.arange(n_v)) % n_v]
        if first == 0 or (c[-1] == c[0] and not np.any(c[1:] == c[:-1])):
            continue                     # the first pass, or a closing strip that could not be staggered
        new = _zip_ring_loop(counts, starts, first)
        if score(new[2]) < score(best[2]):
            best = new
    offsets, faces, _, order = best
    offsets = _stagger_closing_strip(offsets, counts, order)
    ring = np.repeat(np.arange(n_v), counts)
    u = offsets[ring] + 2 * np.pi * (np.arange(starts[-1]) - starts[ring]) / counts[ring]
    return _torus_points(u, v[ring], R, r), faces
