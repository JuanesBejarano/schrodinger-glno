"""Stage-3 shapes: deformed spheres and tori, made by moving the vertices of our meshes.

Both families start from a validated near-equilateral mesh (``mesh.icosphere``,
``mesh.torus``) and move every vertex onto a smooth random surface, so the connectivity
is kept. On a torus the triangles degrade only as much as the surface stretches; on a
sphere the vertices are then spread evenly over the surface (``deformed_sphere``). Every
shape is centred and scaled to area 4π, so k0, σ, T and the number of modes mean the same
on all of them (Generacion_datos_etapa3.md, §3 and §8). ``delaunay_flip`` removes the
negative cotangent weights a deformation leaves, ``mesh_quality``, the smallest radius of
curvature of a sphere (``smallest_curvature_radius``, from the analytic surface) and the tube
numbers of a torus (``tube_quantities``) feed ``check_gates``, which decides whether a shape is
kept, ``make_shape`` draws until one passes and
``refine_shape`` subdivides a kept shape on its own surface, for a refinement study.
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from .mesh import _check_mesh, cotan_laplacian, icosphere, torus, vertex_areas
from .reference import torus_angles
from .sampling import allowed_wavenumber

AREA = 4 * np.pi
SPINE_HARMONICS = (2, 3)                     # k of the spine terms a_k, b_k
# (j, k) of c_jk: one of each pair ±(j, k), so helical modes (j k ≠ 0) twist both ways.
TUBE_MODES = ((0, 1), (0, 2)) + tuple((j, k) for j in range(1, 4) for k in range(-2, 3))
RELAX_ITERATIONS = (150, 100, 100)           # relaxation steps on icosphere levels n - 2, n - 1 and n


def _normalisation(verts: NDArray, faces: NDArray, area: float = AREA) -> tuple[float, NDArray]:
    """Scale and centre of ``normalize_area``, which returns ``scale · (verts - centre)``."""
    mass = vertex_areas(verts, faces)
    return float(np.sqrt(area / mass.sum())), np.sum(mass[:, None] * verts, axis=0) / mass.sum()


def normalize_area(verts: NDArray, faces: NDArray, area: float = AREA) -> tuple[NDArray, float]:
    """Mesh moved to its area-weighted centroid and scaled to total lumped area ``area``.

    Returns the new vertices and the scale factor applied.
    """
    scale, centroid = _normalisation(verts, faces, area)
    return scale * (verts - centroid), scale


def _sh_degrees(l_min: int, l_max: int) -> tuple[NDArray, NDArray]:
    """(ℓ, m) of every real spherical harmonic, by ℓ = l_min..l_max, then m = -ℓ..ℓ."""
    ell = np.concatenate([np.full(2 * l + 1, l) for l in range(l_min, l_max + 1)])
    m = np.concatenate([np.arange(-l, l + 1) for l in range(l_min, l_max + 1)])
    return ell, m


def real_spherical_harmonics(l_max: int, directions: NDArray, l_min: int = 2) -> NDArray:
    """Real orthonormal spherical harmonics at ``directions`` (N, 3), shape (N, (l_max+1)² - l_min²).

    Columns go by ℓ = l_min..l_max, then m = -ℓ..ℓ: ``√2 N_ℓ|m| Q_ℓ|m|(z) Im (x + iy)^|m|`` for
    m < 0, ``N_ℓ0 Q_ℓ0(z)`` and ``√2 N_ℓm Q_ℓm(z) Re (x + iy)^m`` for m > 0 at the unit vector
    (x, y, z), with ``N_ℓm² = (2ℓ+1)/4π · (ℓ-m)!/(ℓ+m)!`` and ``Q_ℓm = P_ℓ^m / sin^m θ`` (P without
    the Condon–Shortley phase), a polynomial in z by the usual three-term recurrence. These are
    ``√2 (-1)^m Im Y_ℓ^|m|``, ``Y_ℓ^0`` and ``√2 (-1)^m Re Y_ℓ^m`` of ``scipy.special.sph_harm_y``,
    eight times faster (the relaxation of a deformed sphere evaluates them 350 times).
    ``∫ Y_ℓm Y_ℓ'm' dω = δ_ℓℓ' δ_mm'``.
    """
    directions = np.asarray(directions, float)
    x, y, z = (directions / np.linalg.norm(directions, axis=1, keepdims=True)).T
    columns = {}
    re, im = np.ones_like(x), np.zeros_like(x)                  # (x + iy)^m
    for m in range(l_max + 1):
        q_prev, q = 0.0, np.full_like(z, np.prod(np.arange(1.0, 2 * m, 2)))    # Q_mm = (2m - 1)!!
        for l in range(m, l_max + 1):
            if l > m:
                q, q_prev = ((2 * l - 1) * z * q - (l + m - 1) * q_prev) / (l - m), q
            if l >= l_min:
                norm = np.sqrt((2 * l + 1) / (4 * np.pi) / np.prod(np.arange(l - m + 1.0, l + m + 1)))
                columns[l, m] = norm * q if m == 0 else np.sqrt(2) * norm * q * re
                if m > 0:
                    columns[l, -m] = np.sqrt(2) * norm * q * im
        re, im = re * x - im * y, re * y + im * x
    return np.stack([columns[key] for key in zip(*_sh_degrees(l_min, l_max))], axis=1)


@dataclass(frozen=True)
class SphereDeformation:
    """Star-shaped surface ``x(ω) = e^{s(ω)} ω`` with ``s = Σ_{ℓ=2}^{l_max} Σ_m s_ℓm Y_ℓm``.

    ℓ = 0 (size) and ℓ = 1 (almost a translation) are left out: the normalisation fixes both.
    ``coefficients`` are ordered as the columns of ``real_spherical_harmonics``.
    """
    coefficients: tuple[float, ...]

    def __post_init__(self):
        if len(self.coefficients) < 5 or (self.l_max + 1)**2 - 4 != len(self.coefficients):
            raise ValueError("need (l_max + 1)² - 4 coefficients with l_max >= 2")
        object.__setattr__(self, "coefficients", tuple(float(c) for c in self.coefficients))

    @property
    def l_max(self) -> int:
        return int(round(np.sqrt(len(self.coefficients) + 4))) - 1

    @property
    def rms(self) -> float:
        """Root mean square of s over the unit sphere, ``(Σ s_ℓm² / 4π)^½``."""
        return float(np.sqrt(np.sum(np.square(self.coefficients)) / (4 * np.pi)))


def sample_sphere_deformation(rng: np.random.Generator, l_max: int = 4, slope: float = 2.0,
                              tau_range: tuple[float, float] = (0.05, 0.15)) -> SphereDeformation:
    """Random ``s_ℓm ~ N(0, ℓ^-slope)`` (variance), rescaled so that s has RMS τ ~ U(tau_range).

    Draws, in this order: the coefficients, then τ. The defaults come from the planning study:
    at the same τ, l_max = 6 halves the smallest radius of curvature, and τ = 0.15 is the largest
    RMS whose relaxed meshes pass the gates with the data error near the round sphere's.
    """
    ell, _ = _sh_degrees(2, l_max)
    s = rng.standard_normal(len(ell)) * ell**(-slope / 2)
    tau = rng.uniform(*tau_range)
    return SphereDeformation(tuple(s * tau / np.sqrt(np.sum(s**2) / (4 * np.pi))))


def sphere_radius(deformation: SphereDeformation, directions: NDArray) -> NDArray:
    """``ρ(ω) = exp(s(ω))`` at unit ``directions`` (N, 3).

    s is summed with NumPy, not BLAS, whose rounding depends on the number of threads: the
    shape must be the same bit for bit with any thread count.
    """
    Y = real_spherical_harmonics(deformation.l_max, directions)
    return np.exp(np.sum(Y * np.asarray(deformation.coefficients), axis=1))


def _onto_sphere_surface(deformation: SphereDeformation, points: NDArray) -> NDArray:
    """``points`` (N, 3) moved along their rays onto the surface ``ρ(ω) ω``."""
    omega = points / np.linalg.norm(points, axis=1, keepdims=True)
    return sphere_radius(deformation, omega)[:, None] * omega


def _sphere_midpoints(deformation: SphereDeformation, points: NDArray, edges: NDArray) -> NDArray:
    """Surface points at the midpoint directions of ``edges`` (E, 2) of ``points`` on the surface.

    The midpoint direction is the normalised mean of the two directions, as in ``mesh.icosphere``.
    """
    omega = points / np.linalg.norm(points, axis=1, keepdims=True)
    mid = omega[edges].mean(axis=1)
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    return sphere_radius(deformation, mid)[:, None] * mid


def _midpoint_subdivision(faces: NDArray, n_verts: int) -> tuple[NDArray, NDArray]:
    """Every face split in four at its edge midpoints, by the rule of ``mesh.icosphere``.

    Returns the edges (E, 2) whose midpoints become vertices ``n_verts``, ``n_verts + 1``, ... in
    that order, and the new faces; on the faces of ``icosphere(n)`` they are those of ``icosphere(n + 1)``.
    """
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    ab, bc, ca = (n_verts + inverse.reshape(-1, 3)).T
    a, b, c = faces.T
    return unique, np.concatenate([np.stack(f, axis=1) for f in ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))])


def _relax(deformation: SphereDeformation, points: NDArray, faces: NDArray, iterations: int,
           step: float = 0.3, umbrella: float = 0.5) -> NDArray:
    """``iterations`` steps that even out the edges of a mesh whose vertices lie on the surface.

    Each step moves vertex i by ``step · F_i`` without its component along the vertex normal, then
    back onto the surface along its ray. ``F_i = Σ_j (1 - l0/|e_ij|) e_ij / deg_i + umbrella ·
    (mean of the neighbours - x_i)``, with e_ij = x_j - x_i and l0 the current mean edge: springs
    pull every edge toward the mean. Umbrella smoothing alone converges to a near-conformal map,
    whose vertex density follows the conformal factor (≈ e^s), so the edges keep the spread of
    the radial map. Sparse products and elementwise sums only, never BLAS: the result does not
    depend on the number of threads.
    """
    n = len(points)
    _, _, keys = _edge_table(faces, n)
    i, j = np.divmod(keys, n)
    edge = np.arange(len(keys))
    spring = sp.csr_matrix((np.r_[np.ones(len(keys)), -np.ones(len(keys))], (np.r_[i, j], np.r_[edge, edge])),
                           shape=(n, len(keys)))                       # row i: Σ over its edges, as e_ij
    adjacency = sp.csr_matrix((np.ones(2 * len(keys)), (np.r_[i, j], np.r_[j, i])), shape=(n, n))
    corner = sp.csr_matrix((np.ones(faces.size), (faces.ravel(), np.repeat(np.arange(len(faces)), 3))),
                           shape=(n, len(faces)))                      # vertex-face incidence
    degree = np.diff(adjacency.indptr)[:, None]
    x = points
    for _ in range(iterations):
        e = x[j] - x[i]
        length = np.linalg.norm(e, axis=1, keepdims=True)
        force = spring @ ((1 - length.mean() / length) * e) / degree
        force += umbrella * (adjacency @ x / degree - x)
        p = x[faces]
        normal = corner @ np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])          # area-weighted
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        force -= np.sum(force * normal, axis=1, keepdims=True) * normal
        x = _onto_sphere_surface(deformation, x + step * force)
    return x


def _sphere_surface(deformation: SphereDeformation, subdivisions: int, relax: bool = False,
                    iterations: tuple[int, ...] = RELAX_ITERATIONS) -> tuple[NDArray, NDArray]:
    """Mesh of the surface with the faces of ``icosphere(subdivisions)``, before normalisation.

    Without ``relax`` each vertex ω of the icosphere moves to ρ(ω) ω. With it, the same is done
    on the icosphere ``len(iterations) - 1`` levels coarser, which is relaxed (``_relax``) for
    ``iterations[0]`` steps, then subdivided (``_midpoint_subdivision``, new vertices at
    ``_sphere_midpoints``) and relaxed for ``iterations[1]`` steps, and so on up to level
    ``subdivisions``. The coarse levels move the vertex density across the whole surface in few
    steps; at the finest level alone the springs would need many more.
    """
    levels = min(len(iterations), subdivisions + 1) if relax else 1
    omega, faces = icosphere(subdivisions - levels + 1)
    x = sphere_radius(deformation, omega)[:, None] * omega
    for k, steps in enumerate(iterations[len(iterations) - levels:] if relax else ()):
        if k:
            edges, faces = _midpoint_subdivision(faces, len(x))
            x = np.vstack([x, _sphere_midpoints(deformation, x, edges)])
        x = _relax(deformation, x, faces, steps)
    return x, faces


def deformed_sphere(deformation: SphereDeformation, subdivisions: int = 5,
                    relax: bool = True) -> tuple[NDArray, NDArray]:
    """Mesh of the surface ρ(ω) ω with the faces of ``icosphere(subdivisions)``, after ``normalize_area``.

    Moving each icosphere vertex ω radially to ρ(ω) ω (``relax=False``) keeps the vertex density
    of the sphere: where ρ grows the edges stretch, and from an RMS τ of about 0.1 the edge spread
    (p95/mean), then the angles, fail the gates. ``relax`` spreads the vertices evenly over the same
    surface (``_sphere_surface``): every vertex stays on it to rounding, so the shape is kept (the
    first 20 eigenvalues move by less than 0.2 %, within the discretisation error) while the
    triangles improve. Unrelaxed levels nest: the first vertices of level n + 1 are those of level
    n, on the same surface (up to each level's normalisation, a similarity a chordal-area deficit
    away from the identity). Relaxed levels do not; ``refine_shape`` gives a nested finer mesh of either.
    """
    verts, faces = _sphere_surface(deformation, subdivisions, relax)
    return normalize_area(verts, faces)[0], faces


def _harmonic_polynomial(deformation: SphereDeformation, omega: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """``p = Σ s_ℓm Y_ℓm`` with each Y_ℓm the polynomial of ``real_spherical_harmonics`` in u = (x, y, z), not
    normalised: its value (N,), gradient (3, N) and Hessian (3, 3, N) in u at unit ``omega`` (3, N), where p = s.

    With w = (x + iy)^m, the m and -m terms are ``Q_ℓm(z) Re(c w)``, ``c = √2 N_ℓm (s_ℓm - i s_ℓ,-m)`` (``N_ℓ0
    s_ℓ0`` for m = 0); ``∂_y w = i ∂_x w``, and Q_ℓm' and Q_ℓm'' follow the recurrence of Q_ℓm, differentiated.
    Elementwise products only, never BLAS.
    """
    x, y, z = omega
    coefficients = dict(zip(zip(*_sh_degrees(2, deformation.l_max)), deformation.coefficients))
    powers = [np.ones_like(x) + 0j]                                    # (x + iy)^k
    for _ in range(deformation.l_max):
        powers.append(powers[-1] * (x + 1j * y))
    value, grad, hess = np.zeros_like(x), np.zeros((3,) + x.shape), np.zeros((3, 3) + x.shape)
    for m in range(deformation.l_max + 1):
        w = powers[m]
        w1 = m * powers[m - 1] if m >= 1 else 0 * w                    # ∂_x w
        w2 = m * (m - 1) * powers[m - 2] if m >= 2 else 0 * w          # ∂_x² w; ∂_x ∂_y w = i w2, ∂_y² w = -w2
        q_prev, dq_prev, ddq_prev = 0.0, 0.0, 0.0
        q, dq, ddq = np.full_like(z, np.prod(np.arange(1.0, 2 * m, 2))), np.zeros_like(z), np.zeros_like(z)
        for l in range(m, deformation.l_max + 1):
            if l > m:
                a, b = (2 * l - 1) / (l - m), (l + m - 1) / (l - m)
                q, q_prev, dq, dq_prev, ddq, ddq_prev = (a * z * q - b * q_prev, q, a * (q + z * dq) - b * dq_prev, dq,
                                                         a * (2 * dq + z * ddq) - b * ddq_prev, ddq)
            if l < 2:
                continue
            norm = np.sqrt((2 * l + 1) / (4 * np.pi) / np.prod(np.arange(l - m + 1.0, l + m + 1)))
            c = norm * (np.sqrt(2) * (coefficients[l, m] - 1j * coefficients[l, -m]) if m else coefficients[l, 0])
            B, Bx, By, Bxx, Bxy = (c * w).real, (c * w1).real, (1j * c * w1).real, (c * w2).real, (1j * c * w2).real
            value += q * B
            grad += [q * Bx, q * By, dq * B]
            hess += [[q * Bxx, q * Bxy, dq * Bx], [q * Bxy, -q * Bxx, dq * By], [dq * Bx, dq * By, ddq * B]]
    return value, grad, hess


def _outer(u: NDArray, v: NDArray) -> NDArray:
    return u[:, None] * v[None]


def _matvec(A: NDArray, v: NDArray) -> NDArray:
    return np.sum(A * v[None], axis=1)


def sphere_curvatures(deformation: SphereDeformation, directions: NDArray) -> tuple[NDArray, NDArray]:
    """Principal curvatures κ1 ≥ κ2 of the surface at ρ(ω) ω for the ``directions`` ω (N, 3), before normalisation.

    The surface is the zero set of ``G(x) = log|x| - p(x/|x|)``, p the polynomial of ``_harmonic_polynomial`` with
    gradient a and Hessian ∇²p. At x = ρω, ``∇G = (ω - g)/ρ`` with g = Pa the gradient of s on the unit sphere
    (P = I - ωωᵀ), and ``ρ² ∇²G = B = I - 2ωωᵀ - ∇²S``, where ``∇²S = P ∇²p P - aωᵀ - ωaᵀ - (a·ω)(I - 3ωωᵀ)`` is the
    Hessian of S(x) = p(x/|x|) at |x| = 1. The curvatures are the eigenvalues of ``Q ∇²G Q / |∇G|`` on the tangent
    plane, Q = I - nnᵀ with n = (ω - g)/√(1 + |g|²): those of ``C = QBQ`` over ``ρ √(1 + |g|²)``. C is 0 along n, so
    they are ``tr C/2 ± |C - (tr C/2) Q|_F / √2``, without the cancellation of ``√((tr C/2)² - det)``: 1/ρ to rounding
    on a round sphere. Positive where the surface bends away from its outward normal, as a sphere does. Elementwise
    products only, never BLAS.
    """
    omega = (np.asarray(directions, float) / np.linalg.norm(directions, axis=1, keepdims=True)).T     # (3, N)
    value, a, hess_p = _harmonic_polynomial(deformation, omega)
    eye = np.eye(3)[:, :, None]
    a_omega = np.sum(a * omega, 0)
    g = a - a_omega * omega
    h_omega, radial = _matvec(hess_p, omega), _outer(omega, omega)
    hess_S = (hess_p - _outer(omega, h_omega) - _outer(h_omega, omega) + np.sum(omega * h_omega, 0) * radial
              - _outer(a, omega) - _outer(omega, a) - a_omega * (eye - 3 * radial))
    B = eye - 2 * radial - hess_S
    stretch = np.sqrt(1 + np.sum(g * g, 0))
    n = (omega - g) / stretch
    Bn = _matvec(B, n)
    C = B - _outer(n, Bn) - _outer(Bn, n) + np.sum(n * Bn, 0) * _outer(n, n)
    mean = (C[0, 0] + C[1, 1] + C[2, 2]) / 2
    half_gap = np.sqrt(np.sum(np.square(C - mean * (eye - _outer(n, n))), axis=(0, 1)) / 2)
    factor = 1 / (np.exp(value) * stretch)
    return factor * (mean + half_gap), factor * (mean - half_gap)


def _largest_curvature(deformation: SphereDeformation, directions: NDArray) -> NDArray:
    """``max(|κ1|, |κ2|)`` at ρ(ω) ω (``sphere_curvatures``)."""
    k1, k2 = sphere_curvatures(deformation, directions)
    return np.maximum(np.abs(k1), np.abs(k2))


def smallest_curvature_radius(deformation: SphereDeformation, scale: float = 1.0, subdivisions: int = 6,
                              margin: float = 0.8, max_starts: int = 32, tol: float = 1e-10) -> float:
    """``scale / max |κ|``: the smallest radius of curvature of the surface ρ(ω) ω, times ``scale``.

    With the ``scale`` of ``build_shape`` it is in the units of the stored mesh (area 4π); 1 on the round sphere.
    ``max(|κ1|, |κ2|)`` (``sphere_curvatures``) on the directions of ``icosphere(subdivisions)``, then a compass search
    from each of its local maxima on the icosphere's edges that is within ``margin`` of the largest (at most
    ``max_starts``, the largest first): a 5 × 5 grid of tangent steps h around the best point, which moves to the
    grid's best when that is larger beyond rounding and keeps h if it moved to the grid's edge; otherwise h halves,
    from half an edge down to ``tol``. The grid alone is up to 1 % low; over 85 draws (ℓ ≤ 8, τ up to 0.3) the search
    was never below an icosphere(8) grid (655k directions), and agreed with it to 5e-4. Elementwise products only,
    never BLAS: the gate decision does not depend on the number of threads.
    """
    omega, faces = icosphere(subdivisions)
    kappa = _largest_curvature(deformation, omega)
    _, _, keys = _edge_table(faces, len(omega))
    i, j = np.divmod(keys, len(omega))
    neighbours = np.full(len(omega), -np.inf)
    np.maximum.at(neighbours, i, kappa[j])
    np.maximum.at(neighbours, j, kappa[i])
    start = np.flatnonzero((kappa >= neighbours) & (kappa >= margin * kappa.max()))
    start = start[np.argsort(-kappa[start], kind="stable")[:max_starts]]
    p, best = omega[start], kappa[start]
    step = np.full(len(p), 0.5 * np.linalg.norm(omega[i[0]] - omega[j[0]]))
    a, b = (g.ravel() for g in np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij"))
    rows = np.arange(len(p))
    for _ in range(200):                                       # about 30 rounds; each move gains beyond rounding
        if step.max() <= tol:
            break
        axis = np.eye(3)[np.argmin(np.abs(p), axis=1)]          # a tangent frame (e1, e2) at p
        e1 = axis - np.sum(axis * p, 1, keepdims=True) * p
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(p, e1)
        trial = p[:, None] + step[:, None, None] * (a[:, None] * e1[:, None] + b[:, None] * e2[:, None])
        trial /= np.linalg.norm(trial, axis=2, keepdims=True)
        values = _largest_curvature(deformation, trial.reshape(-1, 3)).reshape(len(p), -1)
        k = np.argmax(values, axis=1)
        better = values[rows, k] > best * (1 + 1e-13)
        p = np.where(better[:, None], trial[rows, k], p)
        best = np.where(better, values[rows, k], best)
        step = np.where(better & ((np.abs(a[k]) == 2) | (np.abs(b[k]) == 2)), step, step / 2)
    return float(scale / best.max())


@dataclass(frozen=True)
class TorusDeformation:
    """Tube ``x(u, v) = c(u) + r(u, v) (cos v e1(u) + sin v e2(u))`` around a closed spine c.

    Spine ``c = (R cos u, R sin u, z)`` with ``R = R0 (1 + Σ_k a_k cos(k u + φ_k))`` and
    ``z = R0 Σ_k b_k sin(k u + ψ_k)``, k in ``SPINE_HARMONICS``. Tube radius
    ``r = r0 exp(t)``, ``t = Σ c_jk cos(j u + k v + θ_jk)`` over ``TUBE_MODES`` (j = 0..3,
    k = -2..2: bumps that wind around the tube either way as they go along it). ``alpha = r0/R0``
    with ``R0 r0 = 1/π``, so the round torus (every coefficient zero, the default) has area 4π.
    The frame (e1, e2) is described in ``spine_frame``.
    """
    alpha: float = 0.5
    a: tuple[float, ...] = (0.0,) * len(SPINE_HARMONICS)
    phi: tuple[float, ...] = (0.0,) * len(SPINE_HARMONICS)
    b: tuple[float, ...] = (0.0,) * len(SPINE_HARMONICS)
    psi: tuple[float, ...] = (0.0,) * len(SPINE_HARMONICS)
    c: tuple[float, ...] = (0.0,) * len(TUBE_MODES)
    theta: tuple[float, ...] = (0.0,) * len(TUBE_MODES)

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("alpha = r0/R0 must be in (0, 1)")
        for name, size in [("a", len(SPINE_HARMONICS)), ("phi", len(SPINE_HARMONICS)), ("b", len(SPINE_HARMONICS)),
                           ("psi", len(SPINE_HARMONICS)), ("c", len(TUBE_MODES)), ("theta", len(TUBE_MODES))]:
            if len(getattr(self, name)) != size:
                raise ValueError(f"{name} needs {size} values")
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        object.__setattr__(self, "alpha", float(self.alpha))

    @property
    def major_radius(self) -> float:
        return float(np.sqrt(1 / (np.pi * self.alpha)))

    @property
    def minor_radius(self) -> float:
        return self.alpha * self.major_radius

    @property
    def tube_rms(self) -> float:
        """Root mean square of t over (u, v): ``(Σ c_jk² / 2)^½``.

        The mean of ``cos(j u + k v + θ) cos(j' u + k' v + θ')`` over the torus is half the mean
        of ``cos((j-j') u + (k-k') v + ...) + cos((j+j') u + (k+k') v + ...)``, which is ½ when
        (j', k') = (j, k) and 0 otherwise: neither (0, 0) nor both of a pair ±(j, k) is a mode.
        """
        return float(np.sqrt(np.sum(np.square(self.c)) / 2))


def sample_torus_deformation(rng: np.random.Generator, alpha_range: tuple[float, float] = (0.45, 0.55),
                             a_max: float = 0.04, b_max: float = 0.06,
                             tau_range: tuple[float, float] = (0.0, 0.075)) -> TorusDeformation:
    """Random deformed torus: α, a_k, b_k uniform, phases uniform in [0, 2π), c_jk ~ N(0, 1)
    rescaled so that t has RMS τ ~ U(tau_range) over (u, v).

    Draws, in this order: α, a, φ, b, ψ, c, θ, τ. The defaults come from the planning study:
    above α = 0.55 the hole narrows (round α = 0.6 is at the hole gate) and a packet on the inner
    side reaches across it, and bends fold the inner side more often.
    """
    n_s, n_t = len(SPINE_HARMONICS), len(TUBE_MODES)
    alpha = rng.uniform(*alpha_range)
    a, phi = rng.uniform(-a_max, a_max, n_s), rng.uniform(0, 2 * np.pi, n_s)
    b, psi = rng.uniform(-b_max, b_max, n_s), rng.uniform(0, 2 * np.pi, n_s)
    c, theta = rng.standard_normal(n_t), rng.uniform(0, 2 * np.pi, n_t)
    tau = rng.uniform(*tau_range)
    c = c * tau / np.sqrt(np.sum(c**2) / 2)
    return TorusDeformation(alpha, tuple(a), tuple(phi), tuple(b), tuple(psi), tuple(c), tuple(theta))


def _spine(deformation: TorusDeformation, u: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """c(u), c'(u) and c''(u), shape ``u.shape + (3,)``, from their closed forms."""
    u = np.asarray(u, float)
    R0 = deformation.major_radius
    k = np.array(SPINE_HARMONICS, float)
    a, b = np.array(deformation.a), np.array(deformation.b)
    in_plane = k * u[..., None] + np.array(deformation.phi)
    height = k * u[..., None] + np.array(deformation.psi)
    R = R0 * (1 + np.sum(a * np.cos(in_plane), -1))
    R1 = -R0 * np.sum(a * k * np.sin(in_plane), -1)
    R2 = -R0 * np.sum(a * k**2 * np.cos(in_plane), -1)
    z = R0 * np.sum(b * np.sin(height), -1)
    z1 = R0 * np.sum(b * k * np.cos(height), -1)
    z2 = -R0 * np.sum(b * k**2 * np.sin(height), -1)
    cu, su = np.cos(u), np.sin(u)
    return (np.stack([R * cu, R * su, z], -1),
            np.stack([R1 * cu - R * su, R1 * su + R * cu, z1], -1),
            np.stack([R2 * cu - 2 * R1 * su - R * cu, R2 * su + 2 * R1 * cu - R * su, z2], -1))


def _vertical_frame(c1: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """T = c'/|c'|, e2⁰ = the vertical without its T component, e1⁰ = T × e2⁰ (outward on the round torus)."""
    T = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    e2 = -T[..., 2:] * T
    e2[..., 2] += 1
    e2 /= np.linalg.norm(e2, axis=-1, keepdims=True)
    return T, np.cross(T, e2), e2


def _twist(deformation: TorusDeformation, n_grid: int) -> CubicSpline:
    """``θ(u) = -∫_0^u (ω - ω̄) du'`` with ``ω = (de1⁰/du)·e2⁰``, as a periodic cubic spline.

    ω is the rate at which the vertical frame turns about T; it has zero mean once ω̄ is
    taken out, so θ is periodic. Derivative and integral are spectral on ``n_grid`` points.
    """
    u = 2 * np.pi * np.arange(n_grid) / n_grid
    _, e1, e2 = _vertical_frame(_spine(deformation, u)[1])
    k = np.arange(n_grid // 2 + 1, dtype=float)
    k[-1] = 0                                                   # no Nyquist mode (n_grid is even)
    de1 = np.fft.irfft(1j * k[:, None] * np.fft.rfft(e1, axis=0), n_grid, axis=0)
    omega_hat = np.fft.rfft(np.sum(de1 * e2, -1))
    theta_hat = np.zeros_like(omega_hat)
    theta_hat[1:-1] = -omega_hat[1:-1] / (1j * k[1:-1])
    theta = np.fft.irfft(theta_hat, n_grid)
    theta -= theta[0]
    return CubicSpline(np.append(u, 2 * np.pi), np.append(theta, theta[0]), bc_type="periodic")


def spine_frame(deformation: TorusDeformation, u: NDArray,
                n_grid: int = 4096) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Spine point c, unit tangent T and the tube frame (e1, e2) at ``u``, each ``u.shape + (3,)``.

    The vertical frame (``_vertical_frame``) closes up around the loop but twists about T when
    the spine goes up and down, which shears the tube and its triangles. It is rotated by
    ``θ = -∫ (ω - ω̄)`` (``_twist``), which leaves the constant rate ω̄ (zero twist plus a uniform
    rotation, so the frame still closes): ``(de1/du)·e2 = ω̄``. On the round torus ω = 0, θ = 0
    and ``e1 = (cos u, sin u, 0)``, ``e2 = ẑ``.
    """
    if n_grid < 8 or n_grid % 2:
        raise ValueError("n_grid must be even and at least 8")
    u = np.asarray(u, float)
    c, c1, _ = _spine(deformation, u)
    T, e1, e2 = _vertical_frame(c1)
    theta = _twist(deformation, n_grid)(u)[..., None]
    return c, T, np.cos(theta) * e1 + np.sin(theta) * e2, np.cos(theta) * e2 - np.sin(theta) * e1


def tube_log_radius(deformation: TorusDeformation, u: NDArray, v: NDArray) -> NDArray:
    """``t(u, v) = Σ c_jk cos(j u + k v + θ_jk)``, so that ``r = r0 e^t``."""
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    j, k = np.array(TUBE_MODES, float).T
    phase = j * u[..., None] + k * v[..., None] + np.array(deformation.theta)
    return np.sum(np.array(deformation.c) * np.cos(phase), -1)


def torus_surface(deformation: TorusDeformation, u: NDArray, v: NDArray) -> NDArray:
    """Points ``x(u, v)`` of the deformed torus before normalisation, ``broadcast(u, v).shape + (3,)``.

    v = 0 is the outer side and v = π/2 the top, as in ``reference.torus_points``.
    """
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    c, _, e1, e2 = spine_frame(deformation, u)
    r = deformation.minor_radius * np.exp(tube_log_radius(deformation, u, v))
    return c + r[..., None] * (np.cos(v)[..., None] * e1 + np.sin(v)[..., None] * e2)


def deformed_torus(deformation: TorusDeformation, target_edge: float = 0.037) -> tuple[NDArray, NDArray, float]:
    """``torus(target_edge, R0, r0)`` with each vertex moved from its (u, v) to x(u, v), normalised.

    Returns the vertices, the faces and the normalisation scale (``tube_quantities`` needs
    it). target_edge = 0.037 gives about 10.6k vertices with the mean edge of icosphere(5).
    """
    faces, u, v = _base_torus(deformation, target_edge)
    verts, scale = normalize_area(torus_surface(deformation, u, v), faces)
    return verts, faces, scale


def _base_torus(deformation: TorusDeformation, target_edge: float) -> tuple[NDArray, NDArray, NDArray]:
    """Faces of ``torus(target_edge, R0, r0)`` and the angles (u, v) of its vertices."""
    R0 = deformation.major_radius
    verts, faces = torus(target_edge, R0, deformation.minor_radius)
    return (faces, *torus_angles(verts, R0))


def tube_quantities(deformation: TorusDeformation, scale: float = 1.0, sigmas: tuple[float, ...] = (0.15, 0.2),
                    n_u: int = 1024, n_v: int = 128) -> dict:
    """Thickness, folding, hole and self-clearance of a deformed torus, lengths times ``scale``.

    - ``r_min``: the thinnest tube radius.
    - ``fold``: ``max_u κ(u) r_max(u)``, κ the spine curvature and r_max the largest radius of
      the section at u. At 1 the inner side of a bend folds onto itself.
    - ``hole``: ``2 min_u (|c_xy(u)| - r_max(u))``, the diameter of the widest vertical cylinder
      about the z axis (the spine's axis) that misses the ball of radius r_max(u) around every
      c(u), which holds the section at u. A lower bound of the surface's own clearance from the
      axis (the bulges of t need not point inward) and ``2 (R0 - r0)`` on the round torus. The
      planning study calibrated the hole gate on it: a packet on the inner side puts more of its
      norm across the hole as the hole narrows, and ``check_gates``'s 0.58 (area 4π) keeps that
      below about 6e-3 (σ = 0.2, beyond geodesic 4σ), as on the round α = 0.6 torus.
    - ``gap[str(σ)]`` (G7): the smallest ``|c(u1) - c(u2)| - r_max(u1) - r_max(u2)`` over
      sections at least 7σ apart along the surface, by the lower bound ``∫ |c'| (1 - κ r_max) du``
      of any surface path between them (the shorter way round). Each section lies in the ball
      of radius r_max around c(u), so this bounds the straight-line distance between parts of
      the surface the geodesic cut (§4: 1 up to 7σ) separates. ``None`` if no two sections are
      that far apart, which is common on thick tori: then the opposite sides of the hole are
      within 7σ of each other and only ``hole`` bounds them.

    Evaluated on an ``n_u × n_v`` grid in (u, v). Keys ``str(float(σ))`` and ``None`` (not inf)
    keep the dict strict JSON, unchanged by a round trip through a manifest.
    """
    u = 2 * np.pi * np.arange(n_u) / n_u
    v = 2 * np.pi * np.arange(n_v) / n_v
    c, c1, c2 = _spine(deformation, u)
    speed = np.linalg.norm(c1, axis=-1)
    kappa = np.linalg.norm(np.cross(c1, c2), axis=-1) / speed**3
    t = tube_log_radius(deformation, u[:, None], v[None, :])
    r0 = deformation.minor_radius
    r_max = r0 * np.exp(t.max(1))
    path = speed * np.clip(1 - kappa * r_max, 0, None)
    s = scale * np.concatenate([[0], np.cumsum(0.5 * (path + np.roll(path, -1)) * 2 * np.pi / n_u)])
    along = np.abs(s[:-1, None] - s[None, :-1])
    along = np.minimum(along, s[-1] - along)
    clearance = scale * (cdist(c, c) - r_max[:, None] - r_max[None, :])
    gap = {float(sigma): clearance[along >= 7 * sigma] for sigma in sigmas}
    return {"r_min": float(scale * r0 * np.exp(t.min())),
            "fold": float(np.max(kappa * r_max)),
            "hole": float(2 * scale * np.min(np.hypot(c[:, 0], c[:, 1]) - r_max)),
            "gap": {str(sigma): float(g.min()) if g.size else None for sigma, g in gap.items()}}


def _edge_table(faces: NDArray, n_verts: int) -> tuple[NDArray, NDArray, NDArray]:
    """Every edge of a closed, consistently oriented manifold mesh once, as its two half-edges.

    Half-edge ``3f + c`` is the edge of face f opposite its corner c, from ``faces[f, c+1]`` to
    ``faces[f, c+2]``. Returns both half-edges of every edge and the edge keys ``min·V + max``.
    """
    start, end = faces[:, [1, 2, 0]].ravel(), faces[:, [2, 0, 1]].ravel()
    key = np.minimum(start, end) * n_verts + np.maximum(start, end)
    order = np.argsort(key, kind="stable")
    key = key[order]
    first, second = order[0::2], order[1::2]
    if (len(key) % 2 or np.any(key[0::2] != key[1::2]) or np.any(key[2::2] == key[1:-1:2])
            or np.any(start[first] != end[second])):
        raise ValueError("the mesh must be closed, manifold and consistently oriented")
    return first, second, key[0::2]


def _cot(apex: NDArray, p: NDArray, q: NDArray, denom_eps: float = 1e-10) -> NDArray:
    """Cotangent of the angle at ``apex`` of the triangles (apex, p, q), computed as in ``cotan_laplacian``."""
    u, w = p - apex, q - apex
    return np.einsum("fd,fd->f", u, w) / (np.linalg.norm(np.cross(u, w), axis=1) + denom_eps)


def _corner_cotangents(verts: NDArray, faces: NDArray) -> NDArray:
    """Cotangent of every corner angle, (F, 3); edge weights from it equal ``-L_ij`` bit for bit."""
    corners = verts[faces]
    return np.stack([_cot(corners[:, k], corners[:, (k + 1) % 3], corners[:, (k + 2) % 3]) for k in range(3)], 1)


def delaunay_flip(verts: NDArray, faces: NDArray, tol: float = 1e-12, max_rounds: int = 100) -> tuple[NDArray, int]:
    """Flip edges with negative cotangent weight ``(cot α + cot β)/2`` until none is left.

    Extrinsic flips on a closed, consistently oriented manifold mesh: the edge ij of the faces
    (k, i, j) and (l, j, i) becomes kl, with faces (k, i, l) and (l, j, k); vertices stay. A
    weight is negative below ``-tol`` times the largest |weight|. Each round flips, in every
    face, at most its most negative edge, and only when the flip is legal: kl is not already an
    edge (no duplicate edge, the mesh stays a manifold), the new faces keep the orientation of
    the old pair with non-zero area, and the new weight is larger than the old one (no flip
    back and forth). In the plane these are Lawson's flips, which end at the Delaunay
    triangulation. Stops when no weight is negative, no legal flip is left or after
    ``max_rounds``. Returns the new faces and the number of flips.
    """
    _check_mesh(verts, faces)
    faces = np.array(faces, dtype=np.int64)
    n_flips = 0
    for _ in range(max_rounds):
        first, second, keys = _edge_table(faces, len(verts))
        cot = _corner_cotangents(verts, faces).ravel()
        weight = 0.5 * cot[first] + 0.5 * cot[second]
        bad = np.flatnonzero(weight < -tol * np.abs(weight).max())
        if bad.size == 0:
            break
        f1, c1 = np.divmod(first[bad], 3)
        f2, c2 = np.divmod(second[bad], 3)
        k, i, j, l = faces[f1, c1], faces[f1, (c1 + 1) % 3], faces[f1, (c1 + 2) % 3], faces[f2, c2]
        new_key = np.minimum(k, l) * len(verts) + np.maximum(k, l)
        pk, pi, pj, pl = verts[k], verts[i], verts[j], verts[l]
        old_normal = np.cross(pi - pk, pj - pk) + np.cross(pj - pl, pi - pl)
        size = np.einsum("fd,fd->f", old_normal, old_normal)
        n_a, n_b = np.cross(pi - pk, pl - pk), np.cross(pj - pl, pk - pl)
        new_weight = 0.5 * _cot(pi, pl, pk) + 0.5 * _cot(pj, pk, pl)
        legal = ((k != l) & ~np.isin(new_key, keys)
                 & (np.einsum("fd,fd->f", n_a, old_normal) > 1e-8 * size)
                 & (np.einsum("fd,fd->f", n_b, old_normal) > 1e-8 * size)
                 & (new_weight > weight[bad]))
        if not legal.any():
            break
        # An independent set: each face keeps its most negative legal edge.
        candidates = np.flatnonzero(legal)
        rank = np.empty(len(candidates), np.int64)
        rank[np.argsort(weight[bad][candidates], kind="stable")] = np.arange(len(candidates))
        best = np.full(len(faces), len(candidates))
        np.minimum.at(best, f1[candidates], rank)
        np.minimum.at(best, f2[candidates], rank)
        chosen = candidates[(best[f1[candidates]] == rank) & (best[f2[candidates]] == rank)]
        chosen = chosen[np.unique(new_key[chosen], return_index=True)[1]]      # one flip per new edge
        faces[f1[chosen]] = np.stack([k[chosen], i[chosen], l[chosen]], 1)
        faces[f2[chosen]] = np.stack([l[chosen], j[chosen], k[chosen]], 1)
        n_flips += len(chosen)
    return faces, n_flips


def mesh_quality(verts: NDArray, faces: NDArray, tol: float = 1e-12) -> dict:
    """Quality numbers of a triangle mesh, as plain Python values (for a manifest).

    Angles in degrees; ``n_negative`` counts the off-diagonal ``L_ij > tol · max |L_ij|`` of
    ``cotan_laplacian`` (negative edge weights), ``min_weight`` is the most negative weight
    ``-L_ij``; edge spreads are relative to the mean edge; ``k_allowed`` is
    ``allowed_wavenumber(mean edge, 10)``; ``closed``: every edge in exactly two faces;
    ``oriented``: every directed edge used once; genus from the Euler characteristic, which
    assumes a closed, orientable, connected mesh.
    """
    _check_mesh(verts, faces)
    corners = verts[faces]
    angles = []
    for k in range(3):
        u, w = corners[:, (k + 1) % 3] - corners[:, k], corners[:, (k + 2) % 3] - corners[:, k]
        angles.append(np.arctan2(np.linalg.norm(np.cross(u, w), axis=1), np.einsum("fd,fd->f", u, w)))
    angles = np.degrees(np.concatenate(angles))
    n = len(verts)
    start, end = faces.ravel().astype(np.int64), faces[:, [1, 2, 0]].ravel().astype(np.int64)
    keys, count = np.unique(np.minimum(start, end) * n + np.maximum(start, end), return_counts=True)
    edges = np.stack(np.divmod(keys, n), axis=1)
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    mean_edge = lengths.mean()
    L = cotan_laplacian(verts, faces).tocoo()
    off = L.data[L.row < L.col]
    negative = off > tol * np.abs(off).max()
    graph = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    euler = n - len(edges) + len(faces)
    return {"n_verts": len(verts), "n_faces": len(faces), "n_edges": len(edges),
            "min_angle": float(angles.min()), "max_angle": float(angles.max()),
            "n_negative": int(negative.sum()), "negative_fraction": float(negative.mean()),
            "min_weight": float(-off.max()),
            "mean_edge": float(mean_edge), "edge_p95_ratio": float(np.percentile(lengths, 95) / mean_edge),
            "edge_max_ratio": float(lengths.max() / mean_edge),
            "k_allowed": float(allowed_wavenumber(mean_edge, 10)),
            "closed": bool(np.all(count == 2)), "oriented": len(np.unique(start * n + end)) == len(start),
            "components": int(connected_components(graph, directed=False)[0]),
            "euler": int(euler), "genus": (2 - euler) / 2, "area": float(vertex_areas(verts, faces).sum())}


def check_gates(quality: dict, genus: int, tube: dict | None = None, *, min_angle: float = 30.0,
                max_angle: float = 120.0, max_negative: int = 0, max_edge_p95: float = 1.3,
                min_wavenumber: float = 16.0, min_curvature_radius: float = 0.18, min_tube: float = 0.30,
                max_fold: float = 0.8, min_hole: float = 0.58, min_gap: float = 4.0) -> tuple[bool, list[str]]:
    """Whether a mesh is kept for the dataset, and the reasons when it is not.

    ``quality`` comes from ``mesh_quality`` (after ``delaunay_flip``), ``genus`` is the expected
    one (0 sphere, 1 torus) and ``tube`` the ``tube_quantities`` of a torus. Lengths are in the
    units of the normalised mesh (area 4π). ``min_wavenumber`` bounds ``k0 + 3/(2σ)``: 16.0 covers
    the training ranges (k0 ≤ 6, σ ≥ 0.15); the out-of-distribution k0 = 8 with σ ≥ 0.18 needs
    16.33. ``min_curvature_radius`` bounds ``quality["min_curvature_radius"]``, the smallest radius
    of curvature of a sphere (``smallest_curvature_radius``, recorded by ``build_shape``), when
    ``quality`` has it. The planning study found that the data error and the number of modes follow
    it, not τ: draws with radius ≥ 0.19 kept the error within 1.6 % (k0 = 6) and 2.2 % (k0 = 8), and
    a τ = 0.2 draw with radius 0.11 that passed every other gate reached 3.45 %. ``min_hole`` is a
    length, not a multiple of σ: 2.9σ at σ = 0.2, near the plan's 3σ_max, calibrated in the
    planning study (``tube_quantities``). The gaps must be at least ``min_gap · σ``
    at each σ of ``tube["gap"]``; a ``None`` gap (no two sections 7σ apart) passes. Both dicts may
    come back from a JSON manifest.
    """
    q = quality
    checks = [(q["closed"] and q["oriented"], "not a closed, oriented manifold"),
              (q["components"] == 1, f"{q['components']} components"),
              (q["genus"] == genus, f"genus {q['genus']}, expected {genus}"),
              (q["min_angle"] >= min_angle, f"min angle {q['min_angle']:.1f}° < {min_angle}°"),
              (q["max_angle"] <= max_angle, f"max angle {q['max_angle']:.1f}° > {max_angle}°"),
              (q["n_negative"] <= max_negative, f"{q['n_negative']} negative cotangent weights"),
              (q["edge_p95_ratio"] <= max_edge_p95, f"p95/mean edge {q['edge_p95_ratio']:.3f} > {max_edge_p95}"),
              (q["k_allowed"] >= min_wavenumber, f"allowed wavenumber {q['k_allowed']:.2f} < {min_wavenumber}")]
    if "min_curvature_radius" in q:
        radius = q["min_curvature_radius"]
        checks.append((radius >= min_curvature_radius, f"curvature radius {radius:.3f} < {min_curvature_radius}"))
    if tube is not None:
        checks += [(tube["r_min"] >= min_tube, f"tube radius {tube['r_min']:.3f} < {min_tube}"),
                   (tube["fold"] < max_fold, f"fold {tube['fold']:.3f} >= {max_fold}"),
                   (tube["hole"] >= min_hole, f"hole {tube['hole']:.3f} < {min_hole}")]
        checks += [(gap >= min_gap * float(sigma), f"gap {gap:.3f} < {min_gap}σ at σ = {float(sigma)}")
                   for sigma, gap in tube["gap"].items() if gap is not None]
    reasons = [reason for ok, reason in checks if not ok]
    return not reasons, reasons


SAMPLERS = {"sphere": sample_sphere_deformation, "torus": sample_torus_deformation}
GENUS = {"sphere": 0, "torus": 1}


@dataclass(frozen=True, eq=False)
class Shape:
    """A shape that passed the gates: its mesh, the deformation that rebuilds it and its quality.

    ``quality`` is ``mesh_quality`` plus the number of ``flips``, the normalisation ``scale`` and
    ``offset``, the mesh options (``build_shape``) and, for a sphere, the ``min_curvature_radius``
    or, for a torus, the ``tube_quantities``: plain values, strict JSON. ``rejections`` holds the
    gate failures of the earlier draws.
    """
    family: str
    verts: NDArray
    faces: NDArray
    deformation: SphereDeformation | TorusDeformation
    quality: dict
    attempts: int
    rejections: tuple[tuple[str, ...], ...]


def build_shape(deformation: SphereDeformation | TorusDeformation, *, subdivisions: int = 5,
                target_edge: float = 0.037, sigmas: tuple[float, ...] = (0.15, 0.2),
                relax: bool = True) -> tuple[NDArray, NDArray, dict]:
    """Mesh of a deformation after ``delaunay_flip`` (renormalised to area 4π), and its quality.

    ``subdivisions`` and ``relax`` apply to spheres (``deformed_sphere``) and ``target_edge`` to
    tori. The mesh is ``scale · (x - offset)`` with x the vertices on the surface of the
    deformation, ``quality["scale"]`` and ``quality["offset"]`` (a list of 3 floats), the total
    factor and the shift of the normalisations before and after the flips: a revolution torus
    has radii ``R0 · scale`` and ``r0 · scale``, and ``refine_shape`` maps new surface points the
    same way. ``quality`` also records the mesh options, ``subdivisions`` and ``relax`` for a
    sphere and ``target_edge`` for a torus, so that the deformation and its quality alone rebuild
    and refine the mesh, and the shape's own numbers: ``min_curvature_radius``
    (``smallest_curvature_radius``) for a sphere, the ``tube_quantities`` for a torus, both in the
    units of the mesh. The result depends only on the arguments, bit for bit.
    """
    if isinstance(deformation, SphereDeformation):
        x, faces = _sphere_surface(deformation, subdivisions, relax)
        options = {"subdivisions": int(subdivisions), "relax": bool(relax)}
    else:
        faces, u, v = _base_torus(deformation, target_edge)
        x = torus_surface(deformation, u, v)
        options = {"target_edge": float(target_edge)}
    scale, offset = _normalisation(x, faces)
    verts = scale * (x - offset)
    faces, flips = delaunay_flip(verts, faces)
    if flips:                                      # flips change the area slightly
        rescale, shift = _normalisation(verts, faces)
        verts = rescale * (verts - shift)
        scale, offset = scale * rescale, offset + shift / scale
    quality = mesh_quality(verts, faces) | {"flips": flips, "scale": scale, "offset": [float(a) for a in offset]}
    quality |= options
    if isinstance(deformation, TorusDeformation):
        quality |= tube_quantities(deformation, scale, sigmas)
    else:
        quality["min_curvature_radius"] = smallest_curvature_radius(deformation, scale)
    return verts, faces, quality


def refine_shape(deformation: SphereDeformation | TorusDeformation, verts: NDArray, faces: NDArray, quality: dict,
                 *, target_edge: float | None = None, flip: bool = True) -> tuple[NDArray, NDArray]:
    """The mesh of ``build_shape`` midpoint-subdivided once, with the new vertices on the same surface.

    Each face splits in four (``_midpoint_subdivision``) and each edge gets a vertex at the midpoint
    of its ends' parameters: the normalised mean of their directions on a sphere (the rule of
    ``mesh.icosphere``), the mean of their angles (u, v) on a torus, the shorter way round
    (``target_edge``, by default ``quality["target_edge"]``, rebuilds the angles of the torus mesh).
    New points x are mapped like the stored ones, ``scale · (x - offset)`` with ``quality["scale"]``
    and ``quality["offset"]``, so ``verts`` are exactly the first V rows of the fine mesh and a
    solution on both meshes compares without interpolation. Then ``delaunay_flip`` (``flip``),
    which moves no vertex: splitting a curved mesh makes negative weights even where the coarse mesh
    has none (about 220 of 127k edges on the round torus), and the fine mesh must be as free of them.
    """
    scale, offset = quality["scale"], np.asarray(quality["offset"], float)
    edges, fine_faces = _midpoint_subdivision(np.asarray(faces), len(verts))
    if isinstance(deformation, SphereDeformation):
        new = _sphere_midpoints(deformation, verts / scale + offset, edges)
    else:
        target_edge = quality["target_edge"] if target_edge is None else target_edge
        _, u, v = _base_torus(deformation, target_edge)
        if len(u) != len(verts):
            raise ValueError(f"{len(verts)} vertices, but torus({target_edge}, R0, r0) has {len(u)}")

        def middle(angle):
            start, step = angle[edges[:, 0]], np.mod(angle[edges[:, 1]] - angle[edges[:, 0]] + np.pi, 2 * np.pi) - np.pi
            return np.mod(start + step / 2, 2 * np.pi)

        new = torus_surface(deformation, middle(u), middle(v))
    fine = np.vstack([verts, scale * (new - offset)])
    if flip:
        fine_faces = delaunay_flip(fine, fine_faces)[0]
    return fine, fine_faces


def make_shape(family: str, rng: np.random.Generator, *, max_attempts: int = 100, subdivisions: int = 5,
               target_edge: float = 0.037, sigmas: tuple[float, ...] = (0.15, 0.2), relax: bool = True,
               gates: dict | None = None, **params) -> Shape:
    """Draw ``family`` ("sphere" or "torus") deformations until one passes ``check_gates``.

    ``params`` go to the family's sampler (``sample_sphere_deformation``,
    ``sample_torus_deformation``) and ``gates`` to ``check_gates``. The same ``rng`` state
    gives the same shape bit for bit, and ``build_shape(shape.deformation)`` with the mesh
    options recorded in ``shape.quality`` rebuilds it. Raises ``RuntimeError`` after ``max_attempts`` rejections.
    """
    if family not in SAMPLERS:
        raise ValueError(f"family must be one of {sorted(SAMPLERS)}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    rejections = []
    for attempt in range(1, max_attempts + 1):
        deformation = SAMPLERS[family](rng, **params)
        verts, faces, quality = build_shape(deformation, subdivisions=subdivisions, target_edge=target_edge,
                                            sigmas=sigmas, relax=relax)
        tube = quality if family == "torus" else None
        passed, reasons = check_gates(quality, GENUS[family], tube, **(gates or {}))
        if passed:
            return Shape(family, verts, faces, deformation, quality, attempt, tuple(rejections))
        rejections.append(tuple(reasons))
    raise RuntimeError(f"no {family} passed the gates in {max_attempts} draws; last: {rejections[-1]}")
