"""Torus mesh: topology, geometry, triangle quality and convergence of the cotangent Laplacian."""
import numpy as np
import pytest
import scipy.linalg as sla
from schrodinger_glno.mesh import (_zip_ring_loop, blended_mass, cotan_laplacian, laplacian_eigenpairs, torus,
                                   vertex_areas, vertex_normals)

# (major radius, minor radius, target edge): the default torus, thin and fat ones, and a very coarse one.
GEOMETRIES = [(1.0, 0.5, 0.1), (2.0, 0.5, 0.15), (3.0, 1.0, 0.25), (0.7, 0.5, 0.1), (1.0, 0.2, 0.05),
              (1.0, 0.5, 0.45)]


def _edges(faces):
    """Unique undirected edges and how many faces contain each."""
    return np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0, return_counts=True)


def _angles(verts, faces):
    """Interior angles of every face, in degrees, shape (F, 3)."""
    corners = verts[faces]
    out = []
    for k in range(3):
        a, b = corners[:, (k + 1) % 3] - corners[:, k], corners[:, (k + 2) % 3] - corners[:, k]
        cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.stack(out, axis=1)


def _outward_normal(points, major_radius):
    """Unit outward normal of the torus at (or nearest to) ``points``."""
    u = np.arctan2(points[:, 1], points[:, 0])
    v = np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1]) - major_radius)
    return np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], axis=1)


def _cos_v(points, major_radius):
    """cos v: +1 on the outer equator (ρ = R + r), -1 on the inner one (ρ = R - r)."""
    s = np.hypot(points[:, 0], points[:, 1]) - major_radius
    return s / np.hypot(s, points[:, 2])


def _rings(verts, major_radius):
    """Ring of every vertex: rings of constant v, numbered by increasing v ∈ [0, 2π)."""
    v = np.mod(np.arctan2(verts[:, 2], np.hypot(verts[:, 0], verts[:, 1]) - major_radius), 2 * np.pi)
    order = np.argsort(v, kind="stable")
    ring = np.empty(len(v), int)
    ring[order] = np.cumsum(np.concatenate([[0], np.diff(v[order]) > 1e-6]))
    return ring


def _equal_count_strips(verts, major_radius):
    """Strips between consecutive rings of equal count: lower ring a (a + 1 → 0 for the last one,
    which closes the loop) and |phase - ½|, how far ring a + 1 is from sitting half a step of ring
    a ahead (0: staggered, ½: aligned)."""
    ring = _rings(verts, major_radius)
    counts = np.bincount(ring)
    u = np.arctan2(verts[:, 1], verts[:, 0])[np.unique(ring, return_index=True)[1]]   # one point per ring
    a = np.flatnonzero(counts == np.roll(counts, -1))
    b = (a + 1) % len(counts)
    step = 2 * np.pi / counts[a]
    return a, np.abs(np.mod(u[b] - u[a], step) / step - 0.5)


def _is_closed_oriented_manifold(faces, n_verts):
    """Each directed edge is used once and its reverse once (watertight, consistently oriented), χ = 0."""
    head = np.roll(faces, -1, axis=1).astype(np.int64)
    forward = np.sort((faces.astype(np.int64) * n_verts + head).ravel())
    backward = (head * n_verts + faces).ravel()
    return bool(np.all(np.diff(forward) > 0) and np.all(np.isin(backward, forward, assume_unique=True))
                and len(faces) == 2 * n_verts)


def _chordal_area_deficit(major_radius, minor_radius, h):
    """Relative area lost by the inscribed mesh: (κ_v² t² + ⟨κ_u²⟩ h²) / 24.

    A chord of length ℓ on a circle of curvature κ is shorter than its arc by κ²ℓ²/24.
    Rings are t = h√3/2 apart along v (κ_v = 1/r) and have steps ≈ h along u, whose
    normal curvature is κ_u = cos v / ρ (averaged with the area weight ρ).
    """
    v = 2 * np.pi * np.arange(256) / 256
    rho = major_radius + minor_radius * np.cos(v)
    mean_ku2 = np.sum(np.cos(v)**2 / rho) / np.sum(rho)
    return h**2 / 24 * (0.75 / minor_radius**2 + mean_ku2)


def _exact_eigenvalues(major_radius, minor_radius, count, n=63, m_max=8):
    """Smallest eigenvalues of -Δ on the smooth torus, by separation of variables.

    With f = e^{imu} g(v), -Δf = λf becomes -(ρ g')'/r² + m² g/ρ = λ ρ g; a Fourier
    pseudo-spectral discretization in v resolves the low modes to ~1e-11. m ≠ 0 counts twice.
    """
    v = 2 * np.pi * np.arange(n) / n
    rho = major_radius + minor_radius * np.cos(v)
    k = np.fft.fftfreq(n, 1.0 / n)
    D = np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0))
    stiffness = D.T @ (rho[:, None] * D) / minor_radius**2
    evals = [np.repeat(sla.eigh(stiffness + np.diag(m**2 / rho), np.diag(rho), eigvals_only=True), 1 if m == 0 else 2)
             for m in range(m_max + 1)]
    return np.sort(np.concatenate(evals))[:count]


@pytest.mark.parametrize("R, r, h", GEOMETRIES)
def test_torus_is_a_closed_outward_oriented_manifold_on_the_surface(R, r, h):
    verts, faces = torus(h, R, r)
    assert verts.dtype == float and verts.shape[1] == 3 and faces.shape[1] == 3
    assert np.issubdtype(faces.dtype, np.integer)
    assert np.array_equal(np.unique(faces), np.arange(len(verts)))              # every vertex is used
    assert np.all((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0]))
    assert len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces)          # no duplicate faces
    edges, count = _edges(faces)
    assert np.all(count == 2) and len(verts) - len(edges) + len(faces) == 0      # watertight, χ = 0
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    assert len(np.unique(directed, axis=0)) == len(directed)                     # consistent orientation
    s = np.hypot(verts[:, 0], verts[:, 1])
    np.testing.assert_allclose((s - R)**2 + verts[:, 2]**2, r**2, rtol=0, atol=1e-13)
    corners = verts[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    cos = np.sum(face_normals * _outward_normal(corners.mean(1), R), 1) / np.linalg.norm(face_normals, axis=1)
    assert cos.min() > 0.9                       # ≥ 0.96 even at h = 0.45 r; ≥ 0.99 at h ≤ 0.2 r
    # Packets take their tangent plane from vertex_normals: 0.985 at h = 0.45 r, ≥ 0.999 at h ≤ 0.2 r.
    normal_dot = np.sum(vertex_normals(verts, faces) * _outward_normal(verts, R), 1)
    assert normal_dot.min() > (0.98 if h > 0.3 * r else 0.995)


@pytest.mark.parametrize("R, r, h", GEOMETRIES)
def test_torus_triangles_are_near_equilateral_everywhere(R, r, h):
    verts, faces = torus(h, R, r)
    angles = _angles(verts, faces)
    edges, _ = _edges(faces)
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1) / h
    valence = np.bincount(edges.ravel())
    mass = vertex_areas(verts, faces)
    # Measured on all GEOMETRIES: angles in [40.7°, 91.4°], edges in [0.85 h, 1.34 h], area ratio ≤ 1.44.
    # The few angles just above 90° are the near-right pairs at 5-7 dislocations, unavoidable on rings.
    assert angles.min() > 40 and angles.max() < 92
    assert np.mean(angles.max(1) > 90) < 0.02
    assert 0.84 < lengths.min() and lengths.max() < 1.35 and abs(lengths.mean() - 1) < 0.05
    assert valence.min() >= 5 and valence.max() <= 7
    assert mass.max() / mass.min() < 1.5


@pytest.mark.parametrize("R, r", [(1.0, 0.5), (2.0, 0.5), (3.0, 0.2), (5.0, 0.3)])
def test_torus_quality_holds_over_a_sweep_of_resolutions(R, r):
    """25 resolutions per shape, from fine to target_edge = 0.95 r, all with R - r ≥ 2 h.

    Measured: angles 38.8-91.1°, edges 0.806-1.349 h, area ratio ≤ 1.452, equal-count strips ≤ 0.244
    step off the stagger. Nearer the horn torus (R - r < 2 h, a dozen triangles or fewer around the
    inner equator) these bounds are not promised.
    """
    failures = []
    for h in np.geomspace(0.04, 0.95 * r, 25):
        if R - r < 2 * h:
            continue
        verts, faces = torus(h, R, r)
        angles = _angles(verts, faces)
        edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        edges = edges[edges[:, 0] < edges[:, 1]]
        lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1) / h
        valence = np.bincount(faces.ravel())                     # = number of neighbours on a closed manifold
        mass = vertex_areas(verts, faces)
        checks = {"manifold": _is_closed_oriented_manifold(faces, len(verts)),
                  "valence 5-7": valence.min() >= 5 and valence.max() <= 7,
                  "angles": angles.min() > 38 and angles.max() < 92,
                  "edges": lengths.min() > 0.78 and lengths.max() < 1.38,
                  "area ratio": mass.max() / mass.min() < 1.5,
                  "stagger": _equal_count_strips(verts, R)[1].max(initial=0) < 0.3}
        failures += [(round(h, 4), name) for name, ok in checks.items() if not ok]
    assert not failures


@pytest.mark.parametrize("R, r, h", [(1.0, 0.5, 0.1), (1.0, 0.5, 0.064), (2.0, 0.5, 0.102), (5.0, 0.3, 0.08)])
def test_torus_closing_strip_is_staggered_like_the_other_equal_count_strips(R, r, h):
    verts, faces = torus(h, R, r)
    strips, off_stagger = _equal_count_strips(verts, R)
    ring = _rings(verts, R)
    last = ring.max()
    assert strips[-1] == last                   # the strip that closes the loop (last ring → ring 0) is one of them
    # Its phase error is shared evenly by all of them. Before, it alone was off: by 0.233, 0.499
    # (aligned rings: a seam of right triangles), 0.472 and 0.498 step; now 0.116, 0.100, 0.236, 0.159.
    assert np.ptp(off_stagger) < 1e-9 and off_stagger.max() < 0.25
    closing = np.all(np.isin(ring[faces], [last, 0]), axis=1)
    assert closing.sum() == 2 * np.sum(ring == 0)
    angles = _angles(verts, faces[closing])
    assert angles.min() > 45 and angles.max() < 80


@pytest.mark.parametrize("R, r, h", [(2.0, 0.5, 0.4211), (5.0, 0.3, 0.2713)])
def test_torus_has_no_valence_4_vertex_where_the_loop_closes_on_coarse_meshes(R, r, h):
    # The first pass leaves one on ring 0 (area ratio 1.74, 1.75); redone from the inner side: 1.42, 1.40.
    verts, faces = torus(h, R, r)
    valence = np.bincount(faces.ravel())
    mass = vertex_areas(verts, faces)
    assert _is_closed_oriented_manifold(faces, len(verts))
    assert valence.min() == 5 and valence.max() == 7 and mass.max() / mass.min() < 1.45


@pytest.mark.parametrize("counts", [[112, 110, 105, 100, 98, 100, 105, 110],          # R, r, h = 3, 0.2, 0.18
                                    [40, 38, 34, 28, 25, 25, 28, 34, 38]])            # 2, 0.5, 0.389
def test_torus_first_pass_scores_the_strip_that_closes_the_loop(counts):
    # Without that term the last ring leaves a valence-4 vertex on ring 0 here, and torus() needs its retry.
    counts = np.array(counts)
    starts = np.concatenate([[0], np.cumsum(counts)])
    _, faces, excess, _ = _zip_ring_loop(counts, starts)
    assert _is_closed_oriented_manifold(faces, counts.sum())
    assert np.abs(excess).max() == 1 and np.array_equal(excess + 6, np.bincount(faces.ravel()))


@pytest.mark.parametrize("R, r, h", [(0.51, 0.5, 0.1), (0.6, 0.5, 0.2), (1.0, 0.95, 0.3), (1.0, 0.5, 0.45)])
def test_torus_near_the_horn_torus_is_still_an_embedding(R, r, h):
    """Few points on the inner rings degrade the triangles but never fold the mesh: in (u, v) every
    face is positively oriented and together they cover the square exactly once."""
    verts, faces = torus(h, R, r)
    assert _is_closed_oriented_manifold(faces, len(verts))
    u = np.arctan2(verts[:, 1], verts[:, 0])
    v = np.arctan2(verts[:, 2], np.hypot(verts[:, 0], verts[:, 1]) - R)
    du, dv = (np.angle(np.exp(1j * (w[faces[:, 1:]] - w[faces[:, :1]]))) for w in (u, v))
    signed_area = (du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]) / 2
    assert signed_area.min() > 0 and signed_area.sum() == pytest.approx(4 * np.pi**2, rel=1e-12)


def test_torus_at_production_resolution_is_uniform_from_inner_to_outer_side():
    R, r, h = 1.0, 0.5, 0.03
    verts, faces = torus(h, R, r)
    area = 4 * np.pi**2 * R * r
    assert len(verts) == pytest.approx(area / (np.sqrt(3) / 2 * h**2), rel=0.02)   # 25 346 vertices
    angles = _angles(verts, faces)
    assert angles.min() > 40 and angles.max() < 90.5
    edges, _ = _edges(faces)
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    assert 0.85 * h < lengths.min() and lengths.max() < 1.33 * h
    mass = vertex_areas(verts, faces)
    assert mass.max() / mass.min() < 1.45
    # On a (u, v) grid the u-edges and the vertex areas of the inner side would be (R - r)/(R + r) = 1/3 as big.
    inner, outer = _cos_v(verts[edges].mean(1), R) < -0.8, _cos_v(verts[edges].mean(1), R) > 0.8
    assert lengths[inner].mean() / lengths[outer].mean() == pytest.approx(1, abs=0.02)
    inner, outer = _cos_v(verts, R) < -0.8, _cos_v(verts, R) > 0.8
    assert mass[inner].mean() / mass[outer].mean() == pytest.approx(1, abs=0.02)


def test_torus_area_matches_the_chordal_estimate_and_converges_at_second_order():
    deficits = {}
    for R, r, h in [(1.0, 0.5, 0.2), (1.0, 0.5, 0.1), (1.0, 0.5, 0.05), (2.0, 0.5, 0.1), (0.7, 0.5, 0.1)]:
        verts, faces = torus(h, R, r)
        exact = 4 * np.pi**2 * R * r
        deficit = (exact - vertex_areas(verts, faces).sum()) / exact
        # Inscribed mesh: always smaller; measured/estimate within 0.94..1.01 for these cases.
        assert deficit > 0
        assert deficit / _chordal_area_deficit(R, r, h) == pytest.approx(1, abs=0.15)
        deficits[R, r, h] = deficit
    order = [np.log2(deficits[1.0, 0.5, 2 * h] / deficits[1.0, 0.5, h]) for h in (0.1, 0.05)]
    assert np.all((1.8 < np.array(order)) & (np.array(order) < 2.2))


@pytest.mark.parametrize("h", [0.2, 0.04])
def test_torus_cotan_laplacian_is_symmetric_with_nonnegative_weights(h):
    verts, faces = torus(h)
    L = cotan_laplacian(verts, faces)
    assert abs(L - L.T).max() < 1e-13
    np.testing.assert_allclose(L @ np.ones(len(verts)), 0, atol=1e-12)
    coo = L.tocoo()
    weights = -coo.data[coo.row != coo.col]
    # Cotan weight -L_ij ≥ 0 on every edge (no pair of opposite angles sums past 180°). The
    # diagonals of the near-right pairs (0.1-0.4 % of the edges) are cyclic quads: weight 0
    # up to round-off (~1e-14), hence the tolerance. Equilateral would give cot 60° = 0.577.
    assert weights.min() > -1e-12
    assert np.mean(weights < 1e-8) < 0.01 and 0.5 < np.median(weights) < 0.75


def test_torus_cotan_laplacian_is_psd_with_a_one_dimensional_kernel():
    verts, faces = torus(0.2)
    evals = np.linalg.eigvalsh(cotan_laplacian(verts, faces).toarray())
    assert evals[0] > -1e-10 and abs(evals[0]) < 1e-10
    assert evals[1] > 1e-3                                     # connected: constants are the only kernel


def test_torus_laplacian_eigenvalues_converge_to_the_smooth_torus():
    R, r, count = 1.0, 0.5, 13              # λ0 = 0 and 12 nonzero ones; index 12 closes a degenerate pair
    exact = _exact_eigenvalues(R, r, count)
    assert exact[12] < 0.99 * _exact_eigenvalues(R, r, count + 1)[13]
    spectra = []
    for h in (0.24, 0.17, 0.12, 0.085):
        verts, faces = torus(h, R, r)
        L = cotan_laplacian(verts, faces)
        evals = laplacian_eigenpairs(L, blended_mass(verts, faces), count)[0]
        assert abs(evals[0]) < 1e-8
        spectra.append(evals[1:])
    spectra = np.array(spectra)
    changes = np.max(np.abs(np.diff(spectra, axis=0)) / exact[1:], axis=1)
    errors = np.max(np.abs(spectra / exact[1:] - 1), axis=1)
    # h halves from 0.17 to 0.085: an O(h²) method changes / errs ~4x less (measured 0.29, 3.97).
    assert changes[-1] < 0.4 * changes[0]
    assert 3 < errors[1] / errors[3] < 5
    assert errors[3] < 2.5e-3                                    # 0.165 % at 3 175 vertices


def test_torus_is_deterministic():
    first, second = torus(0.1, 1.3, 0.4), torus(0.1, 1.3, 0.4)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


@pytest.mark.parametrize("h, R, r", [(0.1, 0.5, 0.5), (0.1, 0.4, 0.5), (0.1, 1.0, 0.0), (0.1, 1.0, -0.5),
                                     (0.0, 1.0, 0.5), (-0.1, 1.0, 0.5), (0.5, 1.0, 0.5), (0.7, 1.0, 0.5),
                                     (np.nan, 1.0, 0.5), (0.1, np.inf, 0.5), (0.1, np.nan, 0.5)])
def test_torus_rejects_bad_parameters(h, R, r):
    with pytest.raises(ValueError):
        torus(h, R, r)
