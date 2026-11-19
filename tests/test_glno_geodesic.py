"""Heat-method geodesic distance and the geodesic cutoff of the stage-3 packet."""
from functools import lru_cache
import numpy as np
import pytest
from schrodinger_glno.geodesic import HeatGeodesics, cut_gaussian_packet, geodesic_cutoff
from schrodinger_glno.initial_conditions import gaussian_packet, mass_norm
from schrodinger_glno.mesh import blended_mass, icosphere, mean_edge_length, torus, vertex_areas, vertex_normals
from schrodinger_glno.reference import torus_angles
from schrodinger_glno.sampling import direction_from_angle


@pytest.fixture(scope="module")
def sphere5():
    verts, faces = icosphere(5)
    return verts, faces, HeatGeodesics(verts, faces)


@lru_cache(maxsize=1)
def _torus_geodesics(edge, scale):
    verts, faces = torus(edge, scale, 0.5 * scale)
    return verts, faces, HeatGeodesics(verts, faces)


def _sphere_median_error(verts, faces, geodesics):
    """Median relative error against the great-circle distance arccos(x·x0), over four sources,
    leaving out targets within 3 edges of the source (where the heat method smooths the cone)."""
    h = mean_edge_length(verts, faces)
    errors = []
    for source in (0, 7, len(verts) // 3, len(verts) - 1):
        exact = np.arccos(np.clip(verts @ verts[source], -1, 1))
        far = exact > 3 * h
        errors.append(np.abs(geodesics.distance(source)[far] - exact[far]) / exact[far])
    return float(np.median(np.concatenate(errors)))


def _outer_equator_error(edge):
    """Mean relative error along the outer equator of the torus R = 1, r = 0.5, a geodesic of
    length (R + r)Δu (minimizing up to its conjugate point, Δu ≈ 1.8), for 0.15 < Δu < 1.6."""
    verts, faces = torus(edge, 1.0, 0.5)
    geodesics = HeatGeodesics(verts, faces)
    u, v = torus_angles(verts, 1.0)
    ring = np.flatnonzero(v == 0)                     # the ring v = 0 lies exactly on the outer equator
    errors = []
    for source in ring[:: len(ring) // 4][:4]:
        du = np.abs(np.angle(np.exp(1j * (u[ring] - u[source]))))
        sel = (du > 0.15) & (du < 1.6)
        exact = 1.5 * du[sel]
        errors.append(np.abs(geodesics.distance(source)[ring][sel] - exact) / exact)
    return float(np.mean(np.concatenate(errors)))


def test_distance_is_zero_at_the_source_and_nonnegative(sphere5):
    verts, faces, geodesics = sphere5
    tverts, tfaces = torus(0.06)
    for geo, n in ((geodesics, len(verts)), (HeatGeodesics(tverts, tfaces), len(tverts))):
        for source in (0, n // 2, n - 1):
            d = geo.distance(source)
            assert d.shape == (n,) and d[source] == 0
            assert d.min() >= 0 and np.sum(d == 0) == 1


def test_sphere_distance_matches_great_circles_and_converges(sphere5):
    # Measured: 0.91 % (level 4, 2 562 vertices) and 0.70 % (level 5, 10 242). The worst single
    # source of five gives 0.99 %, 0.83 % and 0.75 % at levels 4, 5 and 6: it converges, slowly.
    coarse = _sphere_median_error(*icosphere(4), HeatGeodesics(*icosphere(4)))
    fine = _sphere_median_error(*sphere5)
    assert fine < coarse < 0.012
    assert fine < 0.01


def test_torus_outer_equator_distance_converges():
    # Measured: 2.1 % at edge 0.06 (6 289 vertices), 1.7 % at edge 0.04 (14 294).
    coarse, fine = _outer_equator_error(0.06), _outer_equator_error(0.04)
    assert 0.01 < coarse < 0.03
    assert fine < coarse - 0.002


def test_distance_is_nearly_symmetric(sphere5):
    """d(a, b) ≠ d(b, a) only through discretization: measured ≤ 0.25 h on the sphere and
    ≤ 0.65 h (3.6 % at worst) on the torus."""
    rng = np.random.default_rng(0)
    tverts, tfaces = torus(0.06)
    for verts, faces, geodesics in (sphere5, (tverts, tfaces, HeatGeodesics(tverts, tfaces))):
        sources = rng.choice(len(verts), 8, replace=False)
        D = np.stack([geodesics.distance(s) for s in sources])[:, sources]
        assert np.abs(D - D.T).max() < mean_edge_length(verts, faces)
        off = ~np.eye(len(sources), dtype=bool)
        assert np.median(np.abs(D - D.T)[off] / D[off]) < 0.01


def test_distance_does_not_depend_on_the_mesh_units(sphere5):
    verts, faces, geodesics = sphere5
    d = geodesics.distance(0)
    for scale in (1e-3, 1e3):                                     # measured ≤ 6e-13
        assert np.abs(HeatGeodesics(scale * verts, faces).distance(0) / scale - d).max() < 1e-9


def test_distance_far_from_the_source_until_the_heat_solution_underflows():
    """u ≈ e^{-d/h} must stay a normal double, which holds to about 700 h. The inner equator of
    a torus is a minimizing geodesic (a path that turns by Δu about the axis is at least
    (R - r)Δu long). On torus(0.1, 13, 0.3) it is still right 400 h from the source, where
    u ≈ 1e-168 and |∇u|² underflows: measured +1.5 % on average, 1.7 % at worst for Δu > 0.5.
    About 760 h away on torus(0.1, 24, 0.15), u is subnormal and that raises."""
    R, r = 13.0, 0.3
    verts, faces = torus(0.1, R, r)
    u, v = torus_angles(verts, R)
    ring = np.flatnonzero(np.abs(np.angle(np.exp(1j * (v - np.pi)))) < 1e-12)
    du = np.abs(np.angle(np.exp(1j * (u[ring] - u[ring[0]]))))
    d = HeatGeodesics(verts, faces).distance(ring[0])
    assert d.max() > 380 * mean_edge_length(verts, faces)
    sel = du > 0.5
    assert np.abs(d[ring][sel] / ((R - r) * du[sel]) - 1).max() < 0.025
    with pytest.raises(ValueError, match="underflowed"):
        HeatGeodesics(*torus(0.1, 24.0, 0.15)).distance(0)


def test_rejects_disconnected_or_degenerate_meshes_and_bad_sources():
    verts, faces = icosphere(2)
    two = (np.vstack([verts, verts + 3]), np.vstack([faces, faces + len(verts)]))
    with pytest.raises(ValueError, match="connected"):
        HeatGeodesics(*two)
    with pytest.raises(ValueError, match="connected"):            # a vertex in no face
        HeatGeodesics(np.vstack([verts, [[5.0, 0, 0]]]), faces)
    collapsed = verts.copy()
    collapsed[faces[0, 1]] = collapsed[faces[0, 0]]
    with pytest.raises(ValueError, match="degenerate"):
        HeatGeodesics(collapsed, faces)
    geodesics = HeatGeodesics(verts, faces)
    for source in (-1, len(verts)):
        with pytest.raises(ValueError, match="source"):
            geodesics.distance(source)


def _flat_ellipsoid():
    """x² + y² + (z/0.1)² = 1 from icosphere(4): exact pole vertices, and 912 edges with
    negative cotangent weights in the band around the rim."""
    verts, faces = icosphere(4)
    return verts * np.array([1.0, 1.0, 0.1]), faces


def test_rejects_a_source_whose_heat_solution_turns_negative():
    """On a mesh with negative cotangent weights u can be negative, and the distance is then
    silently wrong. From the rim vertex 0 of the flat ellipsoid u < 0 at 33 vertices (on the
    rotated mesh with an exact meridian a rim source gets 0.78 instead of 1.02 to the pole);
    from the poles u stays positive."""
    verts, faces = _flat_ellipsoid()
    geodesics = HeatGeodesics(verts, faces)
    assert abs(verts[0, 2]) < 1e-12
    with pytest.raises(ValueError, match="negative cotangent weights"):
        geodesics.distance(0)
    geodesics.distance(int(np.argmax(verts[:, 2])))


def test_cutoff_is_one_then_a_c2_smoothstep_then_zero():
    sigma, inner, outer = 0.2, 7.0, 9.0
    d = np.array([0.0, 1.0, inner * sigma, 8 * sigma, outer * sigma, 2.0, 10.0])
    assert np.allclose(geodesic_cutoff(d, sigma), [1, 1, 1, 0.5, 0, 0, 0], atol=1e-15)
    s = np.linspace(0, 1, 101) * sigma                           # w - ½ is odd about the band's middle
    assert np.allclose(geodesic_cutoff(8 * sigma + s, sigma) + geodesic_cutoff(8 * sigma - s, sigma), 1)
    band = np.linspace(inner, outer, 1001) * sigma
    assert np.all(np.diff(geodesic_cutoff(band, sigma)) < 0)
    # Value, slope and curvature continuous at both ends: the weight leaves 1 and reaches 0 as
    # 10 x³ (x = distance into the band / its width), so (1 - w)/x³ and w/x³ tend to 10.
    width = (outer - inner) * sigma
    x = np.array([1e-2, 1e-3, 1e-4])
    assert np.allclose((1 - geodesic_cutoff(inner * sigma + x * width, sigma)) / x**3, 10, rtol=0.2)
    assert np.allclose(geodesic_cutoff(outer * sigma - x * width, sigma) / x**3, 10, rtol=0.2)
    # Its second derivative (finite differences) has no jump anywhere, the ends included.
    grid = np.linspace(6, 10, 4001) * sigma
    step = grid[1] - grid[0]
    second = np.diff(geodesic_cutoff(grid, sigma), 2) / step**2
    assert np.abs(np.diff(second)).max() < 60 / width**3 * step * 1.01


def test_cutoff_rejects_bad_arguments():
    for kwargs in ({"sigma": 0.0}, {"sigma": np.inf}, {"sigma": 0.2, "inner": 9.0, "outer": 7.0},
                   {"sigma": 0.2, "inner": -1.0}):
        with pytest.raises(ValueError):
            geodesic_cutoff(np.ones(3), **kwargs)


def _cut_and_chordal(verts, normals, mass, center, k0, sigma, phi, distance):
    direction = direction_from_angle(normals[center], phi)
    psi = gaussian_packet(verts, mass, normals, center, k0, sigma, direction)
    cut, removed = cut_gaussian_packet(verts, mass, normals, center, k0, sigma, direction, distance)
    return psi, cut, removed


def test_cut_packet_on_the_unit_sphere_is_the_chordal_one(sphere5):
    """7σ = 1.4 rad on the unit sphere at σ = 0.2: the cut removes ≤ 4.4e-11 of ‖ψ‖² and changes ψ
    by ≤ 2.5e-6 in the M-norm (≤ 1.3e-6 of max |ψ| at any vertex), measured over 4 centres,
    k0 ∈ {1, 6} and 3 directions, lumped or blended mass; the two centres here reach the same
    worst values. Both depend only on |ψ|, so k0 and the direction do not matter with lumped
    mass and hardly with blended mass; two pairs cover that small phase effect."""
    verts, faces, geodesics = sphere5
    normals = vertex_normals(verts, faces)
    for center in (0, 5000):
        d = geodesics.distance(center)
        for mass in (blended_mass(verts, faces), vertex_areas(verts, faces)):
            for k0, phi in ((1.0, 0.0), (6.0, 2.0)):
                psi, cut, removed = _cut_and_chordal(verts, normals, mass, center, k0, 0.2, phi, d)
                assert abs(mass_norm(cut, mass) - 1) < 1e-12
                assert 0 <= removed < 1e-10
                assert mass_norm(cut - psi, mass) < 5e-6


@pytest.mark.parametrize("edge, scale, sigma, removed_bound, change_bound", [
    (0.03, 1.0, 0.12, 1e-9, 1e-5),
    (0.03, 1.0, 0.2, 2e-6, 6e-4),
    (0.038, np.sqrt(2 / np.pi), 0.15, 2e-6, 9e-4),
])
def test_cut_packet_on_the_torus_changes_little(edge, scale, sigma, removed_bound, change_bound):
    """The stage-2 torus (R = 1, r = 0.5, edge 0.03, area 2π² ≈ 19.7) and the same torus scaled to
    area 4π (edge 0.038, 10 024 vertices). On a tube of radius r the chordal packet wraps around
    it: the far side is a chord 2r but a geodesic πr away, the place where the cut acts. With
    πr = 1.57 = 7.85σ at σ = 0.2 the 7σ-9σ band sits right there. Measured over the centres
    here (outer equator, top, inner equator) × σ, blended mass; the two (k0, direction) pairs
    agree to 3-4 digits:

    - stage-2 torus, σ = 0.12: removed ≤ 1.4e-10, ‖Δψ‖_M ≤ 5.2e-6;
    - stage-2 torus, σ = 0.2: removed ≤ 9.0e-7, ‖Δψ‖_M ≤ 3.4e-4 (worst on the inner equator;
      1.9e-4 on the outer one), so the change is not below 1e-4 there; over σ ∈ [0.15, 0.25] it
      is 5.4e-5 at σ = 0.15, peaks at σ = 0.18 (4.1e-4) and is 1.9e-5 at σ = 0.25;
    - torus of area 4π, σ = 0.15 (≈ σ = 0.19 on the stage-2 torus): removed ≤ 7.0e-7,
      ‖Δψ‖_M ≤ 4.6e-4.

    The probability removed is about 9e-7 at worst, just under 1e-6 and far below the 1 % data
    error of stage 2.
    """
    verts, faces, geodesics = _torus_geodesics(edge, scale)
    normals, mass = vertex_normals(verts, faces), blended_mass(verts, faces)
    _, v = torus_angles(verts, scale)
    worst = np.zeros(2)
    for target in (0.0, np.pi / 2, np.pi):
        center = int(np.argmin(np.abs(np.angle(np.exp(1j * (v - target))))))
        d = geodesics.distance(center)
        for k0, phi in ((1.0, 0.0), (6.0, 2.0)):
            psi, cut, removed = _cut_and_chordal(verts, normals, mass, center, k0, sigma, phi, d)
            assert abs(mass_norm(cut, mass) - 1) < 1e-12
            worst = np.maximum(worst, [removed, mass_norm(cut - psi, mass)])
    assert worst[0] < removed_bound
    assert worst[1] < change_bound


def test_cut_packet_cleans_the_far_face_of_a_flat_ellipsoid():
    """x² + y² + (z/0.1)² = 1: a packet at the top pole with σ = 0.15 sees the bottom face 0.2
    away in R³ but about 2 away along the surface (half the perimeter of the meridian
    ellipse, 2.030). The chordal packet puts 30 % of its norm on the bottom face; the cut
    removes it and leaves ψᴴMψ = 1."""
    verts, faces = _flat_ellipsoid()
    geodesics = HeatGeodesics(verts, faces)
    top, bottom = int(np.argmax(verts[:, 2])), int(np.argmin(verts[:, 2]))
    d = geodesics.distance(top)
    assert abs(d[bottom] - 2.030) < 0.04                       # measured 2.008
    mass, sigma = blended_mass(verts, faces), 0.15
    psi, cut, removed = _cut_and_chordal(verts, vertex_normals(verts, faces), mass, top, 3.0, sigma, 0.3, d)
    assert abs(mass_norm(cut, mass) - 1) < 1e-12
    lumped = vertex_areas(verts, faces)
    below = verts[:, 2] < 0
    leaked = np.sum(lumped[below] * np.abs(psi[below])**2)
    assert leaked > 0.25                                       # measured 0.30
    assert removed == pytest.approx(leaked, rel=0.05)
    assert np.all(cut[d >= 9 * sigma] == 0)
    assert np.sum(lumped[below] * np.abs(cut[below])**2) < 1e-5   # measured 2e-7, near the rim
    centre_of_bottom = below & (np.hypot(verts[:, 0], verts[:, 1]) < 0.5)
    assert np.abs(psi[centre_of_bottom]).max() > 1 and np.all(cut[centre_of_bottom] == 0)


def test_cut_packet_rejects_a_distance_from_another_vertex(sphere5):
    verts, faces, geodesics = sphere5
    normals, mass = vertex_normals(verts, faces), vertex_areas(verts, faces)
    direction = direction_from_angle(normals[0], 0.0)
    with pytest.raises(ValueError, match="measured from center"):
        cut_gaussian_packet(verts, mass, normals, 0, 2.0, 0.2, direction, geodesics.distance(1))
    with pytest.raises(ValueError, match="one value per vertex"):
        cut_gaussian_packet(verts, mass, normals, 0, 2.0, 0.2, direction, np.zeros(3))
