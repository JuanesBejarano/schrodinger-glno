"""Stage-3 shapes: real spherical harmonics, deformed spheres and tori, relaxation, curvature of the deformed spheres,
refinement, Delaunay flips and quality gates."""
import dataclasses
import functools
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from scipy.special import digamma, sph_harm_y
from schrodinger_glno.mesh import cotan_laplacian, icosphere, laplacian_eigenpairs, torus, vertex_areas
from schrodinger_glno.reference import sphere_quadrature, torus_angles
from schrodinger_glno.shapes import (GENUS, SAMPLERS, TUBE_MODES, SphereDeformation, TorusDeformation, build_shape,
                                     check_gates, deformed_sphere, deformed_torus, delaunay_flip, make_shape,
                                     mesh_quality, normalize_area, real_spherical_harmonics, refine_shape,
                                     sample_sphere_deformation, sample_torus_deformation, smallest_curvature_radius,
                                     sphere_curvatures, sphere_radius, spine_frame, torus_surface, tube_log_radius,
                                     tube_quantities)

ZERO_SPHERE = SphereDeformation((0.0,) * 45)                 # l_max = 6
MESH_OPTIONS = ("subdivisions", "relax", "target_edge")      # the build_shape options recorded in quality


@functools.cache
def _built(deformation):
    """``build_shape(deformation)``, built once for the tests that share it (a relaxed sphere takes 1.3 s)."""
    return build_shape(deformation)


def _flip_random_edges(verts, faces, n_flips, rng, min_weight=0.3):
    """``faces`` with ``n_flips`` random edges flipped, none sharing a face: a valid mesh with known bad edges.

    Only edges with cotangent weight above ``min_weight`` are flipped, so each becomes clearly negative.
    """
    faces = faces.copy()
    L = cotan_laplacian(verts, faces)
    opposite = {}                                            # directed edge -> (face, corner opposite to it)
    for f, tri in enumerate(faces.tolist()):
        for c in range(3):
            opposite[(tri[(c + 1) % 3], tri[(c + 2) % 3])] = (f, c)
    edges = [e for e in opposite if e[0] < e[1] and -L[e[0], e[1]] > min_weight]
    used, done = np.zeros(len(faces), bool), 0
    for p in rng.permutation(len(edges)):
        i, j = edges[p]
        (f1, c1), (f2, c2) = opposite[(i, j)], opposite[(j, i)]
        k, l = faces[f1, c1], faces[f2, c2]
        if used[f1] or used[f2] or (k, l) in opposite:
            continue
        faces[f1], faces[f2] = (k, i, l), (l, j, k)
        opposite[(k, l)] = opposite[(l, k)] = None
        used[f1] = used[f2] = True
        done += 1
        if done == n_flips:
            break
    return faces


def _face_set(faces):
    return np.unique(np.sort(faces, axis=1), axis=0)


def _assert_area_4pi_and_centred(verts, faces):
    mass = vertex_areas(verts, faces)
    assert mass.sum() == pytest.approx(4 * np.pi, rel=1e-13)
    np.testing.assert_allclose(mass @ verts / mass.sum(), 0, atol=1e-14)


def _unnormalised(verts, quality):
    """The points x on the deformation's surface whose normalised image ``scale (x - offset)`` is ``verts``."""
    return verts / quality["scale"] + np.array(quality["offset"])


def _assert_on_sphere_surface(d, x, rtol=1e-12):
    r = np.linalg.norm(x, axis=1)
    np.testing.assert_allclose(r, sphere_radius(d, x / r[:, None]), rtol=rtol, atol=0)


def _strong_sphere(seed):
    """A draw of the default sampler at its largest RMS, τ = 0.15."""
    return sample_sphere_deformation(np.random.default_rng(seed), tau_range=(0.15, 0.15))


def _curvatures_by_differences(d, omega, h=1e-4):
    """Principal curvatures κ1 ≥ κ2 of x = ρ(ω) ω by central differences of x in the chart (a, b) -> ω + a e1 + b e2,
    normalised, as in the planning study; positive on a convex surface. Independent of ``sphere_curvatures``."""
    axis = np.eye(3)[np.argmin(np.abs(omega), axis=1)]
    e1 = axis - np.sum(axis * omega, 1, keepdims=True) * omega
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(omega, e1)

    def x(i, j):
        p = omega + h * i * e1 + h * j * e2
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        return sphere_radius(d, p)[:, None] * p

    xa, xb = (x(1, 0) - x(-1, 0)) / (2 * h), (x(0, 1) - x(0, -1)) / (2 * h)
    xaa, xbb = (x(1, 0) - 2 * x(0, 0) + x(-1, 0)) / h**2, (x(0, 1) - 2 * x(0, 0) + x(0, -1)) / h**2
    xab = (x(1, 1) - x(1, -1) - x(-1, 1) + x(-1, -1)) / (4 * h**2)
    n = np.cross(xa, xb)
    n *= np.sign(np.sum(n * omega, 1, keepdims=True)) / np.linalg.norm(n, axis=1, keepdims=True)     # outward
    E, F, G = np.sum(xa * xa, 1), np.sum(xa * xb, 1), np.sum(xb * xb, 1)
    L, M, N = -np.sum(xaa * n, 1), -np.sum(xab * n, 1), -np.sum(xbb * n, 1)
    K, H = (L * N - M**2) / (E * G - F**2), (E * N - 2 * F * M + G * L) / (2 * (E * G - F**2))
    half_gap = np.sqrt(np.maximum(H**2 - K, 0))
    return H + half_gap, H - half_gap


def test_real_spherical_harmonics_are_orthonormal():
    points, weights = sphere_quadrature(20, 40)              # exact for the degree-12 products
    Y = real_spherical_harmonics(6, points, l_min=0)
    assert Y.shape == (len(points), 49)
    np.testing.assert_allclose(Y.T @ (weights[:, None] * Y), np.eye(49), rtol=0, atol=1e-13)


def test_real_spherical_harmonics_match_their_closed_forms():
    points, _ = sphere_quadrature(6, 12)
    x, y, z = points.T
    Y = real_spherical_harmonics(2, points)                  # ℓ = 2, m = -2..2
    c = np.sqrt(15 / np.pi)
    expected = [c / 2 * x * y, c / 2 * y * z, np.sqrt(5 / np.pi) / 4 * (3 * z**2 - 1), c / 2 * x * z,
                c / 4 * (x**2 - y**2)]
    np.testing.assert_allclose(Y, np.stack(expected, axis=1), rtol=0, atol=1e-14)


def test_real_spherical_harmonics_match_scipy():
    # √2 (-1)^m Im Y_ℓ^|m|, Y_ℓ^0 and √2 (-1)^m Re Y_ℓ^m of sph_harm_y, whose Condon–Shortley phase the (-1)^m undoes;
    # also at the poles and for a direction that is not a unit vector.
    points = np.vstack([sphere_quadrature(6, 12)[0], [[0, 0, 1.0], [0, 0, -1.0], [3.0, -4.0, 1.0]]])
    theta = np.arctan2(np.hypot(points[:, 0], points[:, 1]), points[:, 2])[:, None]
    phi = np.arctan2(points[:, 1], points[:, 0])[:, None]
    ell = np.concatenate([np.full(2 * l + 1, l) for l in range(9)])
    m = np.concatenate([np.arange(-l, l + 1) for l in range(9)])
    Y = sph_harm_y(ell, np.abs(m), theta, phi)
    factor = np.sqrt(2) * (-1.0)**m
    expected = np.where(m < 0, factor * Y.imag, np.where(m > 0, factor * Y.real, Y.real))
    np.testing.assert_allclose(real_spherical_harmonics(8, points, l_min=0), expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(real_spherical_harmonics(8, points, l_min=3), expected[:, 9:], rtol=0, atol=1e-13)


def test_sphere_sampler_draws_the_requested_rms():
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = sample_sphere_deformation(rng, l_max=6, tau_range=(0.1, 0.3))
        assert d.l_max == 6 and len(d.coefficients) == 45 and 0.1 <= d.rms <= 0.3
    d = sample_sphere_deformation(rng, l_max=6, tau_range=(0.2, 0.2))
    assert d.rms == pytest.approx(0.2, rel=1e-12)
    points, weights = sphere_quadrature(20, 40)
    s = np.log(sphere_radius(d, points))
    assert np.sqrt(weights @ s**2 / (4 * np.pi)) == pytest.approx(d.rms, rel=1e-12)
    for _ in range(20):                                      # the defaults: l_max 4, τ in [0.05, 0.15]
        d = sample_sphere_deformation(rng)
        assert d.l_max == 4 and len(d.coefficients) == 21 and 0.05 <= d.rms <= 0.15
    with pytest.raises(ValueError):
        SphereDeformation((0.0,) * 44)


def test_sphere_sampler_follows_the_power_law():
    # Var s_ℓm ∝ ℓ^-slope, so E log(mean_m s_ℓm²) = -slope log ℓ + E log(χ²_n / n) + const with n = 2ℓ + 1;
    # the rescaling to τ adds the same constant to every ℓ of a draw.
    degrees = np.arange(2, 7)
    ell = np.repeat(degrees, 2 * degrees + 1)
    n = 2 * degrees + 1
    for slope in (2.0, 3.0):
        rng = np.random.default_rng(0)
        s = np.array([sample_sphere_deformation(rng, l_max=6, slope=slope).coefficients for _ in range(2000)])
        log_power = np.array([np.log(np.mean(s[:, ell == l]**2, axis=1)).mean() for l in degrees])
        log_power -= digamma(n / 2) + np.log(2 / n)
        assert -np.polyfit(np.log(degrees), log_power, 1)[0] == pytest.approx(slope, abs=0.1)


def test_zero_sphere_deformation_is_the_icosphere_scaled_to_area_4pi():
    omega, faces0 = icosphere(4)
    verts, faces = deformed_sphere(ZERO_SPHERE, 4, relax=False)
    scale = np.sqrt(4 * np.pi / vertex_areas(omega, faces0).sum())
    assert np.array_equal(faces, faces0) and scale > 1
    np.testing.assert_allclose(verts, scale * omega, rtol=0, atol=1e-14)
    _assert_area_4pi_and_centred(verts, faces)


def test_sphere_levels_nest_on_the_same_surface():
    d = sample_sphere_deformation(np.random.default_rng(3), l_max=6, tau_range=(0.1, 0.3))
    coarse, _ = deformed_sphere(d, 3, relax=False)
    fine, _ = deformed_sphere(d, 4, relax=False)
    assert np.array_equal(icosphere(4)[0][:len(coarse)], icosphere(3)[0])
    # Same points of the same surface, up to each level's normalisation (a scale and a shift).
    x, y = fine[:len(coarse)] - fine[:len(coarse)].mean(0), coarse - coarse.mean(0)
    ratio = np.sum(x * y) / np.sum(y * y)
    np.testing.assert_allclose(x, ratio * y, rtol=0, atol=1e-13)
    assert abs(ratio - 1) < 0.01                             # 0.9936: the coarse mesh loses more area


def test_relaxed_sphere_stays_on_the_surface_with_the_icosphere_faces():
    d = _strong_sphere(0)
    verts, faces, quality = _built(d)
    assert quality["flips"] == 0 and np.array_equal(faces, icosphere(5)[1])     # deformed_sphere(d), bit for bit
    _assert_area_4pi_and_centred(verts, faces)
    x = _unnormalised(verts, quality)
    _assert_on_sphere_surface(d, x)                          # |x| = ρ(x/|x|), 4e-16 relative
    omega = icosphere(5)[0]
    moved = np.arccos(np.clip(np.sum(x / np.linalg.norm(x, axis=1, keepdims=True) * omega, 1), -1, 1))
    assert moved.max() > 2 * 0.0378                          # 0.11 rad: up to 3 edges of the unit icosphere(5)
    # The round sphere is relaxed too: still on the sphere, its edges more even (p95/mean 1.09 -> 1.04).
    verts, faces, quality = _built(ZERO_SPHERE)
    np.testing.assert_allclose(np.linalg.norm(_unnormalised(verts, quality), axis=1), 1, rtol=0, atol=1e-12)
    assert quality["edge_p95_ratio"] < mesh_quality(*deformed_sphere(ZERO_SPHERE, relax=False))["edge_p95_ratio"]
    assert check_gates(quality, 0) == (True, [])


@pytest.mark.parametrize("seed", [0, 2, 3])
def test_relaxation_brings_strong_deformations_within_the_gates(seed):
    # Radially moved icosphere vertices keep the sphere's density, so the edges stretch where ρ is large.
    d = _strong_sphere(seed)
    raw = build_shape(d, relax=False)[2]
    passed, reasons = check_gates(raw, 0)
    assert not passed and [r.split()[0] for r in reasons] == ["p95/mean"]          # 1.33-1.36
    verts, faces, quality = _built(d)
    assert check_gates(quality, 0) == (True, [])
    assert quality["edge_p95_ratio"] < 1.25 and quality["min_angle"] > 43 and quality["flips"] == 0
    _assert_area_4pi_and_centred(verts, faces)


def test_relaxation_keeps_the_spectrum():
    # Both meshes discretise the same surface: the first 20 eigenvalues agree to the discretisation error
    # (5e-4), while the shape is far from the round sphere (17 %).
    d = _strong_sphere(0)

    def eigenvalues(verts, faces):
        return laplacian_eigenpairs(cotan_laplacian(verts, faces), vertex_areas(verts, faces), 21)[0][1:]

    raw, relaxed = eigenvalues(*deformed_sphere(d, relax=False)), eigenvalues(*_built(d)[:2])
    ell = np.repeat(np.arange(1, 5), 2 * np.arange(1, 5) + 1)[:20]
    assert np.abs(relaxed / raw - 1).max() < 3e-3
    assert np.abs(raw / (ell * (ell + 1)) - 1).max() > 0.1


def _random_directions(n, seed=0):
    omega = np.random.default_rng(seed).standard_normal((n, 3))
    return np.vstack([omega / np.linalg.norm(omega, axis=1, keepdims=True), [[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0]]])


def test_curvatures_of_the_round_sphere():
    omega = _random_directions(500)
    for k in sphere_curvatures(ZERO_SPHERE, 3 * omega):         # directions need not be unit vectors
        np.testing.assert_allclose(k, 1, rtol=0, atol=1e-14)
    assert smallest_curvature_radius(ZERO_SPHERE) == pytest.approx(1, rel=1e-14)
    assert smallest_curvature_radius(ZERO_SPHERE, scale=2.5) == pytest.approx(2.5, rel=1e-14)


@pytest.mark.parametrize("c", [0.4, -0.4])
def test_curvatures_of_a_zonal_deformation(c):
    # s = c Y_20(θ): a surface of revolution with r(θ) = e^s. Its meridian is the polar curve r(θ), of curvature
    # (r² + 2r'² - r r'')/(r² + r'²)^{3/2}; the parallels give (r sin θ - r' cos θ)/(r sin θ √(r² + r'²)). The largest
    # |κ| is at the poles (an umbilic) for c > 0 and on the equator for c < 0.
    s = np.zeros(5)
    s[2] = c
    d = SphereDeformation(tuple(s))
    theta = np.linspace(0, np.pi, 200001)
    A = c * np.sqrt(5 / np.pi) / 4
    r = np.exp(A * (3 * np.cos(theta)**2 - 1))
    r1, r2 = -3 * A * np.sin(2 * theta) * r, (-6 * A * np.cos(2 * theta) + (3 * A * np.sin(2 * theta))**2) * r
    meridian = (r**2 + 2 * r1**2 - r * r2) / (r**2 + r1**2)**1.5
    with np.errstate(invalid="ignore"):
        parallel = (r * np.sin(theta) - r1 * np.cos(theta)) / (r * np.sin(theta) * np.sqrt(r**2 + r1**2))
    parallel[[0, -1]] = meridian[[0, -1]]                    # the poles are umbilics
    omega = np.stack([np.sin(theta) * np.cos(0.3), np.sin(theta) * np.sin(0.3), np.cos(theta)], 1)
    k1, k2 = sphere_curvatures(d, omega)
    np.testing.assert_allclose(k1, np.maximum(meridian, parallel), rtol=1e-11, atol=0)
    np.testing.assert_allclose(k2, np.minimum(meridian, parallel), rtol=1e-11, atol=0)
    largest = np.maximum(np.abs(meridian), np.abs(parallel)).max()
    assert smallest_curvature_radius(d) == pytest.approx(1 / largest, rel=1e-9)
    # The same from finite differences on the surface itself, along the meridian.
    f1, f2 = _curvatures_by_differences(d, omega[::100])
    np.testing.assert_allclose(np.stack([f1, f2]), np.stack([k1, k2])[:, ::100], rtol=0, atol=1e-6)
    assert smallest_curvature_radius(d) == pytest.approx(1 / np.maximum(np.abs(f1), np.abs(f2)).max(), rel=1e-6)


@pytest.mark.parametrize("l_max, tau", [(4, 0.15), (6, 0.3), (8, 0.2)])
def test_curvatures_match_finite_differences_on_the_surface(l_max, tau):
    # Strongly curved draws too (l_max 6, τ 0.3: |κ| up to 15, saddles).
    d = sample_sphere_deformation(np.random.default_rng(l_max), l_max=l_max, tau_range=(tau, tau))
    omega = _random_directions(1000, seed=l_max)
    k = np.stack(sphere_curvatures(d, omega))
    assert np.all(k[0] >= k[1]) and k.min() < 0 < k.max()
    np.testing.assert_allclose(np.stack(_curvatures_by_differences(d, omega)), k, rtol=0, atol=1e-6 * np.abs(k).max())


@pytest.mark.parametrize("seed, l_max, tau", [(52, 4, None), (3, 6, 0.3), (38, 8, 0.3)])
def test_smallest_curvature_radius_finds_the_largest_curvature(seed, l_max, tau):
    # The search from icosphere(6) is never below a finer grid, and that grid comes within 1e-3 of it; icosphere(6)
    # alone is up to 1 % low. Seed 52 of the default sampler has a radius just under the gate (0.175). Seed 38 (ℓ 8,
    # τ 0.3) needs the many starts and the fine grid: from the grid's largest point alone the search ends at a maximum
    # 0.33 % lower, and from icosphere(3) 10 % lower.
    tau_range = (0.05, 0.15) if tau is None else (tau, tau)
    d = sample_sphere_deformation(np.random.default_rng(seed), l_max=l_max, tau_range=tau_range)
    radius = smallest_curvature_radius(d)
    largest = max(np.abs(k).max() for k in sphere_curvatures(d, icosphere(7)[0]))
    assert 1 - 1e-3 < radius * largest <= 1
    assert smallest_curvature_radius(d, subdivisions=5) == pytest.approx(radius, rel=1e-8)     # the same maximum
    assert smallest_curvature_radius(d, scale=0.97) == pytest.approx(0.97 * radius, rel=1e-15)


def test_normalize_area_centres_and_scales():
    verts, faces = torus(0.1, 1.0, 0.5)
    moved = 3 * verts + np.array([1.0, -2.0, 0.5])
    out, scale = normalize_area(moved, faces)
    _assert_area_4pi_and_centred(out, faces)
    assert scale == pytest.approx(np.sqrt(4 * np.pi / vertex_areas(moved, faces).sum()), rel=1e-14)
    out, _ = normalize_area(moved, faces, area=2.0)
    assert vertex_areas(out, faces).sum() == pytest.approx(2.0, rel=1e-13)


def test_deformed_shapes_have_area_4pi_and_centroid_at_origin():
    rng = np.random.default_rng(5)
    _assert_area_4pi_and_centred(*deformed_sphere(sample_sphere_deformation(rng), 4))
    verts, faces, _ = deformed_torus(sample_torus_deformation(rng), 0.06)
    _assert_area_4pi_and_centred(verts, faces)


def test_torus_sampler_draws_within_its_ranges():
    rng = np.random.default_rng(1)
    for _ in range(20):
        d = sample_torus_deformation(rng)
        assert 0.45 <= d.alpha <= 0.55 and d.major_radius * d.minor_radius == pytest.approx(1 / np.pi)
        assert max(map(abs, d.a)) <= 0.04 and max(map(abs, d.b)) <= 0.06 and 0 <= d.tube_rms <= 0.075
        assert len(d.c) == len(d.theta) == 17
    d = sample_torus_deformation(rng, tau_range=(0.06, 0.06))
    assert d.tube_rms == pytest.approx(0.06, rel=1e-12)
    # |x(u, v) - c(u)| = r(u, v); the 16 × 8 grid integrates t² (|j| ≤ 6, |k| ≤ 4) exactly.
    u, v = np.meshgrid(2 * np.pi * np.arange(16) / 16, 2 * np.pi * np.arange(8) / 8)
    t = np.log(np.linalg.norm(torus_surface(d, u, v) - spine_frame(d, u)[0], axis=-1) / d.minor_radius)
    assert np.sqrt(np.mean(t**2)) == pytest.approx(d.tube_rms, rel=1e-12)


def test_tube_modes_wind_both_ways():
    modes = set(TUBE_MODES)
    assert len(TUBE_MODES) == len(modes) == 17 and (0, 0) not in modes
    assert all((-j, -k) not in modes for j, k in modes)     # one of each pair ±(j, k): the modes are orthogonal
    assert all((j, -k) in modes for j, k in modes if j > 0)  # each helix (j, k ≠ 0) with its mirror image


def test_deformations_follow_their_closed_forms():
    # A stored deformation must rebuild the same surface: the order of the modes and the convention of each term
    # are pinned (relabelling the modes onto themselves, e.g. k -> -k, passes every other test).
    assert TUBE_MODES == ((0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (2, -2), (2, -1), (2, 0), (2, 1),
                          (2, 2), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2))
    c, theta = np.zeros(17), np.zeros(17)
    c[[1, 3]], theta[[1, 3]] = (-0.04, 0.05), (1.1, 0.3)    # modes (0, 2) and (1, -1)
    d = TorusDeformation(0.5, a=(0.03, -0.02), phi=(0.3, 1.1), b=(0.05, -0.04), psi=(0.7, 2.0), c=tuple(c),
                         theta=tuple(theta))
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 13), np.linspace(0, 2 * np.pi, 7))
    np.testing.assert_allclose(tube_log_radius(d, u, v), -0.04 * np.cos(2 * v + 1.1) + 0.05 * np.cos(u - v + 0.3),
                               rtol=0, atol=1e-15)
    R = d.major_radius * (1 + 0.03 * np.cos(2 * u + 0.3) - 0.02 * np.cos(3 * u + 1.1))
    z = d.major_radius * (0.05 * np.sin(2 * u + 0.7) - 0.04 * np.sin(3 * u + 2.0))
    np.testing.assert_allclose(spine_frame(d, u)[0], np.stack([R * np.cos(u), R * np.sin(u), z], -1),
                               rtol=0, atol=1e-14)
    # Sphere: s = 0.3 Y_20, the third coefficient, with Y_20 = √(5/π)/4 (3z² - 1); ρ = e^s.
    points, _ = sphere_quadrature(6, 12)
    s = np.zeros(21)
    s[2] = 0.3
    np.testing.assert_allclose(sphere_radius(SphereDeformation(tuple(s)), points),
                               np.exp(0.3 * np.sqrt(5 / np.pi) / 4 * (3 * points[:, 2]**2 - 1)), rtol=1e-14, atol=0)


def test_mirror_image_of_a_deformed_torus_is_in_the_family():
    # Reflecting z -> -z turns x(u, v) into x(u, -v) of the torus with b -> -b and t(u, v) -> t(u, -v), whose modes
    # are (j, -k) with the same phase, or (0, k) with the opposite phase. With k >= 0 only, helices had one hand.
    d = sample_torus_deformation(np.random.default_rng(6), tau_range=(0.075, 0.075))
    index = {mode: n for n, mode in enumerate(TUBE_MODES)}
    mirror = [index[(j, -k)] if j else n for n, (j, k) in enumerate(TUBE_MODES)]
    mirrored = dataclasses.replace(d, b=tuple(-b for b in d.b), c=tuple(d.c[n] for n in mirror),
                                   theta=tuple(d.theta[n] if TUBE_MODES[n][0] else -d.theta[n] for n in mirror))
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 41), np.linspace(0, 2 * np.pi, 23))
    np.testing.assert_allclose(torus_surface(mirrored, u, v), torus_surface(d, u, -v) * [1, 1, -1], rtol=0, atol=1e-12)
    assert np.abs(torus_surface(mirrored, u, v) - torus_surface(d, u, v)).max() > 0.05     # a different torus


@pytest.mark.parametrize("alpha, target_edge", [(0.45, 0.06), (0.5, 0.037), (0.6, 0.06)])
def test_zero_torus_deformation_reproduces_the_torus(alpha, target_edge):
    d = TorusDeformation(alpha)
    R0, r0 = d.major_radius, d.minor_radius
    assert R0 * r0 == pytest.approx(1 / np.pi) and r0 / R0 == pytest.approx(alpha)
    verts, faces = torus(target_edge, R0, r0)
    u, v = torus_angles(verts, R0)
    np.testing.assert_allclose(torus_surface(d, u, v), verts, rtol=0, atol=1e-12)
    # The twist compensation runs and is zero: the frame is the round torus's.
    _, _, e1, e2 = spine_frame(d, u)
    np.testing.assert_allclose(e1, np.stack([np.cos(u), np.sin(u), 0 * u], 1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(e2, np.tile([0.0, 0.0, 1.0], (len(u), 1)), rtol=0, atol=1e-15)
    out, out_faces, scale = deformed_torus(d, target_edge)
    expected, expected_scale = normalize_area(verts, faces)
    assert np.array_equal(out_faces, faces) and scale == pytest.approx(expected_scale, rel=1e-12)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)


def test_twist_compensated_frame_is_orthonormal_periodic_and_untwisted():
    d = TorusDeformation(0.5, a=(0.03, -0.02), phi=(0.3, 1.1), b=(0.06, -0.05), psi=(0.7, 2.0))
    u, h = np.linspace(0, 2 * np.pi, 2001), 1e-5
    c, T, e1, e2 = spine_frame(d, u)
    for x, y, value in [(T, T, 1), (e1, e1, 1), (e2, e2, 1), (T, e1, 0), (T, e2, 0), (e1, e2, 0)]:
        np.testing.assert_allclose(np.sum(x * y, -1), value, rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.cross(e1, e2), -T, rtol=0, atol=1e-13)         # same handedness as the round torus
    for x in (c, T, e1, e2):
        np.testing.assert_allclose(x[-1], x[0], rtol=0, atol=1e-12)              # closes at u = 2π
    _, T_plus, e1_plus, _ = spine_frame(d, u + h)
    _, T_minus, e1_minus, _ = spine_frame(d, u - h)
    rate = np.sum((e1_plus - e1_minus) / (2 * h) * e2, -1)

    def vertical(T):                                         # the uncompensated frame (e1⁰, e2⁰)
        e2 = np.array([0.0, 0.0, 1.0]) - T[:, 2:] * T
        e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
        return np.cross(T, e2), e2

    rate0 = np.sum((vertical(T_plus)[0] - vertical(T_minus)[0]) / (2 * h) * vertical(T)[1], -1)
    assert np.ptp(rate0) > 0.4                               # the vertical frame twists back and forth (0.50)
    assert np.ptp(rate) < 1e-8                               # compensated: a constant rate (5e-10) ...
    assert rate.mean() == pytest.approx(rate0[:-1].mean(), abs=1e-8)    # ... equal to the mean twist ω̄


def test_tube_quantities_of_round_tori():
    d = TorusDeformation(0.45)
    R0, r0 = d.major_radius, d.minor_radius
    q = tube_quantities(d)
    assert q["r_min"] == pytest.approx(r0, rel=1e-12) and q["fold"] == pytest.approx(0.45, rel=1e-12)
    assert q["hole"] == pytest.approx(2 * (R0 - r0), rel=1e-12)
    for sigma in (0.15, 0.2):
        du = 7 * sigma / (R0 - r0)                           # inner side: |c'| (1 - κ r) = R0 - r0
        assert q["gap"][str(sigma)] == pytest.approx(2 * R0 * np.sin(du / 2) - 2 * r0, abs=1e-3)
    scaled = tube_quantities(d, scale=2.0, sigmas=(0.3,))
    assert scaled["r_min"] == pytest.approx(2 * r0) and scaled["fold"] == q["fold"]
    assert scaled["hole"] == pytest.approx(2 * q["hole"], rel=1e-12)
    assert scaled["gap"]["0.3"] == pytest.approx(2 * q["gap"]["0.15"], rel=1e-12)
    assert tube_quantities(TorusDeformation(0.6))["gap"] == {"0.15": None, "0.2": None}   # no pair 7σ apart


def test_hole_is_the_clearance_of_the_bounding_balls():
    # hole = 2 min_u (|c_xy(u)| - r_max(u)) on the grid of tube_quantities, with r = |x - c| from the surface itself;
    # a lower bound of the surface's own clearance from the z axis.
    d = sample_torus_deformation(np.random.default_rng(4), tau_range=(0.075, 0.075))
    u = 2 * np.pi * np.arange(1024) / 1024
    v = 2 * np.pi * np.arange(128) / 128
    c = spine_frame(d, u)[0]
    x = torus_surface(d, u[:, None], v[None, :])
    r_max = np.linalg.norm(x - c[:, None], axis=-1).max(1)
    hole = tube_quantities(d, scale=1.5)["hole"]
    assert hole == pytest.approx(1.5 * 2 * np.min(np.hypot(c[:, 0], c[:, 1]) - r_max), rel=1e-12)
    assert hole < 1.5 * 2 * np.hypot(x[..., 0], x[..., 1]).min()


def test_fold_bounds_the_folding_of_a_non_planar_torus():
    # x_u · T = |c'| (1 - κ r (cos v e1 + sin v e2) · N), so 1 - x_u · T / |c'| <= κ r_max <= fold, with
    # equality at the grid's best v for a round tube. κ and x_u by central differences.
    round_tube = TorusDeformation(0.55, a=(0.04, -0.03), phi=(0.3, 1.1), b=(0.06, -0.05), psi=(0.7, 2.0))
    rng = np.random.default_rng(4)
    bumpy_tube = dataclasses.replace(round_tube, c=tuple(0.03 * rng.standard_normal(len(TUBE_MODES))),
                                     theta=tuple(rng.uniform(0, 2 * np.pi, len(TUBE_MODES))))
    u = 2 * np.pi * np.arange(1024) / 1024                   # the grid of tube_quantities
    v = 2 * np.pi * np.arange(128) / 128
    c, T, _, _ = spine_frame(round_tube, u)                  # the same spine for both tubes
    h = 1e-4
    c_plus, c_minus = spine_frame(round_tube, u + h)[0], spine_frame(round_tube, u - h)[0]
    c1, c2 = (c_plus - c_minus) / (2 * h), (c_plus - 2 * c + c_minus) / h**2
    kappa = np.linalg.norm(np.cross(c1, c2), axis=1) / np.linalg.norm(c1, axis=1)**3
    assert tube_quantities(round_tube)["fold"] == pytest.approx(round_tube.minor_radius * kappa.max(), rel=1e-6)
    speed, h = np.linalg.norm(c1, axis=1)[:, None], 1e-5
    for d in (round_tube, bumpy_tube):
        x_u = (torus_surface(d, u[:, None] + h, v) - torus_surface(d, u[:, None] - h, v)) / (2 * h)
        shrink = np.max(1 - np.sum(x_u * T[:, None], -1) / speed)
        fold = tube_quantities(d)["fold"]
        assert shrink <= fold + 1e-8
        if d is round_tube:
            assert shrink >= fold * np.cos(np.pi / 128) - 1e-8


def test_delaunay_flip_restores_a_torus_with_flipped_edges():
    verts, faces = torus(0.1, 1.0, 0.5)
    bad = _flip_random_edges(verts, faces, 200, np.random.default_rng(0))
    before = mesh_quality(verts, bad)
    assert before["n_negative"] == 200 and before["closed"] and before["oriented"]
    new, flips = delaunay_flip(verts, bad)
    L = cotan_laplacian(verts, new).tocoo()
    off = L.data[L.row != L.col]
    assert off.max() <= 1e-12 * np.abs(off).max()            # no negative weight left
    after = mesh_quality(verts, new)
    assert after["closed"] and after["oriented"] and after["components"] == 1 and after["euler"] == 0
    assert np.array_equal(np.unique(new), np.arange(len(verts)))
    assert flips == 200 and np.array_equal(_face_set(new), _face_set(faces))     # the original mesh is back
    assert vertex_areas(verts, new).sum() == pytest.approx(vertex_areas(verts, faces).sum(), rel=1e-14)


def test_delaunay_flip_repairs_a_jittered_sphere():
    verts, faces = icosphere(3)
    verts = verts + 0.027 * np.random.default_rng(0).standard_normal(verts.shape)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    before = mesh_quality(verts, faces)
    assert before["n_negative"] > 50
    new, flips = delaunay_flip(verts, faces)
    after = mesh_quality(verts, new)
    assert after["n_negative"] == 0 and flips >= before["n_negative"]
    assert after["closed"] and after["oriented"] and after["components"] == 1 and after["euler"] == 2
    assert np.sum(np.any(new != faces, axis=1)) <= 2 * flips
    assert abs(after["area"] / before["area"] - 1) < 1e-4   # extrinsic flips change the area slightly (1e-5)


def test_delaunay_flip_leaves_a_delaunay_mesh_alone_and_needs_a_closed_one():
    verts, faces = icosphere(3)
    new, flips = delaunay_flip(verts, faces)
    assert flips == 0 and np.array_equal(new, faces)
    with pytest.raises(ValueError):
        delaunay_flip(verts, faces[:-1])


def test_mesh_quality_reads_genus_and_components_from_the_mesh():
    sphere, faces = icosphere(5)
    q = mesh_quality(normalize_area(sphere, faces)[0], faces)
    assert q["closed"] and q["oriented"] and q["components"] == 1 and q["euler"] == 2 and q["genus"] == 0
    assert 54 < q["min_angle"] and q["max_angle"] < 72.1 and q["n_negative"] == 0 and q["min_weight"] > 0.3
    assert q["k_allowed"] == pytest.approx(16.63, abs=0.01) and q["area"] == pytest.approx(4 * np.pi)
    assert q["n_verts"] == 10242 and q["n_edges"] == 30720 and q["edge_p95_ratio"] < q["edge_max_ratio"] < 1.3
    q = mesh_quality(*torus(0.1, 1.0, 0.5))
    assert q["closed"] and q["components"] == 1 and q["euler"] == 0 and q["genus"] == 1
    small, small_faces = icosphere(2)
    q = mesh_quality(np.vstack([small, small + 3]), np.vstack([small_faces, small_faces + len(small)]))
    assert q["components"] == 2 and q["euler"] == 4
    q = mesh_quality(small, small_faces[:-1])
    assert not q["closed"] and q["euler"] == 1


def test_gates_accept_the_round_sphere_and_torus():
    assert check_gates(_built(ZERO_SPHERE)[2], 0) == (True, [])
    verts, faces, quality = build_shape(TorusDeformation())
    assert quality["flips"] == 0 and check_gates(quality, 1, quality) == (True, [])
    _assert_area_4pi_and_centred(verts, faces)


def test_gates_reject_bad_meshes():
    verts, faces = torus(0.037, 1.0, 0.5)
    verts, _ = normalize_area(verts, faces)
    good = mesh_quality(verts, faces)
    tube = {"r_min": 0.4, "fold": 0.5, "hole": 0.7, "gap": {"0.15": 1.0, "0.2": 1.0}}
    assert check_gates(good, 1, tube) == (True, [])
    assert check_gates(good, 1, tube | {"gap": {"0.15": None, "0.2": None}}) == (True, [])   # no pair 7σ apart
    flipped = mesh_quality(verts, _flip_random_edges(verts, faces, 50, np.random.default_rng(1)))
    passed, reasons = check_gates(flipped, 1)
    assert not passed and [r.split()[:2] for r in reasons] == [["min", "angle"], ["max", "angle"], ["50", "negative"]]
    coarse, coarse_faces = icosphere(2)
    coarse, _ = normalize_area(coarse, coarse_faces)
    small, small_faces = icosphere(3)
    two = mesh_quality(np.vstack([small, small + 3]), np.vstack([small_faces, small_faces + len(small)]))
    # Each value just past its threshold fails that gate alone.
    for change, reason in [({"min_angle": 29.9}, "min angle 29.9° < 30.0°"),
                           ({"max_angle": 120.5}, "max angle 120.5° > 120.0°"),
                           ({"edge_p95_ratio": 1.31}, "p95/mean edge 1.310 > 1.3"),
                           ({"n_negative": 1}, "1 negative cotangent weights"),
                           ({"k_allowed": 15.9}, "allowed wavenumber 15.90 < 16.0")]:
        assert check_gates(good | change, 1, tube) == (False, [reason])
    for change, reason in [({"r_min": 0.29}, "tube radius 0.290 < 0.3"),
                           ({"fold": 0.8}, "fold 0.800 >= 0.8"),
                           ({"hole": 0.57}, "hole 0.570 < 0.58"),
                           ({"gap": {"0.15": 1.0, "0.2": 0.5}}, "gap 0.500 < 4.0σ at σ = 0.2")]:
        assert check_gates(good, 1, tube | change) == (False, [reason])
    assert check_gates(good | {"k_allowed": 16.2}, 1, tube, min_wavenumber=16.33) == (
        False, ["allowed wavenumber 16.20 < 16.33"])                                 # k0 = 8 at σ = 0.18
    cases = [(good, 0, None, "genus 1.0, expected 0"),
             (mesh_quality(coarse, coarse_faces), 0, None, "allowed wavenumber"),
             (two, 0, None, "2 components"),
             (mesh_quality(small, small_faces[:-1]), 0, None, "not a closed, oriented manifold"),
             (good, 1, tube_quantities(TorusDeformation(0.2)), "tube radius"),
             (good, 1, tube_quantities(TorusDeformation(0.85)), "fold"),
             (good, 1, tube_quantities(TorusDeformation(0.62)), "hole 0.545 < 0.58")]    # 2 (R0 - r0)
    for quality, genus, tube_q, reason in cases:
        passed, reasons = check_gates(quality, genus, tube_q)
        assert not passed and any(reason in r for r in reasons), reasons
    assert check_gates(good, 1, tube_quantities(TorusDeformation(0.6))) == (True, [])   # hole 0.583, just inside


def test_gates_bound_the_radius_of_curvature_of_a_sphere():
    # build_shape records the radius of a sphere in the units of its mesh: the round sphere's is its scale, ≈ 1.
    quality = _built(ZERO_SPHERE)[2]
    assert quality["min_curvature_radius"] == pytest.approx(quality["scale"], rel=1e-14)
    assert abs(quality["scale"] - 1) < 1e-3
    d = _strong_sphere(0)
    strong = _built(d)[2]
    assert strong["min_curvature_radius"] == smallest_curvature_radius(d, strong["scale"])
    assert json.loads(json.dumps(strong, allow_nan=False)) == strong and type(strong["min_curvature_radius"]) is float
    assert check_gates(quality | {"min_curvature_radius": 0.18}, 0) == (True, [])          # at the threshold
    assert check_gates(quality | {"min_curvature_radius": 0.179}, 0) == (False, ["curvature radius 0.179 < 0.18"])
    assert check_gates(quality | {"min_curvature_radius": 0.19}, 0, min_curvature_radius=0.2) == (
        False, ["curvature radius 0.190 < 0.2"])
    # Applied only where recorded: a torus has its fold gate instead.
    torus_quality = build_shape(TorusDeformation(), target_edge=0.06)[2]                  # coarse: k_allowed 10.25
    assert "min_curvature_radius" not in torus_quality
    assert check_gates(torus_quality, 1, torus_quality, min_wavenumber=10.0, min_curvature_radius=10.0) == (True, [])


def test_make_shape_redraws_a_sphere_curved_too_tightly():
    # Seed 52 of the default sampler (τ = 0.136) passes every mesh gate with a radius of curvature of 0.175; 7 in
    # 6000 default draws fall below 0.18, all with τ > 0.13 (1.7 % at τ = 0.15).
    shape = make_shape("sphere", np.random.default_rng(52))
    assert shape.attempts == 2 and shape.rejections == (("curvature radius 0.175 < 0.18",),)
    assert shape.quality["min_curvature_radius"] >= 0.18 and check_gates(shape.quality, 0) == (True, [])


def test_build_shape_records_the_normalisation_scale_and_offset():
    d = TorusDeformation(0.463)                              # a revolution torus: radii R0 · scale and r0 · scale
    R0, r0 = d.major_radius, d.minor_radius
    verts, _, quality = build_shape(d)
    scale = quality["scale"]
    assert quality["target_edge"] == 0.037 and "relax" not in quality      # the mesh options, by family
    round_verts, round_faces = torus(0.037, R0, r0)
    assert quality["flips"] == 0 and scale == pytest.approx(normalize_area(round_verts, round_faces)[1], rel=1e-13)
    np.testing.assert_allclose(_unnormalised(verts, quality), round_verts, rtol=0, atol=1e-14)
    assert quality["r_min"] == pytest.approx(r0 * scale, rel=1e-12)
    assert quality["hole"] == pytest.approx(2 * (R0 - r0) * scale, rel=1e-12)
    # Scale and offset include the renormalisation after the flips: verts = scale (x - offset) still.
    d = sample_torus_deformation(np.random.default_rng(1))
    verts, _, quality = build_shape(d)
    u, v = torus_angles(torus(0.037, d.major_radius, d.minor_radius)[0], d.major_radius)
    assert quality["flips"] > 0 and len(quality["offset"]) == 3
    np.testing.assert_allclose(_unnormalised(verts, quality), torus_surface(d, u, v), rtol=0, atol=1e-14)
    d = _strong_sphere(0)
    verts, _, quality = build_shape(d, relax=False)
    omega = icosphere(5)[0]
    assert quality["flips"] > 0 and quality["relax"] is False and quality["subdivisions"] == 5
    assert "target_edge" not in quality
    np.testing.assert_allclose(_unnormalised(verts, quality), sphere_radius(d, omega)[:, None] * omega,
                               rtol=0, atol=1e-14)
    # So is the radius of curvature, in the units of the mesh after the 30 flips (the scale before them is 3e-9 off).
    assert quality["min_curvature_radius"] == smallest_curvature_radius(d, quality["scale"])


@pytest.mark.parametrize("family, seed", [("sphere", 7), ("torus", 1)])
def test_typical_sampled_shapes_pass_the_gates(family, seed):
    verts, faces, quality = _built(SAMPLERS[family](np.random.default_rng(seed)))
    passed, reasons = check_gates(quality, GENUS[family], quality if family == "torus" else None)
    assert passed, reasons
    assert quality["n_negative"] == 0
    assert quality["flips"] > 0 or family == "sphere"        # the torus needed a few flips, the relaxed sphere none
    _assert_area_4pi_and_centred(verts, faces)


@pytest.mark.parametrize("family, seed", [("sphere", 7), ("torus", 1)])
def test_make_shape_is_reproducible_from_its_manifest_record(family, seed):
    shape = make_shape(family, np.random.default_rng(seed))
    assert shape.attempts == 1 and shape.family == family
    assert shape.deformation == SAMPLERS[family](np.random.default_rng(seed))      # the same rng state, the same draw
    # The record goes through strict JSON (no inf, keys are strings) unchanged and gets the same verdict. It holds the
    # mesh options, with which it rebuilds the shape bit for bit and refines it on the same surface.
    record = json.loads(json.dumps({"deformation": dataclasses.asdict(shape.deformation), "quality": shape.quality},
                                   allow_nan=False))
    deformation, quality = type(shape.deformation)(**record["deformation"]), record["quality"]
    assert deformation == shape.deformation and quality == shape.quality
    assert ("min_curvature_radius" in quality) == (family == "sphere")
    assert check_gates(quality, GENUS[family], quality if family == "torus" else None) == (True, [])
    verts, faces, rebuilt = build_shape(deformation, **{key: quality[key] for key in MESH_OPTIONS if key in quality})
    assert np.array_equal(verts, shape.verts) and np.array_equal(faces, shape.faces) and rebuilt == shape.quality
    fine, _ = refine_shape(deformation, verts, faces, quality)                     # target_edge from the record
    assert np.array_equal(fine[:len(verts)], verts)


def test_shapes_do_not_depend_on_the_number_of_blas_threads():
    # The relaxed sphere (350 projections onto the surface), a torus, their refined meshes and the quality the gates
    # read, the sphere's radius of curvature included. The defaults keep |s| < 0.5, where a last-bit change of s
    # rounds away in e^s: with s = Y @ coefficients (BLAS) only the raw sphere with l_max = 6 and RMS 0.3 differed
    # between 1 and 4 threads. Both processes run at once.
    script = ("import hashlib, json, numpy as np\n"
              "from schrodinger_glno.shapes import (build_shape, deformed_sphere, refine_shape,\n"
              "                                     sample_sphere_deformation, sample_torus_deformation,\n"
              "                                     smallest_curvature_radius)\n"
              "h = hashlib.sha256()\n"
              "for sample, seed in [(sample_sphere_deformation, 11), (sample_torus_deformation, 1)]:\n"
              "    d = sample(np.random.default_rng(seed))\n"
              "    verts, faces, quality = build_shape(d)\n"
              "    fine, fine_faces = refine_shape(d, verts, faces, quality)\n"
              "    h.update(verts.tobytes() + faces.tobytes() + fine.tobytes() + fine_faces.tobytes())\n"
              "    h.update(json.dumps(quality, sort_keys=True).encode())\n"
              "d = sample_sphere_deformation(np.random.default_rng(11), l_max=6, tau_range=(0.3, 0.3))\n"
              "h.update(deformed_sphere(d, relax=False)[0].tobytes() + repr(smallest_curvature_radius(d)).encode())\n"
              "print(h.hexdigest())\n")
    runs = [subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env={**os.environ, "OMP_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "MKL_NUM_THREADS": n})
            for n in ("1", "4")]
    outputs = [run.communicate() for run in runs]
    assert all(run.returncode == 0 for run in runs), [err for _, err in outputs]
    assert len(outputs[0][0]) == 65 and outputs[0][0] == outputs[1][0]


def _edges(faces):
    """Every edge once, sorted: the order in which ``refine_shape`` appends their midpoints."""
    return np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)


def test_refine_shape_continues_the_unrelaxed_sphere_levels():
    # Refining the radially moved icosphere(4) (no flips) gives the radially moved icosphere(5) under the same
    # normalisation: the same faces and points, with the coarse vertices, exactly, first.
    d = sample_sphere_deformation(np.random.default_rng(3), tau_range=(0.05, 0.05))
    verts, faces, quality = build_shape(d, subdivisions=4, relax=False)
    assert quality["flips"] == 0
    fine, fine_faces = refine_shape(d, verts, faces, quality, flip=False)
    omega, expected_faces = icosphere(5)
    assert np.array_equal(fine_faces, expected_faces) and np.array_equal(fine[:len(verts)], verts)
    np.testing.assert_allclose(fine, quality["scale"] * (sphere_radius(d, omega)[:, None] * omega - quality["offset"]),
                               rtol=0, atol=1e-14)


def test_refine_shape_of_a_relaxed_sphere_nests_on_the_surface():
    d = _strong_sphere(2)
    verts, faces, quality = build_shape(d, subdivisions=4)
    fine, fine_faces = refine_shape(d, verts, faces, quality)
    assert np.array_equal(fine[:len(verts)], verts) and len(fine) == 4 * len(verts) - 6
    _assert_on_sphere_surface(d, _unnormalised(fine, quality))
    coarse_q, q = mesh_quality(verts, faces), mesh_quality(fine, fine_faces)
    assert q["closed"] and q["oriented"] and q["components"] == 1 and q["genus"] == 0 and q["n_negative"] == 0
    assert q["min_angle"] > 30 and q["mean_edge"] == pytest.approx(coarse_q["mean_edge"] / 2, rel=0.01)
    # Each new vertex sits near the middle of its edge, off the chord by O(h²): at most 0.04 h here.
    off = np.linalg.norm(fine[len(verts):] - verts[_edges(faces)].mean(1), axis=1)
    assert off.max() < 0.1 * coarse_q["mean_edge"]


def test_refine_shape_of_a_torus_nests_on_the_surface():
    # Round torus: new vertices on (√(x² + y²) - R0)² + z² = r0², also on the edges across u = 0 and v = 0. Its split
    # has negative weights (137) although the coarse mesh has none; the flips that remove them move no vertex.
    d = TorusDeformation(0.5)
    verts, faces, quality = build_shape(d, target_edge=0.06)
    fine, fine_faces = refine_shape(d, verts, faces, quality)                      # target_edge 0.06 from quality
    x = _unnormalised(fine, quality)
    np.testing.assert_allclose((np.hypot(x[:, 0], x[:, 1]) - d.major_radius)**2 + x[:, 2]**2, d.minor_radius**2,
                               rtol=0, atol=1e-12)
    split, split_faces = refine_shape(d, verts, faces, quality, flip=False)
    assert quality["n_negative"] == 0 and mesh_quality(split, split_faces)["n_negative"] > 100
    assert np.array_equal(split, fine) and mesh_quality(fine, fine_faces)["n_negative"] == 0
    # Deformed torus: each new vertex is x(u, v) at the circular means of its edge's angles, here arg(e^{ia} + e^{ib}).
    d = sample_torus_deformation(np.random.default_rng(1))
    verts, faces, quality = build_shape(d, target_edge=0.06)
    fine, fine_faces = refine_shape(d, verts, faces, quality)
    assert quality["flips"] > 0 and np.array_equal(fine[:len(verts)], verts) and len(fine) == 4 * len(verts)
    u, v = torus_angles(torus(0.06, d.major_radius, d.minor_radius)[0], d.major_radius)
    edges = _edges(faces)
    assert np.sum(np.ptp(u[edges], axis=1) > np.pi) > 10 and np.sum(np.ptp(v[edges], axis=1) > np.pi) > 10   # seams
    mid_u, mid_v = (np.angle(np.exp(1j * angle[edges]).sum(1)) for angle in (u, v))
    np.testing.assert_allclose(_unnormalised(fine[len(verts):], quality), torus_surface(d, mid_u, mid_v),
                               rtol=0, atol=1e-12)
    q = mesh_quality(fine, fine_faces)
    assert q["closed"] and q["oriented"] and q["components"] == 1 and q["genus"] == 1 and q["n_negative"] == 0
    with pytest.raises(ValueError, match="vertices"):
        refine_shape(d, verts, faces, quality, target_edge=0.037)                  # made with target_edge 0.06


def test_make_shape_redraws_until_the_gates_pass():
    shape = make_shape("sphere", np.random.default_rng(2), subdivisions=4, relax=False, gates={"min_wavenumber": 8.0},
                       tau_range=(0.1, 0.2))
    assert shape.attempts == len(shape.rejections) + 1 and shape.attempts > 1
    assert all(reason[0].startswith("p95/mean") for reason in shape.rejections)
    assert check_gates(shape.quality, 0, min_wavenumber=8.0) == (True, [])
    with pytest.raises(RuntimeError, match="no sphere passed"):
        make_shape("sphere", np.random.default_rng(0), max_attempts=2, subdivisions=2)
    with pytest.raises(ValueError):
        make_shape("cube", np.random.default_rng(0))
    with pytest.raises(ValueError):
        make_shape("sphere", np.random.default_rng(0), max_attempts=0)
