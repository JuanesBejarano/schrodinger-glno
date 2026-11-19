"""Exact free Schrödinger evolution on the smooth torus of revolution (``reference.torus_*``)."""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from schrodinger_glno.initial_conditions import mass_norm
from schrodinger_glno.mesh import blended_mass, cotan_laplacian, vertex_areas
from schrodinger_glno.reference import (sphere_packet, torus_angles, torus_eigenpairs, torus_exact_solution,
                                        torus_points, torus_resolution)
from schrodinger_glno.solver import build_cn_matrices, solve_CN

R, r = 1.0, 0.5
# (u0, v0, direction angle from ∂u towards ∂v, k0, σ): outer equator, inner equator, top, generic.
PACKETS = {"outer-along-u": (0.3, 0.0, 0.0, 12.0, 0.12),
           "inner-diagonal": (1.0, np.pi, 0.7, 12.0, 0.12),
           "top-along-v": (2.0, np.pi / 2, np.pi / 2, 8.0, 0.3),
           "generic": (4.0, 2.2, 2.5, 3.0, 0.2)}


def _tangent(u, v, angle):
    """Unit tangent ``cos(angle) e_u + sin(angle) e_v`` at x(u, v)."""
    e_u = np.array([-np.sin(u), np.cos(u), 0.0])
    e_v = np.array([-np.sin(v) * np.cos(u), -np.sin(v) * np.sin(u), np.cos(v)])
    return np.cos(angle) * e_u + np.sin(angle) * e_v


def _grid(n_u, n_v):
    """Tensor (u, v) grid: angles, points (n_u n_v, 3) and trapezoid weights of dA = r ρ du dv."""
    u, v = 2 * np.pi * np.arange(n_u) / n_u, 2 * np.pi * np.arange(n_v) / n_v
    points = torus_points(u[:, None], v[None, :], R, r).reshape(-1, 3)
    weights = np.broadcast_to(r * (R + r * np.cos(v)) * (2 * np.pi)**2 / (n_u * n_v), (n_u, n_v)).ravel()
    return u, v, points, weights


def _targets(u0, v0, sigma, n, seed):
    """Half uniform in (u, v), half within a few σ of the packet centre."""
    rng = np.random.default_rng(seed)
    u = np.concatenate([rng.uniform(0, 2 * np.pi, n // 2), u0 + rng.normal(0, 2 * sigma / (R + r * np.cos(v0)), n // 2)])
    v = np.concatenate([rng.uniform(0, 2 * np.pi, n // 2), v0 + rng.normal(0, 2 * sigma / r, n // 2)])
    return torus_points(u, v, R, r)


def _wrapped(angle):
    return np.angle(np.exp(1j * angle))


@pytest.mark.parametrize("major, minor", [(1.0, 0.5), (2.0, 0.7), (1.0, 0.95)])
def test_torus_angles_invert_the_parametrization(major, minor):
    u, v = np.random.default_rng(1).uniform(0, 2 * np.pi, (2, 1000))
    points = torus_points(u, v, major, minor)
    np.testing.assert_allclose((np.hypot(points[:, 0], points[:, 1]) - major)**2 + points[:, 2]**2, minor**2,
                               rtol=0, atol=1e-13)
    back_u, back_v = torus_angles(points, major)
    assert np.all((0 <= back_u) & (back_u < 2 * np.pi) & (0 <= back_v) & (back_v < 2 * np.pi))
    assert np.abs(_wrapped(back_u - u)).max() < 1e-12 and np.abs(_wrapped(back_v - v)).max() < 1e-12
    np.testing.assert_allclose(torus_points(back_u, back_v, major, minor), points, rtol=0, atol=1e-13)
    # A point moved along the outward normal keeps its angles (they are those of the nearest surface point)
    # while it neither reaches the core circle (shift -r) nor the z axis (shift R - r on the inner equator).
    normal = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], axis=1)
    for shift in (-0.9 * minor, 0.9 * (major - minor)):
        moved_u, moved_v = torus_angles(points + shift * normal, major)
        assert np.abs(_wrapped(moved_u - u)).max() < 1e-12 and np.abs(_wrapped(moved_v - v)).max() < 1e-12
    landmarks = np.array([[major + minor, 0, 0], [major, 0, minor], [major - minor, 0, 0], [0, major, -minor]])
    np.testing.assert_allclose(np.column_stack(torus_angles(landmarks, major)),
                               [[0, 0], [0, np.pi / 2], [0, np.pi], [np.pi / 2, 3 * np.pi / 2]], rtol=0, atol=1e-15)


@pytest.mark.parametrize("m", [0, 3, 40])
def test_eigenvectors_are_orthonormal_with_the_area_weight_and_have_exact_parity(m):
    n_v = 65
    evals, G = torus_eigenpairs(m, n_v, R, r)
    rho = R + r * np.cos(2 * np.pi * np.arange(n_v) / n_v)
    np.testing.assert_allclose((2 * np.pi)**2 * r / n_v * G.T @ (rho[:, None] * G), np.eye(n_v), rtol=0, atol=1e-12)
    assert np.all(np.diff(evals) >= 0) and evals[0] > -1e-10
    mirrored = G[-np.arange(n_v) % n_v]                                 # g(-v_j)
    parity = np.sign(np.sum(mirrored * G, axis=0))
    np.testing.assert_allclose(mirrored, G * parity, rtol=0, atol=1e-12)
    assert np.sum(parity > 0) == (n_v + 1) // 2                        # (n_v + 1)/2 even, (n_v - 1)/2 odd


def test_zero_eigenvalue_is_simple_and_its_eigenfunction_is_constant():
    evals, G = torus_eigenpairs(0, 65, R, r)
    assert abs(evals[0]) < 1e-10 and evals[1] > 3.5                    # λ_{0,1} ≈ 3.907
    np.testing.assert_allclose(np.abs(G[:, 0]), 1 / np.sqrt(4 * np.pi**2 * R * r), rtol=1e-12)
    lowest = [torus_eigenpairs(m, 65, R, r)[0][0] for m in range(1, 6)]
    assert lowest[0] > 0.9 and np.all(np.diff(lowest) > 0)             # λ_{1,0} ≈ 0.997: no other zero


@pytest.mark.parametrize("m", [1, 2, 7, 30])
def test_eigenpairs_are_even_in_m(m):
    (plus, G_plus), (minus, G_minus) = torus_eigenpairs(m, 51, R, r), torus_eigenpairs(-m, 51, R, r)
    np.testing.assert_array_equal(plus, minus)
    np.testing.assert_array_equal(G_plus, G_minus)


@pytest.mark.parametrize("m", [-4, 0, 4, 11])
def test_eigenfunctions_satisfy_the_laplace_beltrami_equation_off_the_grid(m):
    """Non-conservative form Δ = ∂u²/ρ² + ∂v²/r² - sin v ∂v / (r ρ), on a 3x finer grid."""
    n_v, fine, count = 63, 189, 10
    evals, G = torus_eigenpairs(m, n_v, R, r)
    k = np.fft.fftfreq(n_v, 1.0 / n_v)
    coeffs = np.fft.fft(G[:, :count], axis=0) / n_v
    v = 2 * np.pi * np.arange(fine) / fine
    E = np.exp(1j * np.outer(v, k))
    g, dg, d2g = (np.real(E @ (factor[:, None] * coeffs)) for factor in (np.ones(n_v), 1j * k, -k**2))
    rho = (R + r * np.cos(v))[:, None]
    laplacian = -m**2 * g / rho**2 + d2g / r**2 - np.sin(v)[:, None] * dg / (r * rho)
    assert np.abs(-laplacian - evals[:count] * g).max() < 1e-10 * evals[count - 1] * np.abs(g).max()


def _fd_eigenvalues(m, n, count):
    """Second-order finite differences for -(ρ g')'/r² + m² g/ρ = λ ρ g on n periodic points."""
    h = 2 * np.pi / n
    v, j = h * np.arange(n), np.arange(n)
    rho, rho_half = R + r * np.cos(v), R + r * np.cos(v + h / 2)       # ρ_j and ρ_{j+1/2}
    diagonal = (rho_half + np.roll(rho_half, 1)) / (r * h)**2 + m**2 / rho
    off = -rho_half / (r * h)**2
    K = sp.coo_matrix((np.concatenate([diagonal, off, off]),
                       (np.concatenate([j, j, (j + 1) % n]), np.concatenate([j, (j + 1) % n, j]))), shape=(n, n))
    return np.sort(spla.eigsh(K.tocsc(), k=count, M=sp.diags(rho).tocsc(), sigma=-1.0, return_eigenvectors=False))


def test_lowest_eigenvalues_match_richardson_extrapolated_finite_differences():
    per_m, m_values = 6, range(9)
    spectral = {m: torus_eigenpairs(m, 65, R, r)[0][:per_m] for m in m_values}
    lowest = sorted((lam, m, n) for m in m_values for n, lam in enumerate(spectral[m]))[:15]   # ±m counted once
    # These are the 15 smallest of the whole spectrum: all below every per-m cut and below m = 8.
    assert lowest[-1][0] < min(min(s[-1] for s in spectral.values()), spectral[8][0])
    assert lowest[-1][0] == pytest.approx(16.21, abs=0.01)
    extrapolated = {}
    for m in {m for _, m, _ in lowest}:
        coarse, middle, fine = (_fd_eigenvalues(m, 256 * s, per_m) for s in (1, 2, 4))
        np.testing.assert_allclose(fine, spectral[m], rtol=1e-4, atol=1e-9)   # plain O(h²) at n = 1024
        extrapolated[m] = (64 * fine - 20 * middle + coarse) / 45      # removes the h² and h⁴ terms
    for lam, m, n in lowest:                                            # measured: ≤ 2e-12 relative
        assert extrapolated[m][n] == pytest.approx(lam, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("name", PACKETS)
def test_initial_state_is_the_normalized_chordal_packet(name):
    u0, v0, angle, k0, sigma = PACKETS[name]
    x0, t0 = torus_points(u0, v0, R, r), _tangent(u0, v0, angle)
    targets = np.vstack([x0, _targets(u0, v0, sigma, 3000, seed=2)])
    psi = torus_exact_solution(targets, x0, t0, k0, sigma, [0.0], R, r)[:, 0]
    _, _, points, weights = _grid(401, 301)                             # not the solver's grid
    norm = np.sqrt(weights @ np.abs(sphere_packet(points, x0, t0, k0, sigma))**2)
    expected = sphere_packet(targets, x0, t0, k0, sigma) / norm
    assert 1 < np.abs(expected).max() < 4
    assert np.abs(psi - expected).max() < 1e-11                         # measured ~1e-14


@pytest.mark.parametrize("name", ["inner-diagonal", "top-along-v"])
def test_norm_is_one_at_all_times(name):
    u0, v0, angle, k0, sigma = PACKETS[name]
    x0, t0 = torus_points(u0, v0, R, r), _tangent(u0, v0, angle)
    n_u, n_v = torus_resolution(k0, sigma, R, r)
    # ψ is a trigonometric polynomial of degree (n_u - 1)/2 × (n_v - 1)/2: this trapezoid rule is exact for ρ|ψ|².
    _, _, points, weights = _grid(n_u + 2, n_v + 2)
    psi = torus_exact_solution(points, x0, t0, k0, sigma, [0.0, 0.1, 0.25, 0.5], R, r)
    np.testing.assert_allclose(weights @ np.abs(psi)**2, 1, rtol=0, atol=1e-12)


def test_doubling_the_resolution_changes_nothing():
    u0, v0, angle, k0, sigma = PACKETS["inner-diagonal"]
    x0, t0 = torus_points(u0, v0, R, r), _tangent(u0, v0, angle)
    targets = _targets(u0, v0, sigma, 2000, seed=4)
    times = [0.0, 0.2, 0.5]
    n_u, n_v = torus_resolution(k0, sigma, R, r)
    assert (n_u, n_v) == (241, 147)
    default = torus_exact_solution(targets, x0, t0, k0, sigma, times, R, r)
    doubled = torus_exact_solution(targets, x0, t0, k0, sigma, times, R, r, n_u=2 * n_u, n_v=2 * n_v)
    assert np.abs(default).max() > 1
    assert np.abs(doubled - default).max() < 1e-10                     # measured 6e-13 (eigenvalue round-off)


def test_solution_satisfies_the_schrodinger_equation():
    """i ∂ψ/∂t = -½ Δψ on a (u, v) grid: sixth-order differences in t, exact Fourier derivatives in u and v."""
    u0, v0, angle, k0, sigma = 4.0, 2.2, 2.5, 6.0, 0.25
    x0, t0 = torus_points(u0, v0, R, r), _tangent(u0, v0, angle)
    n_u, n_v = torus_resolution(k0, sigma, R, r)
    _, v, points, _ = _grid(n_u, n_v)
    delta = 2.5e-4
    psi = torus_exact_solution(points, x0, t0, k0, sigma, 0.2 + delta * np.arange(-3, 4), R, r).reshape(n_u, n_v, 7)
    m = np.fft.fftfreq(n_u, 1.0 / n_u)[:, None]
    k = np.fft.fftfreq(n_v, 1.0 / n_v)[None, :]
    F = np.fft.fft2(psi[:, :, 3])
    d_uu, d_v, d_vv = (np.fft.ifft2(factor * F) for factor in (-m**2, 1j * k, -k**2))
    rho = R + r * np.cos(v)
    laplacian = d_uu / rho**2 + d_vv / r**2 - np.sin(v) * d_v / (r * rho)
    d_t = np.tensordot(psi, np.array([-1, 9, -45, 0, 45, -9, 1]) / (60 * delta), axes=([2], [0]))
    assert np.abs(laplacian).max() > 10
    assert np.abs(1j * d_t + 0.5 * laplacian).max() < 1e-10 * np.abs(laplacian).max()   # measured 5e-13


def test_packet_on_the_outer_equator_travels_along_it_at_speed_k0():
    """The outer equator is a geodesic: the centre moves at k0, so ⟨u⟩ (R + r) ≈ k0 t, and ⟨v⟩ stays 0.

    Exactly (Ehrenfest, d⟨u⟩/dt = ⟨ρ⁻² (-i∂u)⟩): the initial angular speed is k0 ⟨cos u / ρ⟩_ψ0,
    1 % above k0 / (R + r) because the rings off the equator are shorter.
    """
    k0, sigma, delta = 8.0, 0.15, 1e-4
    x0, t0 = np.array([R + r, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    n_u, n_v = torus_resolution(k0, sigma, R, r)
    u, v, points, weights = _grid(n_u + 2, n_v + 2)
    times = np.array([-delta, 0.0, delta, 0.02, 0.04, 0.06])
    density = weights[:, None] * np.abs(torus_exact_solution(points, x0, t0, k0, sigma, times, R, r))**2
    u_signed = np.repeat(_wrapped(u), len(v))                           # the packet stays far from u = ±π
    v_of = np.tile(v, len(u))
    mean_u = u_signed @ density
    speed = (mean_u[2] - mean_u[0]) / (2 * delta)
    predicted = k0 * density[:, 1] @ (np.cos(u_signed) / (R + r * np.cos(v_of)))
    assert speed == pytest.approx(predicted, rel=1e-6)                  # measured 4e-8
    assert predicted * (R + r) / k0 == pytest.approx(1.0107, abs=1e-3)
    np.testing.assert_allclose(mean_u[3:] * (R + r), k0 * times[3:], rtol=0.03)   # measured +1.2 .. +2.4 %
    np.testing.assert_allclose(np.sin(v_of) @ density, 0, atol=1e-12)


def test_solution_is_covariant_under_rotations_about_the_axis_and_the_mirror_z_to_minus_z():
    u0, v0, angle, k0, sigma = 4.0, 2.2, 2.5, 5.0, 0.2
    x0, t0 = torus_points(u0, v0, R, r), _tangent(u0, v0, angle)
    targets = _targets(u0, v0, sigma, 1500, seed=5)
    times = [0.0, 0.15, 0.4]
    psi = torus_exact_solution(targets, x0, t0, k0, sigma, times, R, r)
    a = 1.234
    rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    mirror = np.diag([1.0, 1.0, -1.0])
    for Q in (rotation, mirror):
        moved = torus_exact_solution(targets @ Q.T, Q @ x0, Q @ t0, k0, sigma, times, R, r)
        assert np.abs(moved - psi).max() < 1e-11 * np.abs(psi).max()


def _grid_mesh(n_u, n_v):
    """Structured triangulation of the torus: each (u, v) cell split along a diagonal."""
    u, v = 2 * np.pi * np.arange(n_u) / n_u, 2 * np.pi * np.arange(n_v) / n_v
    verts = torus_points(u[:, None], v[None, :], R, r).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_u), np.arange(n_v), indexing="ij")
    a, b = i * n_v + j, (i + 1) % n_u * n_v + j
    c, d = (i + 1) % n_u * n_v + (j + 1) % n_v, i * n_v + (j + 1) % n_v
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return verts, faces


def test_crank_nicolson_on_refined_meshes_converges_to_the_exact_solution():
    """The mesh pipeline (cotan Laplacian, blended mass, CN) approaches the continuum at second order."""
    k0, sigma, T, dt = 4.0, 0.3, 0.1, 1e-3
    x0, t0 = torus_points(0.0, 0.0, R, r), _tangent(0.0, 0.0, 0.6)
    errors = []
    for n_u, n_v in [(48, 24), (96, 48)]:
        verts, faces = _grid_mesh(n_u, n_v)
        L, M, lumped = cotan_laplacian(verts, faces), blended_mass(verts, faces), vertex_areas(verts, faces)
        psi0 = sphere_packet(verts, x0, t0, k0, sigma)
        psi0 /= mass_norm(psi0, M)
        lu, u1 = build_cn_matrices(L, M, np.zeros(len(verts)), dt)
        steps = int(round(T / dt)) + 1
        end = solve_CN(psi0, lu, u1, steps, np.array([steps - 1]))[:, 0]
        exact = torus_exact_solution(verts, x0, t0, k0, sigma, [T], R, r)[:, 0]
        errors.append(mass_norm(end - exact, lumped))
    # measured 0.075 and 0.022 (time-step error negligible: dt = 5e-4 gives the same)
    assert errors[1] < 0.03 and 3 < errors[0] / errors[1] < 4.5


@pytest.mark.parametrize("change", [dict(major_radius=0.5), dict(major_radius=0.4), dict(minor_radius=0.0),
                                    dict(minor_radius=-0.1), dict(major_radius=np.inf), dict(minor_radius=np.nan),
                                    dict(sigma=0.0), dict(sigma=-0.2), dict(k0=np.nan)])
def test_rejects_bad_parameters(change):
    p = dict(k0=4.0, sigma=0.2, major_radius=1.0, minor_radius=0.5) | change
    x0, t0 = np.array([1.5, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        torus_exact_solution(x0[None], x0, t0, p["k0"], p["sigma"], [0.0], p["major_radius"], p["minor_radius"])


def test_rejects_bad_grids_and_targets():
    x0, t0 = np.array([1.5, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    for n_v in (64, 1, 2.5):
        with pytest.raises(ValueError):
            torus_eigenpairs(0, n_v, R, r)
    with pytest.raises(ValueError):
        torus_exact_solution(x0, x0, t0, 4.0, 0.2, [0.0], R, r)          # targets must be (N, 3)
    assert torus_exact_solution(x0[None], x0, t0, 4.0, 0.2, 0.0, R, r, n_u=40, n_v=30).shape == (1, 1)
