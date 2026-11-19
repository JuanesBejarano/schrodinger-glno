import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from schrodinger_glno.initial_conditions import (gaussian_packet, mass_norm, points_per_wavelength,
                                                 spectral_fraction, tangent_direction)
from schrodinger_glno.mesh import cotan_laplacian, icosphere, vertex_areas, vertex_normals
from schrodinger_glno.solver import build_cn_matrices, energy, solve_CN, solve_CN_full, spectral_evolution


def _sphere(subdivisions):
    verts, faces = icosphere(subdivisions)
    return verts, faces, cotan_laplacian(verts, faces), vertex_areas(verts, faces), vertex_normals(verts, faces)


def _packet(verts, mass, normals, k0=4.0, sigma=0.3):
    center = int(np.argmax(verts[:, 2]))
    return center, gaussian_packet(verts, mass, normals, center, k0, sigma, np.array([1.0, 0, 0]))


def test_icosphere_counts_and_area():
    verts, faces, _, mass, normals = _sphere(3)
    assert verts.shape == (642, 3) and faces.shape == (1280, 3)
    np.testing.assert_allclose(np.linalg.norm(verts, axis=1), 1)
    assert abs(mass.sum() - 4 * np.pi) / (4 * np.pi) < 0.01
    assert np.all(np.sum(normals * verts, axis=1) > 0.99)


def test_cotan_laplacian_is_symmetric_psd_and_kills_constants():
    verts, _, L, _, _ = _sphere(3)
    assert abs(L - L.T).max() < 1e-14
    np.testing.assert_allclose(L @ np.ones(len(verts)), 0, atol=1e-12)
    x = np.random.default_rng(0).normal(size=len(verts))
    assert x @ (L @ x) > 0


def test_laplacian_spectrum_matches_spherical_harmonics():
    _, _, L, mass, _ = _sphere(4)
    evals = np.sort(spla.eigsh(L, k=16, M=sp.diags(mass), sigma=-1e-3)[0])
    expected = np.repeat([l * (l + 1) for l in range(4)], [2 * l + 1 for l in range(4)])
    assert abs(evals[0]) < 1e-8
    np.testing.assert_allclose(evals[1:], expected[1:], rtol=0.01)


def test_packet_is_mass_normalized_and_centred():
    verts, faces, _, mass, normals = _sphere(4)
    center, psi = _packet(verts, mass, normals)
    assert abs(mass_norm(psi, mass) - 1) < 1e-12
    assert np.argmax(np.abs(psi)) == center
    assert points_per_wavelength(verts, faces, 4.0, 0.3) > 8


def test_tangent_direction_is_tangent_and_rejects_normal_direction():
    normal = np.array([0.0, 0.0, 2.0])
    t = tangent_direction(normal, np.array([1.0, 1.0, 5.0]))
    assert abs(t @ normal) < 1e-12 and abs(np.linalg.norm(t) - 1) < 1e-12
    with pytest.raises(ValueError):
        tangent_direction(normal, np.array([0.0, 0.0, 1.0]))


def test_cn_conserves_mass_norm_and_energy():
    verts, _, L, mass, normals = _sphere(3)
    _, psi = _packet(verts, mass, normals)
    V = np.zeros(len(verts))
    lu, u1 = build_cn_matrices(L, mass, V, 0.01)
    traj = solve_CN_full(psi, lu, u1, 60)
    norms = np.sqrt(np.sum(mass[:, None] * np.abs(traj)**2, axis=0))
    np.testing.assert_allclose(norms, 1, atol=1e-12)
    assert abs(energy(traj[:, -1], L, mass, V) - energy(psi, L, mass, V)) < 1e-10


def test_cn_converges_to_exact_semidiscrete_solution_at_second_order():
    verts, _, L, mass, normals = _sphere(3)
    _, psi = _packet(verts, mass, normals, k0=3.0)
    evals, evecs = sla.eigh(L.toarray(), np.diag(mass))  # all modes: exact for the mesh ODE
    assert spectral_fraction(psi, mass, evecs) == pytest.approx(1, abs=1e-10)
    T, errors = 0.2, []
    for dt in (0.01, 0.005):
        steps = int(round(T / dt)) + 1
        lu, u1 = build_cn_matrices(L, mass, np.zeros(len(verts)), dt)
        end = solve_CN(psi, lu, u1, steps, np.array([steps - 1]))[:, 0]
        reference = spectral_evolution(psi, mass, evals / 2, evecs, [T])[:, 0]
        errors.append(mass_norm(end - reference, mass))
    assert 3.5 < errors[0] / errors[1] < 4.5


def test_packet_centre_follows_great_circle_at_speed_k0():
    verts, _, L, mass, normals = _sphere(4)
    k0, dt, steps = 5.0, 0.001, 61
    center, psi = _packet(verts, mass, normals, k0=k0)
    x0, t0 = verts[center], tangent_direction(normals[center], np.array([1.0, 0, 0]))
    lu, u1 = build_cn_matrices(L, mass, np.zeros(len(verts)), dt)
    end = solve_CN(psi, lu, u1, steps, np.array([steps - 1]))[:, 0]
    mean = (mass * np.abs(end)**2) @ verts
    angle = np.arctan2(mean @ t0, mean @ x0)
    assert angle == pytest.approx(k0 * (steps - 1) * dt, rel=0.05)


def test_build_cn_matrices_rejects_bad_inputs():
    _, _, L, mass, _ = _sphere(1)
    V = np.zeros(len(mass))
    with pytest.raises(ValueError):
        build_cn_matrices(L, mass, V, 0.0)
    with pytest.raises(ValueError):
        build_cn_matrices(L, mass, V[:-1], 0.01)
    with pytest.raises(ValueError):
        build_cn_matrices(L, -mass, V, 0.01)


def test_consistent_mass_rows_sum_to_lumped_mass():
    from schrodinger_glno.mesh import blended_mass, consistent_mass
    verts, faces, _, mass, _ = _sphere(3)
    Mc = consistent_mass(verts, faces)
    assert abs(Mc - Mc.T).max() < 1e-15
    np.testing.assert_allclose(np.asarray(Mc.sum(axis=1)).ravel(), mass, rtol=1e-12)
    np.testing.assert_allclose(blended_mass(verts, faces).sum(), mass.sum(), rtol=1e-12)


def test_blended_mass_eigenvalues_are_much_closer_to_spherical_harmonics():
    from schrodinger_glno.mesh import blended_mass, laplacian_eigenpairs
    verts, faces, L, mass, _ = _sphere(4)
    ell = np.repeat(np.arange(8), 2 * np.arange(8) + 1)          # ℓ = 0..7, 64 modes
    exact = ell * (ell + 1.0)
    errors = {}
    for name, M in (("lumped", mass), ("blended", blended_mass(verts, faces))):
        evals = laplacian_eigenpairs(L, M, len(ell))[0]
        errors[name] = np.max(np.abs(evals[1:] - exact[1:]) / exact[1:])
    # 2 562 vertices: lumped up to -2.1 % (ℓ = 7); blended about +0.07 % per level, at most 0.35 %
    assert errors["lumped"] > 0.015 and errors["blended"] < 0.005 and errors["lumped"] > 4 * errors["blended"]


@pytest.mark.slow
def test_blended_mass_tracks_the_continuous_sphere():
    """Regression for the phase error of the lumped mass: compare with the exact continuum solution."""
    from schrodinger_glno.mesh import blended_mass
    from schrodinger_glno.reference import sphere_exact_solution
    from schrodinger_glno.sampling import direction_from_angle
    verts, faces, L, lumped, normals = _sphere(4)
    center, k0, sigma, T, dt = int(np.argmax(verts[:, 2])), 4.0, 0.3, 0.3, 0.001
    direction = direction_from_angle(normals[center], 0.7)
    exact = sphere_exact_solution(verts, verts[center], direction, k0, sigma, np.array([T]))[:, 0]
    errors = {}
    for name, M in (("lumped", lumped), ("blended", blended_mass(verts, faces))):
        psi0 = gaussian_packet(verts, M, normals, center, k0, sigma, direction)
        lu, u1 = build_cn_matrices(L, M, np.zeros(len(verts)), dt)
        steps = int(round(T / dt)) + 1
        end = solve_CN(psi0, lu, u1, steps, np.array([steps - 1]))[:, 0]
        errors[name] = mass_norm(end - exact, lumped)
    assert errors["blended"] < 0.012 and errors["lumped"] > 4 * errors["blended"]


def test_cn_conserves_the_blended_norm():
    from schrodinger_glno.mesh import blended_mass
    verts, faces, L, _, normals = _sphere(3)
    M = blended_mass(verts, faces)
    _, psi = _packet(verts, M, normals)
    lu, u1 = build_cn_matrices(L, M, np.zeros(len(verts)), 0.01)
    traj = solve_CN_full(psi, lu, u1, 40)
    norms = np.real(np.einsum("vt,vt->t", traj.conj(), M @ traj))
    np.testing.assert_allclose(norms, 1, atol=1e-12)
