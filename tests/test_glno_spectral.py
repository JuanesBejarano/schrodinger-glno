"""Load-time spectral transforms for stage 3: Ritz polar basis, eigenvalue groups, geo feature, CN multiplier."""
import json
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp
from schrodinger_glno.glno.spectral import (SINGULAR_VALUE_BOUNDS, canonical_signs, centroid, eigen_clusters,
                                            geo_feature, mass_relative_error, multiplier, propagate,
                                            ritz_polar_basis, spectral_oracle_error)
from schrodinger_glno.mesh import (blended_mass, cotan_laplacian, icosphere, laplacian_eigenpairs, torus,
                                   vertex_areas)
from schrodinger_glno.solver import build_cn_matrices, solve_CN

SHAPES = Path(__file__).resolve().parents[1] / "data" / "shapes"
K, DT, STEPS = 100, 1e-3, np.arange(0, 501, 50)


def _area_4pi(verts, faces):
    verts = verts * np.sqrt(4 * np.pi / vertex_areas(verts, faces).sum())
    return verts - centroid(verts, vertex_areas(verts, faces))


def _jittered_sphere():
    verts, faces = icosphere(3)
    verts = verts * (1 + 0.05 * np.random.default_rng(1).standard_normal((len(verts), 1)))
    return _area_4pi(verts, faces), faces


def _torus():
    verts, faces = torus(0.12)
    return _area_4pi(verts, faces), faces


@pytest.fixture(scope="module", params=["sphere", "torus"])
def case(request):
    verts, faces = _jittered_sphere() if request.param == "sphere" else _torus()
    L, mass = cotan_laplacian(verts, faces), vertex_areas(verts, faces)
    lam, phi = laplacian_eigenpairs(L, mass, K)
    phi = phi.astype(np.float32)            # stored as float32, like the dataset
    return {"verts": verts, "faces": faces, "L": L, "mass": mass, "blended": blended_mass(verts, faces),
            "lam": lam, "phi": phi, "ritz": ritz_polar_basis(verts, faces, phi)}


def _reduced(case):
    """B^(-1/2) A B^(-1/2) and B^(-1/2) for the Ritz pencil (A, B) of the case."""
    phi = case["phi"].astype(np.float64)
    A = phi.T @ (case["L"] @ phi)
    B = phi.T @ (case["blended"] @ phi)
    w, V = np.linalg.eigh((B + B.T) / 2)
    inv_sqrt = V @ np.diag(w ** -0.5) @ V.T
    return inv_sqrt @ ((A + A.T) / 2) @ inv_sqrt, inv_sqrt


def test_polar_basis_is_lumped_orthonormal_and_diagonalises_the_pencil(case):
    ritz = case["ritz"]
    assert np.abs(ritz.U.T @ ritz.U - np.eye(K)).max() < 1e-12
    E = ritz.E.astype(np.float64)
    assert ritz.E.dtype == np.float32
    assert np.abs(E.T @ (case["mass"][:, None] * E) - np.eye(K)).max() < 1e-6
    assert 0 <= ritz.mu[0] < 1e-10 and np.all(np.diff(ritz.mu) >= 0)
    # U = B^(1/2) y, so U diagonalises B^(-1/2) A B^(-1/2) with the Ritz values μ
    reduced, _ = _reduced(case)
    assert np.abs(reduced @ ritz.U - ritz.U * ritz.mu).max() / ritz.mu[-1] < 1e-10
    low, high = SINGULAR_VALUE_BOUNDS
    assert low - 1e-6 <= ritz.singular_values.min() and ritz.singular_values.max() <= high
    # canonical signs: the largest entry of every column is positive
    assert np.all(ritz.U[np.argmax(np.abs(ritz.U), axis=0), np.arange(K)] > 0)


def test_polar_basis_follows_crank_nicolson_much_better_than_the_lumped_pairs(case):
    """ψ0 from the 30 lowest modes, evolved by the real CN solver with the blended mass.

    The polar basis is within a small factor of exact Ritz (Ψ = Φy, blended projection)
    and far better than the lumped pairs. The meshes are coarse (K/N = 6-16 %), so
    the errors are larger than on the dataset; see the data-gated test for tight bounds.
    """
    phi, mass, ritz = case["phi"].astype(np.float64), case["mass"], case["ritz"]
    rng = np.random.default_rng(30)
    coeffs = np.zeros(K, complex)
    coeffs[1:30] = rng.standard_normal(29) + 1j * rng.standard_normal(29)
    psi0 = phi @ coeffs
    LU, U1 = build_cn_matrices(case["L"], case["blended"], np.zeros(len(psi0)), DT)
    reference = solve_CN(psi0, LU, U1, STEPS[-1] + 1, STEPS)
    _, inv_sqrt = _reduced(case)
    Psi = phi @ (inv_sqrt @ ritz.U)                     # y = B^(-1/2) U
    exact_ritz = Psi @ ((Psi.T @ (case["blended"] @ psi0))[:, None] * multiplier(ritz.mu, STEPS, DT, "cn"))
    error = lambda pred: mass_relative_error(pred[:, 1:], reference[:, 1:], mass).max()
    polar = error(propagate(ritz.E, ritz.mu, mass, psi0, STEPS, DT, "cn"))
    lumped = error(propagate(case["phi"], case["lam"], mass, psi0, STEPS, DT, "cn"))
    assert polar < 1e-2 and lumped > 3e-2 and polar < lumped / 5
    assert polar <= 2 * error(exact_ritz) + 1e-3


def test_canonical_basis_does_not_depend_on_the_lapack_driver(case):
    for driver in ("gv", "gvx"):
        other = ritz_polar_basis(case["verts"], case["faces"], case["phi"], driver=driver)
        assert np.abs(other.U - case["ritz"].U).max() < 1e-8
        assert np.abs(other.E - case["ritz"].E).max() < 1e-6


def test_rigid_motion_changes_nothing(case):
    rng = np.random.default_rng(5)
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    moved = case["verts"] @ rotation.T + np.array([0.3, -1.2, 2.0])
    other = ritz_polar_basis(moved, case["faces"], case["phi"])          # same Φ: only the operators move
    assert np.abs(other.mu - case["ritz"].mu).max() < 1e-10 * case["ritz"].mu[-1]
    assert np.abs(other.E - case["ritz"].E).max() < 1e-6
    mass = case["mass"]
    np.testing.assert_allclose(geo_feature(moved, vertex_areas(moved, case["faces"]), 1.4),
                               geo_feature(case["verts"], mass, 1.4), atol=1e-6)


def test_ritz_rejects_a_basis_that_is_not_lumped_orthonormal(case):
    with pytest.raises(ValueError, match="orthonormal"):
        ritz_polar_basis(case["verts"], case["faces"], 2 * case["phi"])
    with pytest.raises(ValueError):
        ritz_polar_basis(case["verts"], case["faces"], case["phi"][:-1])


def test_eigen_clusters():
    mu = np.array([0.0, 1.0, 1.00001, 1.00002, 2.0, 2.0, 3.0, 3.0004])
    np.testing.assert_array_equal(eigen_clusters(mu, 1e-4), [0, 1, 1, 1, 4, 4, 6, 7])
    np.testing.assert_array_equal(eigen_clusters(mu, 1e-3), [0, 1, 1, 1, 4, 4, 6, 6])
    np.testing.assert_array_equal(eigen_clusters(mu, 0.0), np.arange(8))
    np.testing.assert_array_equal(eigen_clusters(np.zeros(3), 1e-4), [0, 1, 1])   # mode 0 always alone
    rng = np.random.default_rng(0)
    mu = np.concatenate([[0], np.sort(rng.uniform(1, 50, 300))])
    counts = [len(np.unique(eigen_clusters(mu, tau))) for tau in (1e-5, 1e-4, 1e-3, 1e-2)]
    assert counts == sorted(counts, reverse=True)
    labels = eigen_clusters(mu, 1e-3)
    assert np.all(np.diff(labels) >= 0) and np.all(labels <= np.arange(len(mu)))
    assert np.all(labels[labels] == labels)            # each label is the first mode of its group
    with pytest.raises(ValueError):
        eigen_clusters(mu[::-1], 1e-3)


def test_multiplier_and_geo_feature():
    mu, steps = np.array([0.0, 10.0, 700.0]), np.array([0, 1, 500])
    cn = multiplier(mu, steps, DT, "cn")
    r = (1 - 1j * mu * DT / 4) / (1 + 1j * mu * DT / 4)
    np.testing.assert_allclose(cn[:, 1], r, atol=1e-15)
    np.testing.assert_allclose(cn[:, 2], r ** 500, atol=1e-12)
    exact = multiplier(mu, steps, DT, "exact")
    np.testing.assert_allclose(exact[:, 2], np.exp(-0.25j * mu), atol=1e-15)
    assert np.all(cn[:, 0] == 1) and np.abs(cn[1, 2] - exact[1, 2]) < 1e-5     # CN error is O((μΔt)² μt)
    with pytest.raises(ValueError):
        multiplier(mu, np.array([0.5]), DT, "cn")
    with pytest.raises(ValueError):
        multiplier(mu, steps, DT, "euler")
    verts = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 3.0, 0]])
    np.testing.assert_allclose(centroid(verts, np.array([1.0, 1.0, 0.0])), [0, 0, 0])
    np.testing.assert_allclose(geo_feature(verts, np.array([1.0, 1.0, 0.0]), 2.0), [0.5, 0.5, 1.5])
    with pytest.raises(ValueError):
        geo_feature(verts, np.ones(3), 0.0)
    signs = canonical_signs(np.array([[0.1, -0.9], [-0.8, 0.2]]))
    np.testing.assert_array_equal(signs, [[-0.1, 0.9], [0.8, -0.2]])


@pytest.mark.slow
@pytest.mark.skipif(not (SHAPES / "manifest.json").exists(), reason="no stage-3 dataset in data/shapes")
def test_dataset_ritz_basis_matches_the_stored_trajectories():
    """On three spheres and three tori (two round), in and out of distribution:

    the exact evolution in (E, μ) misses the data by their own CN error, the CN multiplier
    in (E, μ) is within 2.5e-3, and the stored lumped pairs are off by at least 1.5 %.
    024_sphere has the largest singular value of y and 137_torus (round, thin) the
    largest driver-to-driver change of U.
    """
    config = json.loads((SHAPES / "manifest.json").read_text())["config"]
    steps = np.linspace(0, round(config["final_time"] / config["dt"]), config["output_steps"]).round().astype(int)
    dt = config["dt"]
    assert steps.tolist() == list(range(0, 501, 50))
    for shape in ("000_sphere", "001_torus", "024_sphere", "137_torus", "180_sphere", "199_torus"):
        with np.load(SHAPES / shape / "mesh.npz") as stored:
            mesh = {key: stored[key] for key in ("verts", "faces", "mass", "evals", "evecs")}
        ritz = ritz_polar_basis(mesh["verts"], mesh["faces"], mesh["evecs"])
        mass = mesh["mass"]
        assert 690 < ritz.mu[-1] < 712 and ritz.singular_values.max() < 1.06    # max on the dataset 1.055 (024)
        for prefix in ("", "ood_"):
            if not (SHAPES / shape / f"{prefix}inputs.npy").exists():
                continue
            inputs = np.load(SHAPES / shape / f"{prefix}inputs.npy")
            outputs = np.load(SHAPES / shape / f"{prefix}outputs.npy")
            cn_error = np.load(SHAPES / shape / f"{prefix}diagnostics.npy")[:, 1]
            for p in range(len(inputs)):
                psi0 = inputs[p, :, 0] + 1j * inputs[p, :, 1]
                target = outputs[p, :, 1:, 0] + 1j * outputs[p, :, 1:, 1]
                exact = propagate(ritz.E, ritz.mu, mass, psi0, steps[1:], dt, "exact")
                assert mass_relative_error(exact, target, mass).max() <= cn_error[p] + 1e-3, (shape, prefix, p)
                assert spectral_oracle_error(ritz.E, ritz.mu, mass, psi0, target, steps[1:], dt).max() < 2.5e-3
                lumped = propagate(mesh["evecs"], mesh["evals"], mass, psi0, steps[1:], dt, "exact")
                assert mass_relative_error(lumped, target, mass).max() > 1.5e-2, (shape, prefix, p)


@pytest.fixture(scope="module")
def coarse_sphere():
    verts, faces = icosphere(2)
    L, mass = cotan_laplacian(verts, faces), vertex_areas(verts, faces)
    _, phi = laplacian_eigenpairs(L, mass, 30)
    return verts, faces, L, mass, phi


def test_singular_value_guard_fires_on_both_sides(coarse_sphere):
    verts, faces, L, mass, phi = coarse_sphere
    with pytest.raises(ValueError, match="singular values"):
        ritz_polar_basis(verts, faces, phi, blended=sp.diags(2 * mass))        # B = 2I: s = 1/√2 < 1
    with pytest.raises(ValueError, match="singular values"):
        ritz_polar_basis(verts, faces, phi, blended=sp.diags(mass / 2))        # B = I/2: s = √2 > √(8/5)


def test_ritz_values_are_clipped_at_zero(coarse_sphere):
    verts, faces, L, mass, phi = coarse_sphere
    shifted = L - 1e-6 * sp.diags(mass)          # μ_0 = -1e-6 before the clip
    assert ritz_polar_basis(verts, faces, phi, L=shifted).mu[0] == 0.0


def test_shape_tolerance_and_driver_checks(coarse_sphere):
    verts, faces, L, mass, phi = coarse_sphere
    with pytest.raises(ValueError, match="evecs must be"):
        ritz_polar_basis(verts, faces, phi[:-1])
    with pytest.raises(ValueError, match="orthonormal"):
        ritz_polar_basis(verts, faces, phi * (1 + 2e-6))         # Gram error 4e-6 > 1e-6
    with pytest.raises(ValueError):
        ritz_polar_basis(verts, faces, phi, driver="no-such-driver")


def test_mass_relative_error_uses_the_mass():
    target, pred = np.array([[1.0], [1.0]]), np.array([[1.0], [0.0]])
    np.testing.assert_allclose(mass_relative_error(pred, target, np.array([1.0, 3.0])), [np.sqrt(3 / 4)])


def test_relative_gap_uses_the_mean_of_the_pair():
    mu = np.array([0.0, 1.0, 1.5])               # gap 0.5 / 1.25 = 0.4 (0.33 if divided by the larger value)
    np.testing.assert_array_equal(eigen_clusters(mu, 0.39), [0, 1, 2])
    np.testing.assert_array_equal(eigen_clusters(mu, 0.41), [0, 1, 1])


def test_input_validation():
    with pytest.raises(ValueError):
        multiplier(np.ones(2), np.array([-1, 2]), 1e-3, "cn")
    for dt in (0.0, -1e-3, np.inf, np.nan):
        with pytest.raises(ValueError):
            multiplier(np.ones(2), np.array([1]), dt, "cn")
    for scale in (np.inf, np.nan, -1.0):
        with pytest.raises(ValueError):
            geo_feature(np.zeros((2, 3)), np.ones(2), scale)
    for mu, tau in ((np.array([0.0, 1.0]), -1e-3), (np.array([0.0, 2.0, 1.0]), 1e-3),
                    (np.array([0.0, np.nan, 1.0]), 1e-3), (np.array([]), 1e-3)):
        with pytest.raises(ValueError):
            eigen_clusters(mu, tau)


def test_oracle_is_zero_on_its_own_cn_prediction(coarse_sphere):
    verts, faces, L, mass, phi = coarse_sphere
    ritz = ritz_polar_basis(verts, faces, phi)
    psi0 = phi[:, 1:10] @ np.random.default_rng(0).standard_normal(9)
    steps = np.arange(50, 501, 50)
    target = propagate(ritz.E, ritz.mu, mass, psi0, steps, DT, "cn")
    assert spectral_oracle_error(ritz.E, ritz.mu, mass, psi0, target, steps, DT).max() < 1e-12
    exact = propagate(ritz.E, ritz.mu, mass, psi0, steps, DT, "exact")
    assert spectral_oracle_error(ritz.E, ritz.mu, mass, psi0, exact, steps, DT).max() > 1e-6
