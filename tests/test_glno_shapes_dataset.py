"""Stage-3 dataset generator: one directory per shape, packet rules, resumable parallel writes, manifest identity."""
import dataclasses
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from schrodinger_glno import shapes_dataset
from schrodinger_glno.geodesic import HeatGeodesics, cut_gaussian_packet
from schrodinger_glno.initial_conditions import max_wavenumber, spectral_fraction
from schrodinger_glno.mesh import blended_mass, cotan_laplacian, mean_edge_length, vertex_areas, vertex_normals
from schrodinger_glno.sampling import PARAM_COLUMNS, allowed_wavenumber, direction_from_angle
from schrodinger_glno.shapes import (GENUS, SphereDeformation, TorusDeformation, _base_torus, build_shape, check_gates,
                                     tube_log_radius)
from schrodinger_glno.shapes_dataset import (DIAGNOSTIC_COLUMNS, SPLITS, ShapesConfig, dataset_summary,
                                             generate_shapes_dataset, round_torus_geometry, shapes_dataset_identity,
                                             tube_section_radii, write_shape)
from schrodinger_glno.solver import build_cn_matrices, solve_CN

# Six coarse shapes (2562-vertex spheres, ~1000-vertex tori): 0 sphere and 1 round torus (train), 2 sphere (val),
# 3 torus, 4 sphere and 5 round torus (test, with OOD packets). The packet rules are tightened so that each one
# rejects some draws of seed 0: the tube rule on the OOD packets of both test tori (σ ≥ 0.18 against tubes of
# radius 0.39), the leak rule on torus 1 and the spectral gate (100 modes) on the spheres.
TINY = ShapesConfig(train_shapes=2, val_shapes=1, test_shapes=3, packets_per_shape=2, ood_packets_per_shape=2,
                    round_torus_every=2, sphere_subdivisions=4, torus_target_edge=0.12, min_wavenumber=5.0,
                    k_eig=100, final_time=0.05, dt=0.005, min_points_per_wavelength=5.0, k0_range=(0.5, 2.0),
                    sigma_range=(0.15, 0.2), ood_k0_range=(2.0, 3.0), ood_sigma_range=(0.18, 0.2),
                    min_tube_sigmas=2.1, max_leak=2e-3, min_spectral_fraction=0.995, min_ood_spectral_fraction=0.99)
COLUMN = {name: k for k, name in enumerate(DIAGNOSTIC_COLUMNS)}
MESH_KEYS = {"verts", "faces", "mass", "evals", "evecs", "geo_feat", "t_out"}
ARRAYS = ("inputs.npy", "outputs.npy", "norms.npy", "params.npy", "diagnostics.npy")
SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_shapes_dataset.py"
BLAS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
# A caller's main script: the config as JSON in argv[2], the directory in argv[1].
BUILD = '''
import json
import sys
from schrodinger_glno.shapes_dataset import ShapesConfig, generate_shapes_dataset

if __name__ == "__main__":
    print(generate_shapes_dataset(ShapesConfig(**json.loads(sys.argv[2])), sys.argv[1], workers=2)["identity"])
'''


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shapes") / "data"
    messages = []
    manifest = generate_shapes_dataset(TINY, directory, workers=2, progress=messages.append)
    return directory, manifest, messages


def _record(directory, index):
    return json.loads((directory / TINY.shape_id(index) / "shape.json").read_text())


def _packet_sets(directory, index):
    """``(name, prefix, k0_range, sigma_range, min_fraction)`` of each packet set stored for shape ``index``."""
    return [(name, prefix, *TINY.packet_rules(name)) for name, prefix, _, _ in TINY.packet_sets(index)]


def _rebuild_packet(verts, faces, mass, normals, row, geodesics):
    center, phi, k0, sigma = int(row[0]), *row[1:]
    return cut_gaussian_packet(verts, mass, normals, center, k0, sigma, direction_from_angle(normals[center], phi),
                               geodesics.distance(center))


def _lumped_norms(states, mass):
    return np.sqrt(np.sum(mass[:, None] * np.abs(states)**2, axis=0))


def _edit_json(path, keys, value):
    """Set ``data[keys[0]][keys[1]]...`` of a JSON file to ``value``; return the original text."""
    original = path.read_text()
    data = target = json.loads(original)
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(data))
    return original


def test_default_config_has_the_adopted_numbers():
    config = ShapesConfig()
    assert (config.train_shapes, config.val_shapes, config.test_shapes, config.n_shapes) == (160, 20, 20, 200)
    assert (config.packets_per_shape, config.ood_packets_per_shape, config.k_eig) == (5, 5, 700)
    assert config.sphere_subdivisions == 5 and config.sphere_relax and config.sphere_l_max == 4
    assert config.torus_target_edge == 0.0305 and config.torus_alpha_range == (0.45, 0.55)
    assert config.k0_range == (1.0, 6.0) and config.sigma_range == (0.15, 0.2)
    assert config.ood_k0_range == (6.0, 8.0) and config.ood_sigma_range == (0.18, 0.2)
    assert config.area == 4 * np.pi and config.steps == 501
    np.testing.assert_allclose(config.output_times(), np.linspace(0, 0.5, 11))
    # The mesh gate covers both packet rectangles, to the 0.003 of the rounded 16.33.
    assert max_wavenumber(config.k0_range[1], config.sigma_range[0]) <= config.min_wavenumber
    assert max_wavenumber(config.ood_k0_range[1], config.ood_sigma_range[0]) == pytest.approx(config.min_wavenumber,
                                                                                              abs=0.004)
    families = [config.family(i) for i in range(200)]
    assert families.count("sphere") == families.count("torus") == 100
    assert sum(config.is_round(i) for i in range(200)) == 25
    for split in SPLITS:
        members = [i for i in range(200) if config.split(i) == split]
        assert sum(config.family(i) == "sphere" for i in members) == len(members) // 2
        assert any(config.is_round(i) for i in members)                 # every split has exact references
    options = config.shape_options(1)
    assert config.is_round(1) and options["a_max"] == options["b_max"] == 0 and options["tau_range"] == (0, 0)
    assert config.shape_options(0)["gates"] == {"min_wavenumber": 16.33, "min_curvature_radius": 0.18}
    assert config.shape_options(3)["tau_range"] == (0.0, 0.075) and config.shape_options(3)["sigmas"] == (0.15, 0.2)
    assert [name for name, *_ in config.packet_sets(0)] == ["train"]
    assert [name for name, *_ in config.packet_sets(199)] == ["test", "ood"] and config.shape_id(199) == "199_torus"


@pytest.mark.parametrize("fields", [
    {"dt": 0.003},                                    # not a whole number of steps
    {"output_steps": 7},                              # 6 does not divide 500 steps
    {"output_steps": 5.5},                            # an integer field with a fractional value
    {"ood_k0_range": (5.0, 9.0)},                     # overlaps the training k0 range
    {"sigma_range": (0.0, 0.2)},
    {"k0_range": (3.0, 2.0)},
    {"k0_range": (2.0, 2.0)},
    {"sphere_tau_range": (0.2, 0.1)},
    {"torus_alpha_range": (0.5, 1.0)},
    {"train_shapes": -1},
    {"train_shapes": 0, "val_shapes": 0, "test_shapes": 0},
    {"packets_per_shape": 0},
    {"seed": -1},
    {"sphere_relax": 1},
    {"area": 1.0},                                    # every shape is scaled to 4π
    {"final_time": np.inf},
    {"cutoff_inner": 9.0},
    {"max_leak": 0.0},
    {"min_spectral_fraction": 1.5},
    {"k_eig": 0},
])
def test_config_rejects_bad_settings(fields):
    with pytest.raises(ValueError):
        ShapesConfig(**fields)


def test_config_round_trips_through_json_and_accepts_numpy_integers():
    assert ShapesConfig(**json.loads(json.dumps(dataclasses.asdict(TINY)))) == TINY
    config = ShapesConfig(train_shapes=np.int64(4), output_steps=11.0, k0_range=[1, 2])
    assert type(config.train_shapes) is int and type(config.output_steps) is int and config.k0_range == (1.0, 2.0)


def test_shape_plan_of_the_tiny_config():
    plan = [(TINY.shape_id(i), TINY.split(i), TINY.is_round(i)) for i in range(TINY.n_shapes)]
    assert plan == [("000_sphere", "train", False), ("001_torus", "train", True), ("002_sphere", "val", False),
                    ("003_torus", "test", False), ("004_sphere", "test", False), ("005_torus", "test", True)]
    with pytest.raises(ValueError):
        TINY.family(6)
    # min_wavenumber is resolved at 5 points per wavelength, which is 2.5 at mesh_quality's 10.
    assert TINY.shape_options(0)["gates"]["min_wavenumber"] == 2.5


def test_every_shape_has_its_files_with_the_right_shapes(tiny):
    directory, manifest, messages = tiny
    assert manifest is not None and not list(directory.glob("*.partial"))
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        ["config.json", "manifest.json"] + [TINY.shape_id(i) for i in range(TINY.n_shapes)])
    assert sum("vertices)" in message for message in messages) == TINY.n_shapes        # one timing line per shape
    for index in range(TINY.n_shapes):
        folder, record = directory / TINY.shape_id(index), _record(directory, index)
        test_shape = TINY.split(index) == "test"
        expected = {"mesh.npz", "shape.json", *ARRAYS} | ({f"ood_{name}" for name in ARRAYS} if test_shape else set())
        assert {p.name for p in folder.iterdir()} == expected                        # OOD arrays only on test shapes
        assert set(record["packets"]) == ({"test", "ood"} if test_shape else {TINY.split(index)})
        with np.load(folder / "mesh.npz") as mesh:
            assert set(mesh.files) == MESH_KEYS
            n_vertices = len(mesh["verts"])
            assert n_vertices == record["n_verts"] == record["quality"]["n_verts"]
            assert mesh["evecs"].shape == (n_vertices, TINY.k_eig) and mesh["evecs"].dtype == np.float32
            assert mesh["evals"].dtype == np.float64 and np.all(np.diff(mesh["evals"]) >= 0)
            assert mesh["mass"].sum() == pytest.approx(4 * np.pi, rel=1e-12)
            np.testing.assert_array_equal(mesh["t_out"], TINY.output_times())
        for name, prefix, *_ in _packet_sets(directory, index):
            n = TINY.packets_per_shape if name != "ood" else TINY.ood_packets_per_shape
            inputs, outputs = np.load(folder / f"{prefix}inputs.npy"), np.load(folder / f"{prefix}outputs.npy")
            assert inputs.shape == (n, n_vertices, 2) and inputs.dtype == np.float32
            assert outputs.shape == (n, n_vertices, TINY.output_steps, 2) and outputs.dtype == np.float32
            np.testing.assert_array_equal(inputs, outputs[:, :, 0])
            for array, width in (("norms", TINY.output_steps), ("params", len(PARAM_COLUMNS)),
                                 ("diagnostics", len(DIAGNOSTIC_COLUMNS))):
                stored = np.load(folder / f"{prefix}{array}.npy")
                assert stored.shape == (n, width) and stored.dtype == np.float64


def test_manifest_is_strict_json_and_its_identity_verifies_every_file(tiny, tmp_path):
    directory, manifest, _ = tiny

    def reject(constant):
        raise ValueError(constant)

    stored = json.loads((directory / "manifest.json").read_text(), parse_constant=reject)   # no NaN or inf
    assert stored == json.loads(json.dumps(manifest))
    assert shapes_dataset_identity(directory) == manifest["identity"]
    assert [entry["index"] for entry in stored["shapes"]] == list(range(TINY.n_shapes))
    assert stored["counts"] == {"shapes": {"train": 2, "val": 1, "test": 3}, "families": {"sphere": 3, "torus": 3},
                                "round_tori": 2, "packets": {"train": 4, "val": 2, "test": 6, "ood": 6}}
    assert stored["diagnostic_columns"] == list(DIAGNOSTIC_COLUMNS) and stored["param_columns"] == list(PARAM_COLUMNS)
    data_files = {str(p.relative_to(directory)) for p in directory.glob("*/*") if p.name != "shape.json"}
    assert set(stored["sha256"]) == data_files
    for entry in stored["shapes"]:
        record = _record(directory, entry["index"])
        assert {"id", "family", "split", "round", "exact_torus", "n_verts", "k_allowed", "quality"} <= set(entry)
        # what rebuilds and refines the shape is in the identity
        assert entry["seed"] == [TINY.seed, entry["index"]] and entry["deformation"] == record["deformation"]
        assert entry["quality"] == record["quality"] and {"scale", "offset"} <= set(entry["quality"])
    copy = tmp_path / "copy"
    shutil.copytree(directory, copy)
    params = copy / "003_torus" / "ood_params.npy"
    original = params.read_bytes()
    np.save(params, np.load(params) + 1e-9)
    assert shapes_dataset_identity(copy, verify=False) == manifest["identity"]
    with pytest.raises(ValueError, match="ood_params"):
        shapes_dataset_identity(copy)
    params.write_bytes(original)
    assert shapes_dataset_identity(copy) == manifest["identity"]
    # shape.json is not hashed (it holds the timings), but verify checks it against the manifest.
    record = copy / "003_torus" / "shape.json"
    for keys, value in [(("deformation", "a", 0), 0.5), (("quality", "offset", 0), 0.0), (("seed",), [1, 3]),
                        (("sha256", "inputs.npy"), "0" * 64)]:
        original = _edit_json(record, keys, value)
        assert shapes_dataset_identity(copy, verify=False) == manifest["identity"]
        with pytest.raises(ValueError, match="003_torus/shape.json"):
            shapes_dataset_identity(copy)
        record.write_text(original)
    _edit_json(record, ("timings", "total"), 0.0)
    assert shapes_dataset_identity(copy) == manifest["identity"]
    # Any edit of the manifest's shapes or hashes changes its identity.
    for keys, value in [(("shapes", 0, "attempts"), 2), (("shapes", 3, "deformation", "a", 0), 0.5),
                        (("shapes", 3, "quality", "offset", 0), 0.0), (("sha256", "003_torus/inputs.npy"), "0" * 64)]:
        original = _edit_json(copy / "manifest.json", keys, value)
        with pytest.raises(ValueError, match="identity"):
            shapes_dataset_identity(copy, verify=False)
        (copy / "manifest.json").write_text(original)


def test_params_are_in_their_ranges_and_resolved_by_the_mesh(tiny):
    directory, *_ = tiny
    for index in range(TINY.n_shapes):
        record = _record(directory, index)
        with np.load(directory / record["id"] / "mesh.npz") as mesh:
            verts, faces = mesh["verts"], mesh["faces"]
        k_allowed = allowed_wavenumber(mean_edge_length(verts, faces), TINY.min_points_per_wavelength)
        assert record["k_allowed"] == pytest.approx(k_allowed, rel=1e-12)
        # The gate read mesh_quality's k_allowed at 10 points per wavelength: the same mesh resolution.
        assert record["k_allowed"] >= TINY.min_wavenumber
        assert record["quality"]["k_allowed"] * 10 == pytest.approx(k_allowed * TINY.min_points_per_wavelength)
        for name, prefix, k0_range, sigma_range, _ in _packet_sets(directory, index):
            center, phi, k0, sigma = np.load(directory / record["id"] / f"{prefix}params.npy").T
            assert np.all(center == np.round(center)) and center.min() >= 0 and center.max() < len(verts)
            assert np.all((0 <= phi) & (phi < 2 * np.pi))
            assert np.all((k0_range[0] <= k0) & (k0 <= k0_range[1]))
            assert np.all((sigma_range[0] <= sigma) & (sigma <= sigma_range[1]))
            assert np.all(max_wavenumber(k0, sigma) <= k_allowed)
            stats = record["packets"][name]
            assert stats["k0"] == [k0.min(), k0.max()] and 0 < stats["resolved_fraction"] <= 1


def test_norms_are_conserved(tiny):
    directory, *_ = tiny
    for index in range(TINY.n_shapes):
        for _, prefix, *_ in _packet_sets(directory, index):
            norms = np.load(directory / TINY.shape_id(index) / f"{prefix}norms.npy")
            np.testing.assert_allclose(norms[:, 0], 1, atol=1e-12)            # ψ0 normalised in the blended mass
            assert np.abs(norms - norms[:, :1]).max() < 1e-12


def test_every_packet_satisfies_the_rules(tiny):
    directory, *_ = tiny
    rejected = {"thin": 0, "leak": 0, "spectral": 0}
    for index in range(TINY.n_shapes):
        record = _record(directory, index)
        folder = directory / record["id"]
        with np.load(folder / "mesh.npz") as mesh:
            verts, faces, lumped, evecs = mesh["verts"], mesh["faces"], mesh["mass"], mesh["evecs"].astype(float)
        normals, blended = vertex_normals(verts, faces), blended_mass(verts, faces)
        geodesics = HeatGeodesics(verts, faces)
        sections = None
        if record["family"] == "torus":
            deformation, quality = TorusDeformation(**record["deformation"]), record["quality"]
            sections = tube_section_radii(deformation, quality, verts)
            # The thinnest radius of the section through a vertex is at most the vertex's own (to the 128-angle
            # grid in v), and the thinnest of all is the tube's r_min.
            _, u, v = _base_torus(deformation, quality["target_edge"])
            own = quality["scale"] * deformation.minor_radius * np.exp(tube_log_radius(deformation, u, v))
            assert np.all(sections <= own * (1 + 1e-3))
            assert sections.min() == pytest.approx(quality["r_min"], rel=1e-3)
            if record["round"]:
                np.testing.assert_allclose(sections, record["exact_torus"]["minor_radius"], rtol=1e-12)
        for name, prefix, _, _, min_fraction in _packet_sets(directory, index):
            params = np.load(folder / f"{prefix}params.npy")
            diagnostics = np.load(folder / f"{prefix}diagnostics.npy")
            inputs = np.load(folder / f"{prefix}inputs.npy")
            for row, diag, stored in zip(params, diagnostics, inputs):
                center, sigma = int(row[0]), row[3]
                assert sections is None or sections[center] >= TINY.min_tube_sigmas * sigma            # rule a
                distance = geodesics.distance(center)
                density = lumped * np.exp(-np.sum((verts - verts[center])**2, axis=1) / (2 * sigma**2))
                leak = density[distance > TINY.leak_radius * sigma].sum() / density.sum()
                assert leak <= TINY.max_leak and leak == pytest.approx(diag[COLUMN["leak_fraction"]], rel=1e-12)  # b
                psi0, removed = _rebuild_packet(verts, faces, blended, normals, row, geodesics)
                np.testing.assert_allclose(stored[:, 0] + 1j * stored[:, 1], psi0, rtol=0,
                                           atol=1e-6 * np.abs(psi0).max())
                assert removed == pytest.approx(diag[COLUMN["removed_fraction"]], rel=1e-9, abs=1e-15)
                fraction = diag[COLUMN["spectral_fraction"]]
                assert fraction >= min_fraction                                                          # rule c
                assert spectral_fraction(psi0, lumped, evecs) == pytest.approx(fraction, abs=1e-6)   # float32 basis
            stats = record["packets"][name]
            counts = diagnostics[:, [COLUMN["thin_draws"], COLUMN["leak_draws"], COLUMN["spectral_draws"]]].sum(axis=0)
            assert list(counts) == [stats["rejected"][key] for key in ("thin", "leak", "spectral")]
            assert stats["draws"] == len(params) + counts.sum()
            for key in rejected:
                rejected[key] += stats["rejected"][key]
    assert all(count > 0 for count in rejected.values()), rejected       # every rule was exercised


def test_stored_trajectories_and_cn_estimates_reproduce(tiny):
    directory, *_ = tiny
    folder = directory / TINY.shape_id(3)
    with np.load(folder / "mesh.npz") as mesh:
        verts, faces, lumped = mesh["verts"], mesh["faces"], mesh["mass"]
    L, blended, normals = cotan_laplacian(verts, faces), blended_mass(verts, faces), vertex_normals(verts, faces)
    geodesics, zero = HeatGeodesics(verts, faces), np.zeros(len(verts))
    coarse, fine = build_cn_matrices(L, blended, zero, TINY.dt), build_cn_matrices(L, blended, zero, TINY.dt / 2)
    indices = TINY.output_indices()
    for prefix in ("", "ood_"):
        outputs = np.load(folder / f"{prefix}outputs.npy")
        diagnostics = np.load(folder / f"{prefix}diagnostics.npy")
        for row, stored, diag in zip(np.load(folder / f"{prefix}params.npy"), outputs, diagnostics):
            psi0, _ = _rebuild_packet(verts, faces, blended, normals, row, geodesics)
            trajectory = solve_CN(psi0, *coarse, TINY.steps, indices)
            half = solve_CN(psi0, *fine, 2 * TINY.steps - 1, 2 * indices)
            np.testing.assert_allclose(stored[..., 0] + 1j * stored[..., 1], trajectory, rtol=0,
                                       atol=1e-6 * np.abs(trajectory).max())
            estimate = 4 / 3 * np.max(_lumped_norms(trajectory - half, lumped) / _lumped_norms(half, lumped))
            assert diag[COLUMN["cn_error_estimate"]] == pytest.approx(estimate, rel=1e-9)
            assert 0 < estimate < 0.05


def test_shapes_rebuild_from_their_records_and_pass_the_gates(tiny):
    directory, *_ = tiny
    for index in range(TINY.n_shapes):
        record = _record(directory, index)
        family, quality = record["family"], record["quality"]
        deformation = (SphereDeformation if family == "sphere" else TorusDeformation)(**record["deformation"])
        options = {key: quality[key] for key in ("subdivisions", "relax", "target_edge") if key in quality}
        verts, faces, rebuilt = build_shape(deformation, sigmas=TINY.shape_options(index)["sigmas"], **options)
        with np.load(directory / record["id"] / "mesh.npz") as mesh:
            assert np.array_equal(verts, mesh["verts"]) and np.array_equal(faces, mesh["faces"])
            assert np.array_equal(mesh["mass"], vertex_areas(verts, faces))
        assert rebuilt == quality and record["attempts"] == len(record["rejections"]) + 1
        assert check_gates(quality, GENUS[family], quality if family == "torus" else None,
                           **TINY.shape_options(index)["gates"]) == (True, [])


def test_round_tori_have_no_deformation_and_record_their_exact_geometry(tiny):
    directory, manifest, _ = tiny
    for index in range(1, TINY.n_shapes, 2):
        record = _record(directory, index)
        d = TorusDeformation(**record["deformation"])
        assert TINY.torus_alpha_range[0] <= d.alpha <= TINY.torus_alpha_range[1]
        assert record["round"] == TINY.is_round(index) == (manifest["shapes"][index]["exact_torus"] is not None)
        if not record["round"]:
            assert any(d.a) and any(d.b) and any(d.c) and record["exact_torus"] is None
            continue
        assert all(x == 0 for x in d.a + d.b + d.c) and d.tube_rms == 0
        geometry = record["exact_torus"]
        assert geometry == round_torus_geometry(d, record["quality"]) == manifest["shapes"][index]["exact_torus"]
        with np.load(directory / record["id"] / "mesh.npz") as mesh:
            x = mesh["verts"] - geometry["center"]
        R, r = geometry["major_radius"], geometry["minor_radius"]
        np.testing.assert_allclose((np.hypot(x[:, 0], x[:, 1]) - R)**2 + x[:, 2]**2, r**2, rtol=1e-12)
        assert record["quality"]["hole"] == pytest.approx(2 * (R - r), rel=1e-12)


def test_identity_does_not_depend_on_workers_and_survives_a_resume(tiny, tmp_path):
    directory, manifest, _ = tiny
    again = tmp_path / "again"
    single = generate_shapes_dataset(TINY, again, workers=1)
    assert single["identity"] == manifest["identity"] and single["sha256"] == manifest["sha256"]   # the same bits
    # An interrupted run: one shape lost, another one half-written, the stale manifest still there.
    shutil.rmtree(again / "003_torus")
    (again / "003_torus.partial").mkdir()
    (again / "003_torus.partial" / "inputs.npy").write_bytes(b"half")
    (again / "001_torus.partial").mkdir()
    messages = []
    assert generate_shapes_dataset(TINY, again, workers=1, families=["sphere"], progress=messages.append) is None
    assert not (again / "manifest.json").exists() and not list(again.glob("*.partial"))   # nothing complete-looking
    assert sum("cleared" in message for message in messages) == 2
    resumed = generate_shapes_dataset(TINY, again, workers=2, progress=messages.append)
    assert resumed["identity"] == manifest["identity"] == shapes_dataset_identity(again)
    assert resumed["sha256"] == manifest["sha256"]
    assert any("building 1 (sphere, torus)" in message for message in messages)          # only the lost shape
    assert messages[-1].startswith("manifest written")
    assert generate_shapes_dataset(TINY, again)["identity"] == manifest["identity"]          # nothing left to build


def test_family_filter_builds_only_that_family_and_writes_no_manifest(tiny, tmp_path):
    directory, manifest, _ = tiny
    target = tmp_path / "tori"
    assert generate_shapes_dataset(TINY, target, workers=2, families=["torus"]) is None
    assert sorted(p.name for p in target.iterdir()) == ["001_torus", "003_torus", "005_torus", "config.json"]
    assert dataset_summary(target)["shapes"]["built"] == 3
    assert generate_shapes_dataset(TINY, target, workers=2, families=["sphere"])["identity"] == manifest["identity"]


def test_a_different_config_or_a_foreign_directory_is_refused(tiny, tmp_path):
    directory, manifest, _ = tiny
    with pytest.raises(FileExistsError, match="different config"):
        generate_shapes_dataset(dataclasses.replace(TINY, seed=1), directory)
    with pytest.raises(FileExistsError, match="different config"):
        generate_shapes_dataset(dataclasses.replace(TINY, packets_per_shape=3), directory, families=["sphere"])
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "notes.txt").write_text("not a dataset")
    with pytest.raises(FileExistsError, match="not empty"):
        generate_shapes_dataset(TINY, tmp_path / "other")
    with pytest.raises(ValueError):
        generate_shapes_dataset(TINY, tmp_path / "new", families=["cube"])
    with pytest.raises(ValueError):
        generate_shapes_dataset(TINY, tmp_path / "new", workers=0)
    assert not (tmp_path / "new").exists() and shapes_dataset_identity(directory) == manifest["identity"]


def test_a_second_run_is_refused_while_another_one_holds_the_directory(tiny, tmp_path):
    directory, manifest, _ = tiny
    target = tmp_path / "data"
    shutil.copytree(directory, target)
    shutil.rmtree(target / "003_torus")
    (target / "003_torus.partial").mkdir()                  # the shape the other run is building
    other = os.open(target, os.O_RDONLY)                     # the lock a running generate_shapes_dataset holds
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(BlockingIOError, match="another run"):
            generate_shapes_dataset(TINY, target, families=["sphere"])
        refused = subprocess.run([sys.executable, str(SCRIPT), "--data-dir", str(target), "--family", "sphere"],
                                 capture_output=True, text=True)
        assert refused.returncode == 1 and "another run" in refused.stderr and "Traceback" not in refused.stderr
        assert (target / "003_torus.partial").is_dir() and (target / "manifest.json").is_file()   # left untouched
    finally:
        os.close(other)
    assert generate_shapes_dataset(TINY, target)["identity"] == manifest["identity"]


def test_an_interrupted_shape_leaves_only_its_partial_directory(tiny, tmp_path, monkeypatch):
    directory, manifest, _ = tiny
    target = tmp_path / "data"
    target.mkdir()
    shutil.copy(directory / "config.json", target)

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(shapes_dataset, "_write_packets", interrupted)     # after mesh.npz is written
    with pytest.raises(RuntimeError, match="interrupted"):
        write_shape(TINY, target, 2)
    monkeypatch.undo()
    assert not (target / "002_sphere").exists() and (target / "002_sphere.partial" / "mesh.npz").is_file()
    for index in set(range(TINY.n_shapes)) - {2}:
        shutil.copytree(directory / TINY.shape_id(index), target / TINY.shape_id(index))
    messages = []
    resumed = generate_shapes_dataset(TINY, target, progress=messages.append)
    assert "cleared 002_sphere.partial, left by an interrupted run" in messages
    assert any("building 1 (sphere, torus)" in message for message in messages)
    assert resumed["identity"] == manifest["identity"] == shapes_dataset_identity(target)


def test_a_failed_shape_stops_the_run_without_a_complete_looking_directory(tiny, tmp_path):
    directory, manifest, _ = tiny
    target = tmp_path / "data"
    one_draw = dataclasses.replace(TINY, max_packet_draws=1)               # every rule rejects some first draw
    with pytest.raises(RuntimeError, match=r"shape \d{3}_(sphere|torus): RuntimeError\(.no \w+ packet passed"):
        generate_shapes_dataset(one_draw, target, workers=2)
    assert not (target / "manifest.json").exists() and list(target.glob("*.partial"))
    for folder in target.iterdir():
        if folder.is_dir() and folder.suffix != ".partial":                # no draw rejected: the fixture's shape
            record = json.loads((folder / "shape.json").read_text())
            assert all(manifest["sha256"][f"{folder.name}/{name}"] == digest
                       for name, digest in record["sha256"].items())


def test_a_worker_that_dies_stops_the_run_instead_of_hanging(tmp_path):
    script = tmp_path / "build.py"
    script.write_text("import os\nfrom schrodinger_glno import shapes_dataset\n"
                      "shapes_dataset.write_shape = lambda *args: os._exit(9)      # run by every worker: it dies\n"
                      + BUILD)
    run = subprocess.run([sys.executable, str(script), str(tmp_path / "data"), json.dumps(dataclasses.asdict(TINY))],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 1 and "a worker process died" in run.stderr
    assert not (tmp_path / "data" / "manifest.json").exists()


def test_workers_run_single_threaded_blas_whatever_the_callers_threads(tiny, tmp_path):
    directory, manifest, _ = tiny
    script = tmp_path / "build.py"
    script.write_text(BUILD)
    run = subprocess.run([sys.executable, str(script), str(tmp_path / "data"), json.dumps(dataclasses.asdict(TINY))],
                         capture_output=True, text=True, env={**os.environ, **dict.fromkeys(BLAS, "4")})
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == [manifest["identity"]]


def test_a_main_module_that_sets_the_blas_threads_is_refused(tmp_path):
    script = tmp_path / "build.py"
    script.write_text('import os\nos.environ["OPENBLAS_NUM_THREADS"] = "4"   # unguarded: every worker runs it\n'
                      + BUILD)
    run = subprocess.run([sys.executable, str(script), str(tmp_path / "data"), json.dumps(dataclasses.asdict(TINY))],
                         capture_output=True, text=True)
    assert run.returncode == 1 and "BLAS is not single-threaded" in run.stderr
    assert not (tmp_path / "data" / "manifest.json").exists() and not list((tmp_path / "data").glob("0*"))


def test_summary_counts_shapes_packets_and_rejections(tiny):
    directory, manifest, _ = tiny
    summary = dataset_summary(directory)
    assert summary["identity"] == manifest["identity"]
    shapes = summary["shapes"]
    assert shapes["built"] == shapes["total"] == 6 and shapes["by_split"] == {"train": 2, "val": 1, "test": 3}
    assert shapes["round_tori"] == 2 and shapes["sphere"]["built"] == shapes["torus"]["built"] == 3
    packets = summary["packets"]
    assert {name: packets[name]["packets"] for name in packets} == {"train": 4, "val": 2, "test": 6, "ood": 6}
    assert packets["ood"]["thin"] > 0 and packets["ood"]["diagnostics"]["spectral_fraction"][0] >= 0.99
    assert all(packets[name]["max_norm_drift"] < 1e-12 for name in packets)
    json.dumps(summary, allow_nan=False)


def test_command_line_script_builds_resumes_and_refuses_other_settings(tiny, tmp_path):
    directory, manifest, _ = tiny
    settings = tmp_path / "tiny.json"
    settings.write_text(json.dumps({k: v for k, v in dataclasses.asdict(TINY).items() if k != "train_shapes"}))
    command = [sys.executable, str(SCRIPT), "--data-dir", str(tmp_path / "data"), "--config", str(settings),
               "--train", "2", "--workers", "2"]
    run = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "4"})
    assert run.returncode == 0, run.stderr
    summary = json.loads(run.stdout[run.stdout.index("\n{") + 1:])
    assert summary["complete"] and summary["identity"] == manifest["identity"]
    assert "vertices)" in run.stdout and "manifest written" in run.stdout
    rerun = subprocess.run(command, capture_output=True, text=True)                    # resuming a complete dataset
    assert rerun.returncode == 0 and "nothing to build" in rerun.stdout
    refused = subprocess.run(command + ["--seed", "1"], capture_output=True, text=True)
    assert refused.returncode != 0 and "different config" in refused.stderr
