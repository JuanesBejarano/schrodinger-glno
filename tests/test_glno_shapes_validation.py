"""Stage-3 validator and figure on a tiny dataset: it passes, tampering is found, the error estimates run."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pytest
from schrodinger_glno.data_utils import _digest
from schrodinger_glno.geodesic import HeatGeodesics
from schrodinger_glno.mesh import blended_mass, vertex_normals
from schrodinger_glno.reference import torus_exact_solution
from schrodinger_glno.sampling import direction_from_angle
from schrodinger_glno.shapes_dataset import (DIAGNOSTIC_COLUMNS, ShapesConfig, _identity, generate_shapes_dataset,
                                             shapes_dataset_identity)
from scripts import validate_shapes_dataset as validate
from scripts import visualize_shapes_dataset as visualize

# The six coarse shapes of tests/test_glno_shapes_dataset.py: 0 sphere and 1 round torus (train), 2 sphere (val),
# 3 torus, 4 sphere and 5 round torus (test, with OOD packets), two packets per set, 11 states up to t = 0.05.
TINY = ShapesConfig(train_shapes=2, val_shapes=1, test_shapes=3, packets_per_shape=2, ood_packets_per_shape=2,
                    round_torus_every=2, sphere_subdivisions=4, torus_target_edge=0.12, min_wavenumber=5.0,
                    k_eig=100, final_time=0.05, dt=0.005, min_points_per_wavelength=5.0, k0_range=(0.5, 2.0),
                    sigma_range=(0.15, 0.2), ood_k0_range=(2.0, 3.0), ood_sigma_range=(0.18, 0.2),
                    min_tube_sigmas=2.1, max_leak=2e-3, min_spectral_fraction=0.995, min_ood_spectral_fraction=0.99)
SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_shapes_dataset.py"
COLUMN = {name: k for k, name in enumerate(DIAGNOSTIC_COLUMNS)}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shapes") / "data"
    generate_shapes_dataset(TINY, directory, workers=2)
    return directory


@pytest.fixture
def copy(tiny, tmp_path):
    shutil.copytree(tiny, tmp_path / "copy")
    return tmp_path / "copy"


def _rehash(directory: Path, shape_id: str, name: str) -> None:
    """Rewrite the SHA-256 of one changed file in shape.json and in the manifest, with its identity."""
    digest = _digest(directory / shape_id / name)
    record = json.loads((directory / shape_id / "shape.json").read_text())
    record["sha256"][name] = digest
    (directory / shape_id / "shape.json").write_text(json.dumps(record))
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["sha256"][f"{shape_id}/{name}"] = digest
    manifest["identity"] = _identity(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))


def _rewrite(directory: Path, shape_id: str, name: str, change) -> None:
    """Apply ``change`` in place to one array file of a shape (a .npy array or the dict of mesh.npz), then rehash."""
    path = directory / shape_id / name
    if name.endswith(".npz"):
        with np.load(path) as stored:
            arrays = {key: stored[key] for key in stored.files}
        change(arrays)
        np.savez(path, **arrays)
    else:
        array = np.load(path)
        change(array)
        np.save(path, array)
    _rehash(directory, shape_id, name)


def _scale(index, factor: float):
    """A change for ``_rewrite``: multiply ``array[index]`` by ``factor``."""
    def change(array):
        array[index] *= factor
    return change


def _edit_entry(directory: Path, shape_id: str, change) -> None:
    """Apply ``change`` to a shape's record in shape.json and in the manifest, and sign the manifest again."""
    record = json.loads((directory / shape_id / "shape.json").read_text())
    change(record)
    (directory / shape_id / "shape.json").write_text(json.dumps(record))
    manifest = json.loads((directory / "manifest.json").read_text())
    change(next(entry for entry in manifest["shapes"] if entry["id"] == shape_id))
    manifest["identity"] = _identity(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))


def _shape(tiny: Path, shape_id: str) -> tuple[dict, dict]:
    """Manifest entry of a shape and its mesh, with the blended mass and the normals the validator adds."""
    entry = next(entry for entry in json.loads((tiny / "manifest.json").read_text())["shapes"]
                 if entry["id"] == shape_id)
    with np.load(tiny / shape_id / "mesh.npz") as stored:
        mesh = {key: stored[key] for key in stored.files}
    return entry, mesh | {"blended": blended_mass(mesh["verts"], mesh["faces"]),
                          "normals": vertex_normals(mesh["verts"], mesh["faces"])}


def test_validator_passes_on_a_tiny_dataset(tiny, tmp_path, capsys):
    out = tmp_path / "report.json"
    report = validate.main(["--data-dir", str(tiny), "--output", str(out)])
    assert report["passed"] and report["problems"] == [] and "Todo en orden" in capsys.readouterr().out
    manifest = json.loads((tiny / "manifest.json").read_text())
    assert report["identity"] == shapes_dataset_identity(tiny) and report["files"]["files"] == len(manifest["sha256"])

    def reject(constant):
        raise ValueError(constant)

    stored = json.loads(out.read_text(), parse_constant=reject)                          # strict JSON
    assert [shape["id"] for shape in stored["shapes"]] == [TINY.shape_id(i) for i in range(TINY.n_shapes)]
    for shape in report["shapes"]:
        assert shape["mesh"]["surface_residual"] < 1e-12 and shape["mesh"]["mass_error"] == 0
        assert shape["basis"]["norm"] < 1e-6 and shape["basis"]["rayleigh"] < 1e-9        # float32 eigenvectors
        for numbers in shape["sets"].values():
            assert numbers["norm_start"] < 1e-12 and numbers["norm_drift"] < 1e-12
            assert numbers["norm_recomputed"] < 1e-6                                      # from float32 outputs
        evolved = [row["evolve_error"] for row in shape["packets"] if "evolve_error" in row]
        assert len(evolved) == len(shape["sets"]) and max(evolved) < 1e-7                # --evolve 1; measured 4e-8
        checked = [row["spectral"] for row in shape["packets"] if "spectral" in row]
        assert len(checked) == 2 * len(shape["sets"])                                    # --spectral 2 per set
        assert max(abs(check["difference"]) for check in checked) < 1e-7                 # measured ≤ 1e-8
        assert max(check["input_error"] for check in checked) < 1e-6
        assert max(abs(check["removed_difference"]) for check in checked) < 1e-12
        if shape["round"]:
            assert shape["mesh"]["exact_torus"]["offset"] < 1e-4 and shape["mesh"]["exact_torus"]["fit"] < 1e-12
    summary = report["summary"]
    assert {key: group["packets"] for key, group in summary.items()} == {
        "sphere": 8, "torus": 10, "train": 4, "val": 2, "test": 6, "ood": 6, "all": 18}
    assert summary["all"]["shapes"] == 6 and summary["ood"]["shapes"] == 3 and summary["all"]["shape_draws"] >= 6
    assert summary["ood"]["packet_rejections"]["thin"] > 0                               # the tube rule worked
    assert summary["all"]["packet_draws"] == 18 + sum(summary["all"]["packet_rejections"].values())
    assert summary["all"]["refine"] == summary["all"]["exact"] == {"n": 0, "median": None, "max": None}


def test_a_tampered_file_fails_the_command_line_check(copy):
    path = copy / "003_torus" / "ood_diagnostics.npy"
    diagnostics = np.load(path)
    diagnostics[0, 3] *= 1.5                             # leak_fraction: only its hash can tell
    np.save(path, diagnostics)
    run = subprocess.run([sys.executable, str(SCRIPT), "--data-dir", str(copy), "--spectral", "0"],
                         capture_output=True, text=True)
    assert run.returncode == 1, run.stderr
    assert "003_torus/ood_diagnostics.npy: su SHA-256 difiere del manifiesto" in run.stdout
    assert "FALLA: 1 problema" in run.stdout


def test_an_unsigned_manifest_and_another_config_json_are_found(copy):
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["shapes"][3]["attempts"] += 1                                  # not signed again
    (copy / "manifest.json").write_text(json.dumps(manifest))
    stored = json.loads((copy / "config.json").read_text())
    stored["config"]["dt"] = 0.01
    (copy / "config.json").write_text(json.dumps(stored))
    report = validate.main(["--data-dir", str(copy), "--shapes", "002_sphere", "--spectral", "0", "--evolve", "0"])
    assert report["files"]["problems"] == ["manifest.json: Dataset manifest identity is invalid",
                                           "config.json: no coincide con la configuración del manifiesto"]
    assert report["shapes"][0]["problems"] == [] and not report["passed"]


def test_a_corrupted_output_is_found_even_with_rewritten_hashes(copy):
    _rewrite(copy, "002_sphere", "outputs.npy", _scale(np.s_[1, :, -1], 1.01))    # last state of packet 1, 1 % large
    shapes_dataset_identity(copy)                        # the hashes agree again
    report = validate.main(["--data-dir", str(copy), "--shapes", "002_sphere", "--spectral", "0"])
    assert not report["passed"] and report["files"]["problems"] == []
    assert report["problems"] == [
        "002_sphere: paquetes val: ψᴴMψ recalculada de outputs difiere de norms.npy en 2.0e-02"]


def test_a_norm_preserving_corruption_is_found_by_evolving_the_input(copy):
    def corrupt(outputs):
        state = validate.as_complex(outputs[0, :, -1]) * np.exp(0.3j)       # a phase error in packet 0
        outputs[0, :, -1, 0], outputs[0, :, -1, 1] = state.real, state.imag
        outputs[1, :, [4, 5]] = outputs[1, :, [5, 4]]                       # two saved times swapped in packet 1

    _rewrite(copy, "002_sphere", "outputs.npy", corrupt)
    quiet = validate.main(["--data-dir", str(copy), "--shapes", "002_sphere", "--evolve", "0"])
    assert quiet["passed"]                                                   # ψᴴMψ and the rest cannot see it
    report = validate.main(["--data-dir", str(copy), "--shapes", "002_sphere", "--evolve", "2"])
    problems = report["shapes"][0]["problems"]
    assert [problem.split(" en ")[0] for problem in problems] == [
        f"paquetes val {i}: outputs difiere de Crank–Nicolson desde inputs" for i in (0, 1)]
    assert all(row["evolve_error"] > 1e-2 for row in report["shapes"][0]["packets"])


def test_an_edited_shape_record_is_found(copy):
    record = json.loads((copy / "003_torus" / "shape.json").read_text())   # not hashed: it holds the timings
    record["deformation"]["c"][0] += 0.01
    record["quality"]["min_angle"] += 1
    record["seed"] = [7, 3]
    record["sha256"]["params.npy"] = "0" * 64
    (copy / "003_torus" / "shape.json").write_text(json.dumps(record))
    report = validate.main(["--data-dir", str(copy), "--shapes", "3", "--spectral", "0"])
    # The other checks read the manifest's copy, which the identity covers, so they pass.
    assert sorted(report["shapes"][0]["problems"]) == [
        "shape.json: deformation difiere del manifiesto", "shape.json: quality[min_angle] difiere del manifiesto",
        "shape.json: seed difiere del manifiesto", "shape.json: sha256 difiere del manifiesto"]


def test_an_edited_and_signed_entry_is_found_by_the_other_checks(copy):
    def deform(record):
        record["deformation"]["c"][0] += 0.01
        record["quality"]["min_angle"] += 1
        record["seed"] = [7, 3]

    def move(record):
        record["exact_torus"]["center"][0] += 1e-3
        record["exact_torus"]["major_radius"] *= 1.001

    _edit_entry(copy, "003_torus", deform)
    _edit_entry(copy, "005_torus", move)
    shapes_dataset_identity(copy)
    report = validate.main(["--data-dir", str(copy), "--shapes", "3", "5", "--spectral", "0", "--evolve", "0"])
    assert report["files"]["problems"] == []
    deformed, moved = (shape["problems"] for shape in report["shapes"])
    assert {"forma: seed no es el de la configuración", "malla: min_angle recalculado difiere del registro",
            "malla: r_min recalculado difiere del registro"} <= set(deformed)
    assert any(problem.startswith("malla: vértices a") for problem in deformed)
    assert [problem.split(" a ")[0] for problem in moved] == [
        "toro de revolución: exact_torus recalculado de su deformación difiere del registro",
        "toro de revolución: centro", "toro de revolución: vértices"]


def test_tampered_arrays_with_rewritten_hashes_are_found(copy):
    tampered = {
        "000_sphere": ("inputs.npy", _scale(0, 1 + 1e-3),
                       ["paquetes train: el estado en t = 0 no es el de entrada",
                        "paquetes train 0: outputs difiere de Crank–Nicolson desde inputs en 1.0e-03",
                        "paquetes train 0: ψ0 reconstruido de params difiere de la entrada (1.0e-03)"]),
        "002_sphere": ("params.npy", _scale(np.s_[1, 2], 10.0),                       # k0 of packet 1
                       ["paquetes val: parámetros fuera de rango en [1]"]),
        "004_sphere": ("diagnostics.npy", _scale(np.s_[1, COLUMN["spectral_fraction"]], 0.5),
                       ["paquetes test: diagnósticos no finitos o fracción espectral bajo 0.995"]),
    }
    for shape_id, (name, change, _) in tampered.items():
        _rewrite(copy, shape_id, name, change)

    def mesh(arrays):
        arrays["evals"][-1] *= 1.01
        arrays["mass"][0] *= 1.001
        arrays["t_out"][-1] += 1e-3
        arrays["geo_feat"][0] += 1e-6

    _rewrite(copy, "003_torus", "mesh.npz", mesh)
    report = validate.main(["--data-dir", str(copy), "--spectral", "1"])
    problems = {shape["id"]: shape["problems"] for shape in report["shapes"]}
    assert report["files"]["problems"] == [] and problems["001_torus"] == problems["005_torus"] == []
    for shape_id, (_, _, expected) in tampered.items():
        assert sorted(problems[shape_id]) == sorted(expected)
    assert {"mesh.npz: t_out no son los tiempos guardados de la configuración",
            "mesh.npz: geo_feat no es la distancia al vértice medio",
            "malla: la masa concentrada guardada no es vertex_areas"} <= set(problems["003_torus"])
    assert any(problem.startswith("base float32: error relativo de Rayleigh") for problem in problems["003_torus"])


def test_failing_gates_are_reported(tiny, monkeypatch):
    monkeypatch.setattr(validate, "check_gates", lambda *args, **kwargs: (False, ["min angle 1° < 30°"]))
    report = validate.main(["--data-dir", str(tiny), "--shapes", "0", "--spectral", "0", "--evolve", "0"])
    assert report["problems"] == ["000_sphere: malla: no pasa la compuerta: min angle 1° < 30°"]


def test_round_tori_against_the_exact_solution(tiny, capsys):
    report = validate.main(["--data-dir", str(tiny), "--shapes", "001_torus", "005_torus", "--exact", "--refine", "1",
                            "--spectral", "0"])
    assert report["passed"]
    for shape in report["shapes"]:
        rows = shape["packets"]
        assert len(rows) == (2 if shape["id"] == "001_torus" else 4) and all("exact" in row for row in rows)
        for row in rows:
            exact = row["exact"]
            assert exact["start"] <= exact["error"] < 0.03                               # measured 1.1-1.5 %
            removed = row["removed_fraction"]
            assert 0 < removed < 1e-6 and exact["cut"] == pytest.approx(np.sqrt(removed) + removed, rel=1e-12)
    # --shapes names the round tori, so --refine 1 refines the first one: the estimate is near the exact error.
    refined = [row for shape in report["shapes"] for row in shape["packets"] if "refine" in row]
    assert len(refined) == 2 and all(0.5 < row["refine"]["error"] / row["exact"]["error"] < 2 for row in refined)
    assert "refinamiento/exacta" in capsys.readouterr().out
    assert report["summary"]["torus"]["exact"]["n"] == 6 and report["summary"]["ood"]["exact"]["n"] == 2


def test_exact_comparison_measures_a_known_error(tiny):
    """States equal to the exact solution about the recorded centre, but for ``δ`` at one vertex at t = 0."""
    entry, mesh = _shape(tiny, "001_torus")
    verts, mass, normals, times = mesh["verts"], mesh["mass"], mesh["normals"], mesh["t_out"]
    params = np.load(tiny / "001_torus" / "params.npy")[:1]
    center, phi, k0, sigma = int(params[0, 0]), *params[0, 1:]
    x = verts - np.asarray(entry["exact_torus"]["center"])
    states = torus_exact_solution(x, x[center], direction_from_angle(normals[center], phi), k0, sigma, times,
                                  entry["exact_torus"]["major_radius"], entry["exact_torus"]["minor_radius"])
    vertex, delta = int(np.argmin(mass)), 0.05
    states[vertex, 0] += delta
    diagnostics = np.zeros((1, len(DIAGNOSTIC_COLUMNS)))
    diagnostics[0, COLUMN["removed_fraction"]] = 1e-6
    rows = validate.exact_comparison(entry, {"train": {"params": params, "states": states[None],
                                                       "diagnostics": diagnostics}}, verts, mass, normals, times)
    expected = np.sqrt(mass[vertex]) * delta / np.sqrt(np.sum(mass * np.abs(states[:, 0])**2))
    assert rows["train", 0]["error"] == rows["train", 0]["start"] == pytest.approx(expected, rel=1e-9)
    assert rows["train", 0]["cut"] == pytest.approx(1e-3 + 1e-6, rel=1e-12)


def test_richardson_errors_on_synthetic_trajectories():
    """c = u + A + B, f = u + A/4 + B and c½ = u + A + B/4: a mesh error A and a time error B, both second order."""
    rng = np.random.default_rng(0)
    n, t = 50, 4
    mass = rng.uniform(0.5, 2.0, n)
    u, a, b = (rng.normal(size=(n, t)) + 1j * rng.normal(size=(n, t)) for _ in range(3))
    a, b = 0.03 * a, 0.01 * b
    c, f, c_half = u + a + b, u + a / 4 + b, u + a + b / 4

    def relative(error, size):
        return np.max(np.sqrt(np.sum(mass[:, None] * np.abs(error)**2, axis=0) /
                              np.sum(mass[:, None] * np.abs(size)**2, axis=0)))

    estimate = validate.richardson_errors(c, f, c_half, mass)
    assert estimate == pytest.approx({"error": relative(a + b, c), "space": relative(a, c), "time": relative(b, c),
                                      "cn_estimate": relative(b, c_half)}, rel=1e-12)


def test_the_fine_packet_is_the_same_function_of_x(tiny):
    """On the refined mesh, a params row gives the stored packet at the stored vertices up to a real factor
    (its blended normalisation) and its geodesic cutoff (removed fraction below 1e-7)."""
    entry, mesh = _shape(tiny, "003_torus")
    n, mass = len(mesh["verts"]), mesh["mass"]
    fine = validate.fine_mesh(entry, mesh)
    assert len(fine["verts"]) > 3.9 * n and np.array_equal(fine["verts"][:n], mesh["verts"])
    geodesics = HeatGeodesics(mesh["verts"], mesh["faces"])
    for row in np.load(tiny / "003_torus" / "params.npy"):
        stored, _ = validate.rebuild_packet(TINY, mesh, row, geodesics.distance(int(row[0])))
        refined, _ = validate.rebuild_packet(TINY, fine, row, fine["geodesics"].distance(int(row[0])))
        factor = np.sum(mass * stored.conj() * refined[:n]) / np.sum(mass * np.abs(stored)**2)
        residual = np.sqrt(np.sum(mass * np.abs(refined[:n] - factor * stored)**2) / np.sum(mass * np.abs(stored)**2))
        # Measured: residual ≤ 3e-5, |arg| 4e-18; with the fine mesh's own normals, 3e-4-2e-3 and 2e-4-9e-4.
        assert abs(np.angle(factor)) < 1e-12 and abs(factor - 1) < 0.02 and residual < 1e-4


def test_refinement_estimate_on_a_deformed_torus(copy):
    _rewrite(copy, "003_torus", "ood_diagnostics.npy", _scale(np.s_[1, COLUMN["cn_error_estimate"]], 1.5))
    report = validate.main(["--data-dir", str(copy), "--shapes", "003_torus", "--refine", "1", "--spectral", "0"])
    shape, = report["shapes"]
    assert [problem.split(" (")[0] for problem in report["problems"]] == [
        "003_torus: paquetes ood 1: cn_error_estimate no se reproduce"]                  # the tampered one
    assert shape["refinement"]["fine_verts"] > 3.9 * shape["n_verts"]
    assert len(shape["packets"]) == 4                                                    # test and OOD packets
    for row in shape["packets"]:
        estimate = row["refine"]
        if (row["set"], row["packet"]) != ("ood", 1):
            assert estimate["cn_estimate"] == pytest.approx(row["cn_error_estimate"], abs=1e-6)   # the stored one
        assert estimate["time"] == pytest.approx(estimate["cn_estimate"], rel=1e-2)
        assert abs(estimate["space"] - estimate["time"]) > 0.2 * estimate["time"]        # 1.4-1.7 % and 0.5-0.9 %
        assert 0 < estimate["error"] <= estimate["space"] + estimate["time"] and estimate["error"] < 0.05
    assert report["summary"]["torus"]["refine"]["n"] == 4


def test_refined_shapes_spread_over_each_family_and_skip_round_tori_unless_named():
    entries = [{"id": TINY.shape_id(i), "family": TINY.family(i), "round": TINY.is_round(i)} for i in range(6)]
    assert validate._refined(entries, 1, explicit=False) == {"000_sphere", "003_torus"}
    assert validate._refined(entries, 2, explicit=False) == {"000_sphere", "004_sphere", "003_torus"}
    assert validate._refined(entries, 1, explicit=True) == {"000_sphere", "001_torus"}
    assert validate._refined(entries, 0, explicit=False) == set()


def test_figure_is_written(tiny, tmp_path):
    assert visualize.default_shapes(tiny) == ["000_sphere", "002_sphere", "003_torus", "001_torus"]
    out = visualize.main(["--data-dir", str(tiny), "--times", "0", "0.025", "0.05",
                          "--output", str(tmp_path / "f.png")])
    height, width, _ = plt.imread(out).shape
    assert height > 1.5 * width > 1000                                                   # four rows of three panels
    narrow = visualize.main(["--data-dir", str(tiny), "--shapes", "005_torus", "--set", "ood", "--times", "0", "0.05",
                             "--output", str(tmp_path / "g.png")])
    assert plt.imread(narrow).shape[1] == width                                          # room for the title
    row = visualize.load_row(tiny, "005_torus", "ood", 1, [0.0, 0.05])
    np.testing.assert_allclose(row["times"], [0.0, 0.05])
    assert row["density"].shape == (len(row["verts"]), 2) and row["kind"] == "toro de revolución"
    with pytest.raises(SystemExit):
        visualize.load_row(tiny, "000_sphere", "ood", 0, [0.0])                         # OOD packets: test shapes only


def test_surface_panel_draws_only_the_faces_toward_the_camera(tiny):
    with np.load(tiny / "000_sphere" / "mesh.npz") as mesh:
        verts, faces = mesh["verts"], mesh["faces"]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    visualize.surface_panel(ax, verts, faces, np.ones(len(verts)), PowerNorm(0.5, 0, 1), np.array([0.0, 0.0, 1.0]))
    assert 0.4 < len(ax.collections[0].get_facecolor()) / len(faces) < 0.6              # the near half of a sphere
    plt.close(fig)
