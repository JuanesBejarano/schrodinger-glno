"""Stage-3 store: per-shape network inputs on one device, padding by family, batches of one family, fingerprint."""
import dataclasses
import json
import math
from pathlib import Path
import shutil
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from schrodinger_glno.glno import shapes_data
from schrodinger_glno.glno.augment import augment_batch, gauge_generator
from schrodinger_glno.glno.data import glno_config
from schrodinger_glno.glno.layers import GLNONet
from schrodinger_glno.glno.shapes_data import (GROUPS, SETS, ShapesPreprocess, ShapeStore, family_batches,
                                               fingerprint_mismatch, group_means, shape_arrays,
                                               training_geo_scale)
from schrodinger_glno.glno.spectral import (centroid, eigen_clusters, geo_feature, ritz_polar_basis,
                                            spectral_oracle_error)
from schrodinger_glno.shapes_dataset import ShapesConfig, generate_shapes_dataset

ROOT = Path(__file__).resolve().parents[1]
SMALL_DATA, FULL_DATA = ROOT / "data" / "shapes_small", ROOT / "data" / "shapes"
# Six coarse shapes, 100 modes, 11 saved times: 0 sphere and 1 round torus (train), 2 sphere (val), 3 torus,
# 4 sphere and 5 round torus (test, with out-of-distribution packets); two packets per set. The spheres have
# 2562 vertices and the tori 1006, 1006 and 1015, so tori 1 and 3 are padded by 9 rows.
TINY = ShapesConfig(train_shapes=2, val_shapes=1, test_shapes=3, packets_per_shape=2, ood_packets_per_shape=2,
                    round_torus_every=2, sphere_subdivisions=4, torus_target_edge=0.12, min_wavenumber=5.0,
                    k_eig=100, final_time=0.05, dt=0.005, min_points_per_wavelength=5.0, k0_range=(0.5, 2.0),
                    sigma_range=(0.15, 0.2), ood_k0_range=(2.0, 3.0), ood_sigma_range=(0.18, 0.2),
                    min_tube_sigmas=2.1, max_leak=2e-3, min_spectral_fraction=0.995, min_ood_spectral_fraction=0.99)
PRE = ShapesPreprocess(eval_scale=100.0)                # 100 modes: μ/100 in [0, ~1]
PACKETS = {"train": 4, "val": 2, "test": 6, "ood": 6}
NET = {"blocks": 2, "C_width": 8, "encoder_hidden_layers": [8], "decoder_hidden_layers": [8],
       "mlp_hidden_layers": [8]}
TENSORS = ("input", "labels", "mass", "evals", "evecs", "geo_feat")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shapes_data") / "data"
    generate_shapes_dataset(TINY, directory, workers=2)
    return directory


@pytest.fixture(scope="module")
def store(tiny):
    return ShapeStore(tiny, PRE)


def _mesh(directory, index):
    with np.load(directory / TINY.shape_id(index) / "mesh.npz") as stored:
        return {key: stored[key] for key in stored.files}


def _packets(directory, index, name):
    prefix = "ood_" if name == "ood" else ""
    return tuple(np.load(directory / TINY.shape_id(index) / f"{prefix}{kind}.npy")
                 for kind in ("inputs", "outputs", "params", "diagnostics"))


def _expected_table():
    """``(shape, set, packet)`` of every packet, in the order the store promises."""
    return [(index, name, packet) for index in range(TINY.n_shapes)
            for name, _, count, _ in TINY.packet_sets(index) for packet in range(count)]


def _net(c_out, aperiodic, seed=0):
    config = glno_config(C_out=c_out, k_eig=TINY.k_eig, C_in=2, **NET)
    config["aperiodic"] = aperiodic
    torch.manual_seed(seed)
    return GLNONet(config).eval()


def test_preprocess_has_the_adopted_numbers_and_checks_them():
    pre = ShapesPreprocess()
    assert (pre.basis, pre.eval_scale, pre.cluster_gap, pre.geo_scale, pre.drop_initial) == ("ritz", 700.0, 1e-4,
                                                                                            1.4030, True)
    assert pre.amplitude == math.sqrt(8 * math.pi)
    for bad in ({"basis": "consistent"}, {"eval_scale": 0.0}, {"geo_scale": -1.0}, {"amplitude": float("nan")},
                {"cluster_gap": -1e-4}, {"cluster_gap": float("inf")}):
        with pytest.raises(ValueError):
            ShapesPreprocess(**bad)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pre.eval_scale = 1.0


def test_store_lists_every_packet_in_order(store, tiny):
    expected = _expected_table()
    assert len(store) == len(expected) == sum(PACKETS.values())
    table = store.table
    assert [(s, SETS[c], p) for s, c, p in zip(table["shape"], table["set"], table["packet"])] == expected
    for name, count in PACKETS.items():
        assert len(store.rows(sets=name)) == count
    assert set(table["shape"][store.rows(sets="ood")]) == {3, 4, 5}        # only on the test shapes
    manifest = json.loads((tiny / "manifest.json").read_text())
    by_index = {entry["index"]: entry for entry in manifest["shapes"]}
    for row, (index, name, packet) in enumerate(expected):
        entry = by_index[index]
        assert shapes_data.FAMILIES[table["family"][row]] == entry["family"]
        assert bool(table["round"][row]) == entry["round"] and table["n_verts"][row] == entry["n_verts"]
        _, _, params, diagnostics = _packets(tiny, index, name)
        np.testing.assert_array_equal(store.params[row], params[packet])
        np.testing.assert_array_equal(store.diagnostics[row], diagnostics[packet])
    assert store.identity == manifest["identity"] and store.n_max == {"sphere": 2562, "torus": 1015}
    assert store.k_eig == 100 and store.c_out == 20 and store.dt == TINY.dt
    assert store.steps.tolist() == list(range(1, 11))
    np.testing.assert_allclose(store.times, np.linspace(0.005, 0.05, 10))
    assert [s["id"] for s in store.shapes] == [TINY.shape_id(i) for i in range(6)]
    # slots and positions number the shapes and the packets of each family from 0, without gaps
    for family, n_shapes, n_packets in (("sphere", 3, 8), ("torus", 3, 10)):
        rows = store.rows(family=family)
        assert table["position"][rows].tolist() == list(range(n_packets))
        assert sorted(set(table["slot"][rows])) == list(range(n_shapes))
        assert [s["slot"] for s in store.shapes if s["family"] == family] == list(range(n_shapes))
    floats = 3 * 2562 * (100 + 2) + 3 * 100 + 8 * 2562 * 22 + 3 * 1015 * (100 + 2) + 3 * 100 + 10 * 1015 * 22
    assert store.nbytes == 4 * floats + 8 * 2 * 3 * 100                     # float32 tensors + int64 groups


def test_rows_filters(store):
    everything = np.arange(len(store))
    np.testing.assert_array_equal(store.rows(), everything)
    table = store.table
    np.testing.assert_array_equal(store.rows(family="torus"), everything[table["family"] == 1])
    np.testing.assert_array_equal(store.rows(is_round=True), everything[np.isin(table["shape"], [1, 5])])
    np.testing.assert_array_equal(store.rows(is_round=False), everything[~np.isin(table["shape"], [1, 5])])
    np.testing.assert_array_equal(store.rows(sets="test", is_round=np.bool_(False)),
                                  everything[np.isin(table["shape"], [3, 4]) & (table["set"] == 2)])
    np.testing.assert_array_equal(store.rows(sets=("test", "ood"), family="sphere"), everything[table["shape"] == 4])
    np.testing.assert_array_equal(store.rows(shapes=[5, 0]), everything[np.isin(table["shape"], [0, 5])])
    np.testing.assert_array_equal(store.rows(sets="ood", shapes=3, is_round=False),
                                  everything[(table["shape"] == 3) & (table["set"] == 3)])
    assert len(store.rows(sets="val", family="torus")) == 0
    for shapes in ({5, 0}, (index for index in (5, 0)), np.array([0, 5]), torch.tensor([5, 0])):
        np.testing.assert_array_equal(store.rows(shapes=shapes), store.rows(shapes=[5, 0]))
    np.testing.assert_array_equal(store.rows(shapes=range(2)), everything[table["shape"] < 2])
    assert len(store.rows(shapes=[])) == 0
    with pytest.raises(ValueError):
        store.rows(sets="validation")
    with pytest.raises(ValueError):
        store.rows(family="creature")
    # nothing is read as something else: "3" is not 3, 3.7 is not 3, "no" is not False
    for bad in ({"shapes": "3"}, {"shapes": 3.7}, {"shapes": [3.7]}, {"shapes": True}, {"shapes": [[3, 4]]},
                {"is_round": "no"}, {"is_round": 0}):
        with pytest.raises(ValueError):
            store.rows(**bad)


def test_batch_holds_what_the_files_hold(store, tiny):
    """Every tensor of every packet against an independent reading of the files."""
    amplitude = PRE.amplitude
    for index in range(TINY.n_shapes):
        mesh = _mesh(tiny, index)
        n, ritz = len(mesh["verts"]), ritz_polar_basis(mesh["verts"], mesh["faces"], mesh["evecs"])
        for name, _, count, _ in TINY.packet_sets(index):
            rows = store.rows(sets=name, shapes=index)
            batch = store.batch(rows)
            assert {key: batch[key].dtype for key in TENSORS} == dict.fromkeys(TENSORS, torch.float32)
            padded = store.n_max[TINY.family(index)]
            assert batch["evecs"].shape == (count, padded, 100) and batch["labels"].shape == (count, padded, 20)
            assert batch["input"].shape == (count, padded, 2) and batch["evals"].shape == (count, 100)
            assert batch["mass"].shape == batch["geo_feat"].shape == (count, padded)
            assert batch["n_verts"].tolist() == [n] * count and batch["rows"].tolist() == rows.tolist()
            inputs, outputs, _, _ = _packets(tiny, index, name)
            for b in range(count):
                np.testing.assert_array_equal(batch["evecs"][b, :n].numpy(), ritz.E)
                np.testing.assert_array_equal(batch["evals"][b].numpy(), (ritz.mu / 100).astype(np.float32))
                np.testing.assert_array_equal(batch["mass"][b, :n].numpy(), mesh["mass"].astype(np.float32))
                np.testing.assert_array_equal(batch["geo_feat"][b, :n].numpy(),
                                              geo_feature(mesh["verts"], mesh["mass"], PRE.geo_scale))
                np.testing.assert_array_equal(batch["clusters"][b].numpy(), eigen_clusters(ritz.mu, 1e-4))
            np.testing.assert_array_equal(batch["input"][:, :n].numpy(), np.float32(amplitude) * inputs)
            np.testing.assert_allclose(batch["input"][:, :n].double().numpy() / amplitude, inputs, rtol=5e-7, atol=1e-30)
            labels = batch["labels"][:, :n].double().numpy().reshape(count, n, 10, 2) / amplitude
            for step in range(10):                                  # every saved time after t = 0, Re and Im
                np.testing.assert_allclose(labels[:, :, step], outputs[:, :, step + 1], rtol=5e-7, atol=1e-30)
    # the centroid matters: the stored geo_feat (distance to the mean vertex) is another feature
    mesh = _mesh(tiny, 3)
    feature = np.linalg.norm(mesh["verts"] - centroid(mesh["verts"], mesh["mass"]), axis=1) / PRE.geo_scale
    np.testing.assert_allclose(store.batch(store.rows(shapes=3)[:1])["geo_feat"][0, :len(feature)], feature, rtol=1e-6)


def test_a_shuffled_batch_of_several_shapes_holds_each_packet_with_its_own_mesh(store, tiny):
    """The batches of training mix the shapes of a family in any order: every item against the files."""
    amplitude = np.float32(PRE.amplitude)
    meshes = {index: _mesh(tiny, index) for index in range(TINY.n_shapes)}
    bases = {index: ritz_polar_basis(mesh["verts"], mesh["faces"], mesh["evecs"]) for index, mesh in meshes.items()}
    for family in ("sphere", "torus"):
        rows = np.random.default_rng(0).permutation(store.rows(family=family))
        shapes = store.table["shape"][rows]
        assert len(set(shapes)) == 3 and np.any(np.diff(shapes) < 0) and np.any(np.diff(rows) < 0)
        batch = store.batch(rows)
        assert batch["rows"].tolist() == rows.tolist()
        oracle = store.oracle_error(batch)
        for b, row in enumerate(rows):
            index, name, packet = shapes[b], SETS[store.table["set"][row]], store.table["packet"][row]
            mesh, ritz = meshes[index], bases[index]
            n = len(mesh["verts"])
            inputs, outputs, _, _ = _packets(tiny, index, name)
            expected = {"evecs": ritz.E, "mass": mesh["mass"].astype(np.float32),
                        "geo_feat": geo_feature(mesh["verts"], mesh["mass"], PRE.geo_scale),
                        "input": amplitude * inputs[packet],
                        "labels": amplitude * outputs[packet][:, 1:].reshape(n, 20)}
            for key, value in expected.items():
                np.testing.assert_array_equal(batch[key][b, :n].numpy(), value, err_msg=f"{family} {key} {b}")
            np.testing.assert_array_equal(batch["evals"][b].numpy(), (ritz.mu / 100).astype(np.float32))
            np.testing.assert_array_equal(batch["clusters"][b].numpy(), eigen_clusters(ritz.mu, 1e-4))
            assert batch["n_verts"][b] == n
            alone = spectral_oracle_error(ritz.E, ritz.mu, mesh["mass"], inputs[packet, :, 0] + 1j * inputs[packet, :, 1],
                                          outputs[packet, :, 1:, 0] + 1j * outputs[packet, :, 1:, 1],
                                          store.steps, store.dt)
            np.testing.assert_allclose(oracle[b], alone, rtol=0, atol=2e-6)


def test_padded_rows_are_exactly_zero(store):
    padded_packets = 0
    for row in range(len(store)):
        batch = store.batch([row])
        n = int(batch["n_verts"][0])
        assert (batch["mass"][0, :n] > 0).all()
        if batch["mass"].shape[1] > n:
            padded_packets += 1
            for key in ("input", "labels", "mass", "geo_feat", "evecs"):
                assert not batch[key][0, n:].any(), key
    assert padded_packets == 6                                          # tori 1 and 3: 2 + 4 packets, 9 rows each
    # the random gauge keeps them at zero
    batch = store.batch(store.rows(shapes=[1, 3]))
    moved = augment_batch(batch, gauge_generator(0, 0, 0))["evecs"]
    assert not moved[:, 1006:].any() and not torch.equal(moved, batch["evecs"])


def test_batches_are_copies(store):
    rows = store.rows(shapes=0)
    first = store.batch(rows)
    for key in TENSORS + ("clusters",):
        first[key].zero_()
    again = store.batch(rows)
    assert all(again[key].any() for key in TENSORS) and again["clusters"].any()


@pytest.mark.parametrize("aperiodic", [False, True])
def test_padded_batch_gives_what_each_mesh_gives_alone(store, aperiodic):
    """Tori of 1006, 1006 and 1015 vertices in one batch, against each one alone and unpadded."""
    rows = np.concatenate([store.rows(shapes=index)[:1] for index in (1, 3, 5)])
    batch = store.batch(rows)
    net = _net(store.c_out, aperiodic)
    with torch.no_grad():
        together = net(batch)
        for b, n in enumerate(batch["n_verts"].tolist()):
            alone = {key: batch[key][b:b + 1, :n] for key in ("input", "mass", "evecs", "geo_feat")}
            alone["evals"] = batch["evals"][b:b + 1]
            weights = batch["mass"][b, :n, None]
            difference = torch.sum(weights * (together[b, :n] - net(alone)[0]) ** 2).sqrt()
            relative = float(difference / torch.sum(weights * together[b, :n] ** 2).sqrt())
            # measured 5e-8 to 2.1e-7 over 400 seeds; ~1e-5 here is conftest's first-call anomaly, not the padding
            assert relative < 1e-6, (b, n, relative)
    assert batch["n_verts"].tolist() == [1006, 1006, 1015]


def test_oracle_from_a_batch_is_the_oracle_from_the_files(store, tiny):
    for index in range(TINY.n_shapes):
        mesh = _mesh(tiny, index)
        ritz = ritz_polar_basis(mesh["verts"], mesh["faces"], mesh["evecs"])
        for name, _, count, _ in TINY.packet_sets(index):
            inputs, outputs, _, _ = _packets(tiny, index, name)
            from_batch = store.oracle_error(store.batch(store.rows(sets=name, shapes=index)))
            assert from_batch.shape == (count, 10)
            for p in range(count):
                expected = spectral_oracle_error(ritz.E, ritz.mu, mesh["mass"], inputs[p, :, 0] + 1j * inputs[p, :, 1],
                                                 outputs[p, :, 1:, 0] + 1j * outputs[p, :, 1:, 1], store.steps, store.dt)
                np.testing.assert_allclose(from_batch[p], expected, rtol=0, atol=2e-6)
                assert 1e-4 < expected.max() < 0.2           # a real error (what 100 modes miss), far above the tolerance
    # the oracle reads the labels' layout: swapping two times is seen
    batch = store.batch(store.rows(shapes=0))
    swapped = batch["labels"].reshape(2, -1, 10, 2).flip(2).reshape(batch["labels"].shape)
    assert store.oracle_error(batch | {"labels": swapped}).max() > 0.05


def test_family_batches(store):
    rows = np.arange(len(store))
    batches = family_batches(store, rows, 3, seed=0, epoch=0)
    assert len(batches) == math.ceil(8 / 3) + math.ceil(10 / 3)
    assert sorted(np.concatenate(batches).tolist()) == rows.tolist()
    for members in batches:
        assert 1 <= len(members) <= 3 and len(set(store.table["family"][members])) == 1
        store.batch(members)
    assert sorted(len(members) for members in batches) == [1, 2, 3, 3, 3, 3, 3]
    again = family_batches(store, rows, 3, seed=0, epoch=0)
    assert all(np.array_equal(a, b) for a, b in zip(batches, again))
    for other in (family_batches(store, rows, 3, seed=0, epoch=1), family_batches(store, rows, 3, seed=1, epoch=0)):
        assert not all(np.array_equal(a, b) for a, b in zip(batches, other))
    # the packets are shuffled inside each family (not only the batches), anew every epoch, and (seed, epoch)
    # is a pair, not a sum
    def companions(seed=None, epoch=0):
        return {frozenset(members.tolist()) for members in family_batches(store, rows, 3, seed=seed, epoch=epoch)}

    assert companions(0, 0) != companions()
    assert companions(0, 0) != companions(0, 1) and companions(0, 0) != companions(1, 0)
    assert companions(0, 1) != companions(1, 0)
    # ... and they depend on which rows are given, not on their order
    reordered = family_batches(store, rows[::-1], 3, seed=0, epoch=0)
    assert all(np.array_equal(a, b) for a, b in zip(batches, reordered))
    # the batches of the two families interleave (seed 0: not all spheres first)
    families = [int(store.table["family"][members[0]]) for members in batches]
    assert families != sorted(families) and families != sorted(families, reverse=True)
    # without a seed: in order, the spheres and then the tori
    ordered = family_batches(store, rows, 4)
    assert np.concatenate(ordered).tolist() == store.rows(family="sphere").tolist() + store.rows(family="torus").tolist()
    assert [len(members) for members in ordered] == [4, 4, 4, 4, 2]
    assert family_batches(store, store.rows(sets="val", family="torus"), 4, seed=0) == []
    assert [members.tolist() for members in family_batches(store, [5, 2], 4)] == [[5], [2]]   # sphere 2, torus 1
    with pytest.raises(ValueError):
        family_batches(store, rows, 0)
    with pytest.raises(ValueError):
        family_batches(store, [0, 0, 1], 2)


def test_groups_and_their_means(store):
    assert tuple(store.group_positions(np.arange(len(store)))) == GROUPS
    # validation is one sphere: no torus and no torus of revolution
    rows = store.rows(sets="val")
    groups = store.group_positions(rows)
    assert {name: positions.tolist() for name, positions in groups.items()} == {
        "select": [0, 1], "sphere": [0, 1], "torus": [], "round": []}
    means = group_means(np.array([1.0, 3.0]), groups)
    assert means == {"select": 2.0, "sphere": 2.0, "torus": None, "round": None}
    per_time = group_means(np.array([[1.0, 10.0], [3.0, 30.0]]), groups)
    np.testing.assert_array_equal(per_time["sphere"], [2.0, 20.0])
    assert per_time["round"] is None
    # test + ood: sphere 4, torus 3 and the torus of revolution 5; positions refer to `rows`, not to packets
    rows = store.rows(sets=("test", "ood"))[::-1].copy()
    groups = store.group_positions(rows)
    shapes = store.table["shape"][rows]
    assert set(shapes[groups["round"]]) == {5} and set(shapes[groups["torus"]]) == {3}
    assert set(shapes[groups["sphere"]]) == {4} and set(shapes[groups["select"]]) == {3, 4}
    assert sorted(np.concatenate([groups["select"], groups["round"]]).tolist()) == list(range(len(rows)))
    empty = store.group_positions([])
    assert group_means(np.zeros(0), empty) == dict.fromkeys(GROUPS)


def test_results_of_family_batches_follow_the_batches_not_the_rows(store):
    rows = store.rows(sets="test")                                      # torus 3, sphere 4, torus of revolution 5
    batches = family_batches(store, rows, 4)
    order = np.concatenate(batches)
    assert order.tolist() != rows.tolist() and sorted(order.tolist()) == rows.tolist()
    # one value per packet, collected batch by batch as an evaluation does: 0 on a sphere, 1 on a torus
    values = np.concatenate([store.table["family"][store.batch(members)["rows"].numpy()] for members in batches]) + 0.0
    expected = {"select": 0.5, "sphere": 0.0, "torus": 1.0, "round": 1.0}
    assert group_means(values, store.group_positions(order)) == expected
    by_packet = np.full(len(store), np.nan)
    by_packet[order] = values
    assert group_means(by_packet[rows], store.group_positions(rows)) == expected
    assert group_means(values, store.group_positions(rows))["sphere"] == 1.0    # the trap: values in another order
    for wrong in (values[:-1], np.append(values, 0.0)):
        with pytest.raises(ValueError, match="values for groups"):
            group_means(wrong, store.group_positions(order))


def test_batch_refuses_bad_rows(store):
    sphere, torus = store.rows(family="sphere")[0], store.rows(family="torus")[0]
    for bad in ([sphere, torus], [], [len(store)], [-1], [0.0], [[0, 1]]):
        with pytest.raises(ValueError):
            store.batch(bad)


def test_a_store_of_some_shapes_pads_and_holds_the_same(store, tiny):
    part = ShapeStore(tiny, PRE, sets=("test", "ood"), shapes=[3])
    assert len(part) == 4 and part.n_max == store.n_max and [s["id"] for s in part.shapes] == ["003_torus"]
    assert part.table["set"].tolist() == [2, 2, 3, 3] and part.table["slot"].tolist() == [0] * 4
    ours, theirs = part.batch(np.arange(4)), store.batch(store.rows(shapes=3))
    for key in TENSORS + ("clusters", "n_verts"):
        assert torch.equal(ours[key], theirs[key]), key
    assert ours["evecs"].shape[1] == 1015                               # padded as the whole dataset
    assert len(ShapeStore(tiny, PRE, sets="val")) == 2
    with pytest.raises(ValueError, match="no packets"):
        ShapeStore(tiny, PRE, sets=("ood",), shapes=[0])
    assert [s["id"] for s in ShapeStore(tiny, PRE, sets="test", shapes=3).shapes] == ["003_torus"]
    with pytest.raises(ValueError, match="unknown packet sets"):
        ShapeStore(tiny, PRE, sets=("train", "validation"))
    with pytest.raises(ValueError, match="shape indices"):
        ShapeStore(tiny, PRE, shapes=[0, 6])
    for bad in ("35", [3.9], 3.7, True):
        with pytest.raises(ValueError, match="integer"):
            ShapeStore(tiny, PRE, shapes=bad)


def test_preprocess_values_other_than_the_defaults_are_used(store, tiny):
    other = ShapeStore(tiny, dataclasses.replace(PRE, amplitude=2.0, geo_scale=0.5, cluster_gap=1e-2), shapes=[1])
    mesh = _mesh(tiny, 1)
    n, ritz = len(mesh["verts"]), ritz_polar_basis(mesh["verts"], mesh["faces"], mesh["evecs"])
    inputs, outputs, _, _ = _packets(tiny, 1, "train")
    batch, default = other.batch([0, 1]), store.batch(store.rows(shapes=1))
    np.testing.assert_array_equal(batch["input"][:, :n].numpy(), np.float32(2.0) * inputs)
    np.testing.assert_array_equal(batch["labels"][:, :n].numpy(), np.float32(2.0) * outputs[:, :, 1:].reshape(2, n, 20))
    np.testing.assert_array_equal(batch["geo_feat"][0, :n].numpy(), geo_feature(mesh["verts"], mesh["mass"], 0.5))
    np.testing.assert_array_equal(batch["clusters"][0].numpy(), eigen_clusters(ritz.mu, 1e-2))
    assert len(batch["clusters"].unique()) < len(default["clusters"].unique())      # a wider gap: fewer groups
    assert torch.equal(batch["evecs"], default["evecs"]) and torch.equal(batch["evals"], default["evals"])
    # the oracle divides the amplitude out again
    np.testing.assert_allclose(other.oracle_error(batch), store.oracle_error(default), rtol=1e-5)


def test_initial_state_can_be_kept(tiny):
    kept = ShapeStore(tiny, dataclasses.replace(PRE, drop_initial=False), shapes=[0])
    assert kept.c_out == 22 and kept.steps.tolist() == list(range(11)) and kept.times[0] == 0
    batch = kept.batch([0])
    np.testing.assert_array_equal(batch["labels"][0, :, :2].numpy(), batch["input"][0].numpy())
    # at t = 0 the oracle is what the 100 modes miss of ψ0, which the dataset recorded
    assert kept.oracle_error(batch)[0, 0] == pytest.approx(math.sqrt(1 - kept.diagnostics[0, 0]), rel=1e-3)


def test_lumped_basis_is_the_stored_one(tiny):
    lumped = ShapeStore(tiny, dataclasses.replace(PRE, basis="lumped"), shapes=[3])
    mesh = _mesh(tiny, 3)
    batch = lumped.batch([0])
    np.testing.assert_array_equal(batch["evecs"][0, :1006].numpy(), mesh["evecs"])
    np.testing.assert_array_equal(batch["evals"][0].numpy(), (np.maximum(mesh["evals"], 0) / 100).astype(np.float32))
    np.testing.assert_array_equal(batch["clusters"][0].numpy(), eigen_clusters(np.maximum(mesh["evals"], 0), 1e-4))
    ritz = ShapeStore(tiny, PRE, shapes=[3])
    assert "basis differs" in fingerprint_mismatch(lumped.fingerprint(), ritz.fingerprint())
    # the stored pairs follow the data worse than the Ritz basis, even here, where most of the error is
    # what 100 modes miss of the packet (the same for both)
    assert lumped.oracle_error(batch)[0, -1] > 1.03 * ritz.oracle_error(ritz.batch([0]))[0, -1]


def _altered_ritz(monkeypatch, change):
    original = shapes_data.ritz_polar_basis

    def altered(verts, faces, evecs):
        return change(original, verts, faces, evecs)

    monkeypatch.setattr(shapes_data, "ritz_polar_basis", altered)


def test_fingerprint_tolerates_the_lapack_driver_and_catches_a_sign(store, tiny, monkeypatch):
    fingerprint = store.fingerprint()
    json.dumps(fingerprint)                                             # plain numbers: it goes into a checkpoint
    assert fingerprint["identity"] == store.identity and len(fingerprint["shapes"]) == 6
    assert all(len(probes) == 20 for probes in fingerprint["shapes"].values())
    assert fingerprint_mismatch(fingerprint, ShapeStore(tiny, PRE).fingerprint()) is None
    part = ShapeStore(tiny, PRE, shapes=[3, 4]).fingerprint()
    assert fingerprint_mismatch(fingerprint, part) is None and fingerprint_mismatch(part, fingerprint) is None
    assert fingerprint_mismatch(part, ShapeStore(tiny, PRE, shapes=[0]).fingerprint()) == "no shape in common"

    # another LAPACK driver: the same basis up to rounding, and the same fingerprint
    _altered_ritz(monkeypatch, lambda ritz, v, f, e: ritz(v, f, e, driver="gv"))
    driver = ShapeStore(tiny, PRE).fingerprint()
    assert fingerprint_mismatch(fingerprint, driver) is None
    changes = [np.abs(np.subtract(fingerprint["shapes"][name], driver["shapes"][name])[:16]).max()
               for name in fingerprint["shapes"]]
    assert max(changes) < 1e-6                                           # far inside the 1e-3 tolerance

    # one eigenvector with the other sign, in one shape
    def flipped(ritz, verts, faces, evecs):
        basis = ritz(verts, faces, evecs)
        if len(verts) != 1015:
            return basis
        signs = np.ones(100)
        signs[37] = -1
        return basis._replace(U=basis.U * signs, E=basis.E * signs.astype(np.float32))

    _altered_ritz(monkeypatch, flipped)
    reason = fingerprint_mismatch(fingerprint, ShapeStore(tiny, PRE).fingerprint())
    assert reason.startswith("005_torus: the basis differs")

    # every single column is seen, with a wide margin over the tolerance
    mesh = _mesh(tiny, 0)
    ritz = ritz_polar_basis(mesh["verts"], mesh["faces"], mesh["evecs"])
    reference = shapes_data._probes(ritz.U, ritz.mu)
    moves = []
    for column in range(100):
        signs = np.ones(100)
        signs[column] = -1
        moves.append(np.abs(shapes_data._probes(ritz.U * signs, ritz.mu) - reference)[:16].max())
    assert min(moves) > 0.05

    # eigenvalues off by 1e-4
    _altered_ritz(monkeypatch, lambda ritz, v, f, e: (lambda b: b._replace(mu=b.mu * (1 + 1e-4)))(ritz(v, f, e)))
    assert "eigenvalues differ" in fingerprint_mismatch(fingerprint, ShapeStore(tiny, PRE, shapes=[0]).fingerprint())

    for key, value in (("identity", "0" * 64), ("ritz_version", -1), ("version", 0), ("k_eig", 99),
                       ("basis", "lumped")):
        assert fingerprint_mismatch(fingerprint, fingerprint | {key: value}).startswith(f"{key} differs")

    def with_probes(probes):
        return fingerprint | {"shapes": fingerprint["shapes"] | {"000_sphere": probes}}

    # the tolerance itself: a probe moved by half of it passes, by twice it does not
    ours = fingerprint["shapes"]["000_sphere"]
    assert fingerprint_mismatch(fingerprint, with_probes([ours[0] + 5e-4] + ours[1:])) is None
    reason = fingerprint_mismatch(fingerprint, with_probes([ours[0] + 2e-3] + ours[1:]))
    assert reason.startswith("000_sphere: the basis differs")
    assert fingerprint_mismatch(fingerprint, with_probes(ours[:16] + [ours[16] * (1 + 5e-7)] + ours[17:])) is None
    for broken in ([float("nan")] * 20, ours[:19], [ours]):
        assert "malformed" in fingerprint_mismatch(fingerprint, with_probes(broken))
    assert "malformed" in fingerprint_mismatch(with_probes(ours[:3]), with_probes(ours[:3]))   # too short on both sides
    for other in (None, {}, fingerprint | {"shapes": None}):            # a checkpoint without a fingerprint
        assert fingerprint_mismatch(fingerprint, other) == fingerprint_mismatch(other, fingerprint) == "not a fingerprint"


def test_store_refuses_files_that_disagree(tiny, tmp_path, monkeypatch):
    def copy(name):
        return Path(shutil.copytree(tiny, tmp_path / name))

    # a manifest that was edited
    directory = copy("manifest")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["config"]["dt"] = 0.004
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        ShapeStore(directory, PRE)

    # inputs that are not the stored states at t = 0; with verify, the hash fails first
    directory = copy("inputs")
    path = directory / "002_sphere" / "inputs.npy"
    np.save(path, np.load(path)[::-1].copy())
    with pytest.raises(ValueError, match="t = 0"):
        ShapeStore(directory, PRE, shapes=[2])
    with pytest.raises(ValueError, match="differs from manifest"):
        ShapeStore(directory, PRE, shapes=[2], verify=True)
    assert len(ShapeStore(directory, PRE, shapes=[0])) == 2             # the other shapes still load

    # packets: one case at a time on a fresh copy of the files of shape 2
    folder, original = directory / "002_sphere", tiny / "002_sphere"
    inputs, outputs, params = (np.load(original / f"{kind}.npy") for kind in ("inputs", "outputs", "params"))

    def refused(match, **files):
        for kind in ("inputs", "outputs", "params"):
            shutil.copy(original / f"{kind}.npy", folder / f"{kind}.npy")
        for kind, array in files.items():
            np.save(folder / f"{kind}.npy", array)
        with pytest.raises(ValueError, match=match):
            ShapeStore(directory, PRE, shapes=[2])

    late = inputs.copy()
    late[1] *= 1.5                                                      # only the second packet
    refused("t = 0", inputs=late)
    refused("float32", inputs=inputs.astype(np.float64))
    refused("wrong shape", outputs=outputs[:, :, :-1])
    refused("wrong length", params=params[:1])
    not_a_number, infinite_start, infinite = outputs.copy(), inputs.copy(), outputs.copy()
    not_a_number[1, 17, 5, 0] = np.nan                                  # after t = 0
    refused("not finite", outputs=not_a_number)
    infinite_start[0, 3, 0] = infinite[0, 3, 0, 0] = np.inf             # at t = 0, in both files
    refused("not finite", inputs=infinite_start, outputs=infinite)

    # a mesh of another size (mass and times checked before the basis is built)
    directory = copy("mesh")
    path = directory / "000_sphere" / "mesh.npz"
    mesh = _mesh(directory, 0)
    np.savez(path, **(mesh | {"mass": 4 * mesh["mass"]}))
    with pytest.raises(ValueError, match="lumped mass"):
        ShapeStore(directory, PRE, shapes=[0])
    np.savez(path, **(mesh | {"t_out": mesh["t_out"] + 1e-6}))
    with pytest.raises(ValueError, match="saved times"):
        ShapeStore(directory, PRE, shapes=[0])
    np.savez(path, **(mesh | {"evecs": mesh["evecs"][:, :99]}))
    with pytest.raises(ValueError, match="eigenpairs"):
        ShapeStore(directory, PRE, shapes=[0])
    # the same surface 1.3 times larger, every file consistent with it: only the area tells
    np.savez(path, **(mesh | {"verts": 1.3 * mesh["verts"], "mass": 1.3 ** 2 * mesh["mass"],
                              "evecs": mesh["evecs"] / np.float32(1.3), "evals": mesh["evals"] / 1.3 ** 2}))
    with pytest.raises(ValueError, match="area of the mesh"):
        ShapeStore(directory, PRE, shapes=[0])

    # the stored pairs, used as they are by the lumped basis, get the checks the Ritz basis makes
    lumped = dataclasses.replace(PRE, basis="lumped")
    for change, match in (({"evecs": 2 * mesh["evecs"]}, "not orthonormal"),
                          ({"evecs": mesh["evecs"][::-1].copy()}, "not orthonormal"),
                          ({"evals": mesh["evals"][::-1].copy()}, "not finite and ascending"),
                          ({"evals": np.where(np.arange(100) == 40, np.nan, mesh["evals"])}, "not finite and ascending")):
        np.savez(path, **(mesh | change))
        with pytest.raises(ValueError, match=match):
            ShapeStore(directory, lumped, shapes=[0])
    np.savez(path, **mesh)
    assert len(ShapeStore(directory, lumped, shapes=[0])) == 2

    # eigenvalues far from K, above or below: not in the units of a surface of area 4π
    for factor in (3, 1 / 3):
        with monkeypatch.context() as patched:
            _altered_ritz(patched, lambda ritz, v, f, e: (lambda b: b._replace(mu=factor * b.mu))(ritz(v, f, e)))
            with pytest.raises(ValueError, match="Weyl"):
                ShapeStore(tiny, PRE, shapes=[0])


def test_shape_arrays_are_unpadded(tiny):
    arrays = shape_arrays(tiny / "005_torus", TINY, PRE)
    assert arrays["evecs"].shape == (1015, 100) and arrays["evecs"].dtype == np.float32
    assert arrays["evals"].dtype == arrays["mass"].dtype == arrays["geo_feat"].dtype == np.float32
    assert arrays["clusters"].dtype == np.int64 and arrays["probes"].shape == (20,)
    assert 0 <= arrays["evals"][0] < 1e-6 and 0.5 < arrays["evals"][-1] < 2
    gram = arrays["evecs"].astype(np.float64).T @ (arrays["mass"].astype(np.float64)[:, None] * arrays["evecs"])
    np.testing.assert_allclose(gram, np.eye(100), atol=1e-6)


def test_training_geo_scale(tiny, tmp_path):
    largest = 0.0
    for index in (0, 1):                                                # the training shapes
        mesh = _mesh(tiny, index)
        largest = max(largest, np.linalg.norm(mesh["verts"] - centroid(mesh["verts"], mesh["mass"]), axis=1).max())
    assert training_geo_scale(tiny) == largest
    assert 1.0 < largest < 2.0                                          # surfaces of area 4π
    (tmp_path / "manifest.json").write_text(json.dumps({"shapes": [{"id": "000_sphere", "split": "val"}]}))
    with pytest.raises(ValueError, match="no training shape"):
        training_geo_scale(tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_store_on_the_gpu_holds_the_same(store, tiny):
    on_gpu = ShapeStore(tiny, PRE, device="cuda")
    assert on_gpu.nbytes == store.nbytes
    for rows in (store.rows(family="sphere"), store.rows(family="torus")):
        ours, theirs, moved = on_gpu.batch(rows), store.batch(rows), store.batch(rows, device="cuda")
        for key in TENSORS:
            assert ours[key].is_cuda and moved[key].is_cuda and not theirs[key].is_cuda
            assert torch.equal(ours[key].cpu(), theirs[key]) and torch.equal(moved[key].cpu(), theirs[key])
        for key in ("clusters", "n_verts", "rows"):
            assert not ours[key].is_cuda and not moved[key].is_cuda and torch.equal(ours[key], theirs[key])
    assert not on_gpu.batch(store.rows(family="torus"), device="cpu")["evecs"].is_cuda


@pytest.mark.slow
@pytest.mark.skipif(not (SMALL_DATA / "manifest.json").exists(), reason="no stage-3 dataset in data/shapes_small")
def test_small_dataset_through_the_store():
    """Two real shapes (700 modes, 11 times) with the adopted preprocessing."""
    store = ShapeStore(SMALL_DATA, shapes=[3, 9])
    assert store.n_max == {"sphere": 10242, "torus": 15677} and store.c_out == 20
    assert store.steps.tolist() == list(range(50, 501, 50))
    assert {name: len(store.rows(sets=name)) for name in SETS} == {"train": 5, "val": 0, "test": 5, "ood": 5}
    for index, n in ((3, 15662), (9, 15677)):
        rows = store.rows(shapes=index)
        batch = store.batch(rows)
        assert batch["n_verts"].tolist() == [n] * len(rows) and not batch["evecs"][:, n:].any()
        assert 0 <= batch["evals"].min() and 0.95 < batch["evals"].max() < 1.02
        assert 0.5 < batch["geo_feat"][:, :n].max() <= 1.0
        assert 0.9 < batch["input"][:, :n].square().mean().sqrt() < 1.1     # the amplitude brings |ψ| near 1
        assert store.oracle_error(batch).max() < 2.5e-3
        outputs = np.concatenate([np.load(SMALL_DATA / f"{index:03d}_torus" / f"{prefix}outputs.npy")
                                  for prefix in (("", "ood_") if index == 9 else ("",))])
        labels = batch["labels"][:, :n].double().numpy().reshape(len(rows), n, 10, 2) / store.preprocess.amplitude
        np.testing.assert_allclose(labels, outputs[:, :, 1:], rtol=5e-7, atol=1e-30)


@pytest.mark.slow
@pytest.mark.skipif(not (FULL_DATA / "manifest.json").exists(), reason="no stage-3 dataset in data/shapes")
def test_full_dataset_layout():
    """The numbers of Plan_GLNO_etapa3.md, step 5, from the manifest and four shapes."""
    assert training_geo_scale(FULL_DATA) == pytest.approx(ShapesPreprocess().geo_scale, abs=5e-5)
    store = ShapeStore(FULL_DATA, shapes=[0, 161, 185, 198])
    assert store.n_max == {"sphere": 10242, "torus": 15679}
    assert [s["split"] for s in store.shapes] == ["train", "val", "test", "test"]
    assert set(store.table["shape"][store.rows(sets="ood")]) == {185, 198}
    assert set(store.table["shape"][store.rows(is_round=True)]) == {161, 185}
    # bytes of the store of everything, padded
    manifest = json.loads((FULL_DATA / "manifest.json").read_text())
    total = 0
    for family, n in store.n_max.items():
        shapes = [entry for entry in manifest["shapes"] if entry["family"] == family]
        packets = 5 * len(shapes) + 5 * sum(entry["split"] == "test" for entry in shapes)
        total += 4 * (len(shapes) * (n * 702 + 700) + packets * n * 22) + 8 * len(shapes) * 700
    assert total / 2**30 == pytest.approx(7.95, abs=0.005)
    per_shape = sum(4 * (store.n_max[s["family"]] * 702 + 700) + 8 * 700 for s in store.shapes)
    per_packet = sum(4 * 22 * store.n_max[shapes_data.FAMILIES[code]] for code in store.table["family"])
    assert store.nbytes == per_shape + per_packet
