"""Stage-3 trainer: guards of the configuration, dropped and clipped steps, checkpoints, resuming, evaluation."""
import dataclasses
import json
import math
import statistics
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from schrodinger_glno.glno import shapes_train
from schrodinger_glno.glno.augment import augment_batch, gauge_generator
from schrodinger_glno.glno.data import mass_weighted_relative_l2
from schrodinger_glno.glno.layers import GLNONet
from schrodinger_glno.glno.shapes_data import ShapesPreprocess, ShapeStore, family_batches
from schrodinger_glno.glno.shapes_train import (VARIANTS, ShapesTrainSettings, calibrate, check_config,
                                                checkpoint_preprocess, clip_threshold, evaluate_rows,
                                                load_checkpoint, packet_errors, predict_shape, run_meta,
                                                shapes_glno_config, train_shapes, train_step, validation_summary)
from schrodinger_glno.shapes_dataset import ShapesConfig, generate_shapes_dataset

# The tiny dataset of test_glno_shapes_data.py: 0 sphere and 1 round torus (train), 2 sphere (val), 3 torus,
# 4 sphere and 5 round torus (test, with out-of-distribution packets); two packets per set, 100 modes, 10 labels.
TINY = ShapesConfig(train_shapes=2, val_shapes=1, test_shapes=3, packets_per_shape=2, ood_packets_per_shape=2,
                    round_torus_every=2, sphere_subdivisions=4, torus_target_edge=0.12, min_wavenumber=5.0,
                    k_eig=100, final_time=0.05, dt=0.005, min_points_per_wavelength=5.0, k0_range=(0.5, 2.0),
                    sigma_range=(0.15, 0.2), ood_k0_range=(2.0, 3.0), ood_sigma_range=(0.18, 0.2),
                    min_tube_sigmas=2.1, max_leak=2e-3, min_spectral_fraction=0.995, min_ood_spectral_fraction=0.99)
PRE = ShapesPreprocess(eval_scale=100.0)
NET = {"blocks": 2, "C_width": 8, "encoder_hidden_layers": [8], "decoder_hidden_layers": [8],
       "mlp_hidden_layers": [8]}
FAST = ShapesTrainSettings(epochs=2, batch_size=2, snapshots=(1,))      # 2 steps per epoch: a sphere and a torus batch


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shapes_train") / "data"
    generate_shapes_dataset(TINY, directory, workers=2)
    return directory


@pytest.fixture(scope="module")
def store(tiny):
    return ShapeStore(tiny, PRE)


def _config(store, variant="full", **overrides):
    return shapes_glno_config(store, variant, **(NET | overrides))


def _weights(model_or_state):
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return {key: value.clone() for key, value in state.items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _quiet(_):
    pass


def test_settings_are_the_papers_and_checked():
    settings = ShapesTrainSettings()
    assert (settings.epochs, settings.batch_size, settings.lr, settings.weight_decay) == (300, 4, 1e-3, 1e-4)
    assert (settings.step_size, settings.gamma, settings.augment, settings.grad_clip) == (100, 0.5, True, 10.0)
    assert settings.max_skipped == 0.02 and settings.snapshots == (100, 200)
    for bad in ({"epochs": -1}, {"batch_size": 0}, {"lr": 0.0}, {"gamma": 0.0}, {"grad_clip": 0.0},
                {"grad_clip": float("inf")}, {"max_skipped": 1.5}, {"snapshots": (0,)}, {"step_size": 0}, {"weight_decay": -1e-4}, {"gamma": 1.5},
                {"max_skipped": -0.1}):
        with pytest.raises(ValueError):
            ShapesTrainSettings(**bad)


def test_config_of_the_two_variants(store):
    full, periodic = shapes_glno_config(store, "full"), shapes_glno_config(store, "periodic")
    assert (full["C_in"], full["C_out"], full["k_eig"], full["k_max"]) == (2, 20, 100, 100)
    assert (full["blocks"], full["C_width"], full["glno_poles"]) == (4, 64, 1)             # the paper's network
    assert full["aperiodic"] is True and full["gaussian_factor"] == 50.0 and full["gaussian_stable"] is True
    assert periodic["aperiodic"] is False and periodic["gaussian_stable"] is True
    assert full["device"] == "cpu" and tuple(VARIANTS) == ("periodic", "full")
    assert _config(store, "periodic", gaussian_stable=False, aperiodic=True)["aperiodic"] is True
    with pytest.raises(ValueError, match="unknown variant"):
        shapes_glno_config(store, "aperiodic")
    with pytest.raises(KeyError):
        shapes_glno_config(store, "full", blocs=2)
    # the periodic network does not see the signs of the basis; the full one does
    batch = store.batch(store.rows(family="torus")[:2])
    lone = batch | {"clusters": torch.arange(100).expand(2, 100)}             # groups of one mode: signs only
    flipped = augment_batch(lone, gauge_generator(0, 0, 0))
    assert not torch.equal(flipped["evecs"], batch["evecs"])
    with torch.no_grad():
        for variant, changes in (("periodic", False), ("full", True)):
            torch.manual_seed(0)
            net = GLNONet(_config(store, variant)).eval()
            difference = (net(flipped) - net(batch)).abs().max() / net(batch).abs().max()
            assert (difference > 1e-3) == changes, (variant, float(difference))


def test_config_refuses_what_breaks_on_padded_batches(store, tmp_path):
    good = _config(store)
    check_config(good, store)
    refused = [{"mlp_norm": "batch"}, {"encoder_norm": "instance"}, {"decoder_norm": "group"},
               {"glno_period_norm": "batch"}, {"glno_aperiod_norm": "batch"}, {"glno_norm": "group"},
               {"norm_feature": "batch"}, {"connect_method": "edge_conv"}, {"rotate": True},
               {"normalize_pole": "max"}, {"geo_preprocess": "max"}, {"outputs_at": "faces"},
               {"k_min": 1}, {"k_max": 99}, {"k_eig": 99}, {"C_in": 5}, {"C_out": 22}, {"encoder": "gnn"}, {"decoder": None}]
    for change in refused:
        with pytest.raises(ValueError, match=next(iter(change))):
            check_config(good | change, store)
        if next(iter(change)) not in ("C_out", "k_eig"):
            with pytest.raises(ValueError, match=next(iter(change))):
                _config(store, **change)
    for change in ({"normalize_basis": True}, {"high_fre_norm": "batch"}, {"geo_feat_norm": "instance"}):
        with pytest.raises(ValueError, match=next(iter(change))):       # keys glno_config does not list
            check_config(good | change, store)
    for allowed in ({"mlp_norm": "layer"}, {"norm_feature": None}, {"mlp_dropout": 0.1}, {"glno_sigma": 3},
                    {"normalize_basis": False}, {"rotate": False}):
        check_config(good | allowed, store)
    with pytest.raises(ValueError, match="mlp_norm"):
        train_shapes(store, tmp_path / "unused", good | {"mlp_norm": "batch"}, FAST, _quiet)


def test_meta_leaves_out_what_may_change_between_sessions(store):
    config = _config(store)
    meta = run_meta(store, config, FAST)
    assert set(meta) == {"trainer_version", "dataset", "preprocess", "network", "settings", "shapes"}
    assert meta["shapes"] == {"train": [0, 1], "val": [2]}
    assert meta["dataset"] == store.identity and meta["preprocess"] == dataclasses.asdict(PRE)
    assert "device" not in meta["network"] and meta["network"]["aperiodic"] is True
    assert set(meta["settings"]) == {"batch_size", "lr", "weight_decay", "step_size", "gamma", "seed", "augment"}
    later = dataclasses.replace(FAST, epochs=300, grad_clip=None, max_skipped=1.0, snapshots=())
    assert run_meta(store, config | {"device": "cuda"}, later) == meta
    assert json.loads(json.dumps(meta)) == meta                                  # plain values only


def test_packet_errors_per_time_and_overall():
    generator = torch.Generator().manual_seed(0)
    target, mass = torch.randn(3, 7, 8, generator=generator), torch.rand(3, 7, generator=generator)
    mass[:, 5:] = 0                                                              # padded vertices
    pred = target + 0.1 * torch.randn(3, 7, 8, generator=generator)
    pred[:, 5:] = 1e6                                                            # meaningless there
    each, whole = packet_errors(pred, target, mass)
    assert each.shape == (3, 4) and whole.shape == (3,)
    for b in range(3):
        weights = mass[b, :5, None]
        for t in range(4):
            columns = slice(2 * t, 2 * t + 2)
            expected = (torch.sum(weights * (pred[b, :5, columns] - target[b, :5, columns]) ** 2)
                        / torch.sum(weights * target[b, :5, columns] ** 2)).sqrt()
            assert each[b, t] == pytest.approx(float(expected), rel=1e-5)
    assert whole.mean() == pytest.approx(float(mass_weighted_relative_l2(pred, target, mass)), rel=1e-6)
    zero = packet_errors(torch.zeros_like(target), target, mass)
    assert torch.allclose(zero[0], torch.ones(3, 4)) and torch.allclose(zero[1], torch.ones(3))


def test_step_trains_clips_and_drops(store, monkeypatch):
    torch.manual_seed(0)
    model = GLNONet(_config(store)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = store.batch(store.rows(sets="train", family="torus"))
    before = _weights(model)
    loss, norm, clipped = train_step(model, optimizer, batch)
    assert 0 < loss < 10 and norm > 0 and clipped is False and not _same(before, _weights(model))
    with torch.no_grad():                       # the loss returned is the one before the step
        expected = float(mass_weighted_relative_l2(_loaded(store, before)(batch), batch["labels"], batch["mass"]))
    assert loss == pytest.approx(expected, rel=1e-6)

    # a clipped step moves the weights as a gradient of that norm would: with SGD, exactly clip * lr
    for clip, expect_clipped in ((norm * 10, False), (norm / 100, True)):
        plain = _loaded(store, before).train()
        sgd = torch.optim.SGD(plain.parameters(), lr=1.0)
        _, measured, was_clipped = train_step(plain, sgd, batch, clip)
        moved = math.sqrt(sum(float((plain.state_dict()[key] - before[key]).abs().square().sum()) for key in before))
        assert was_clipped is expect_clipped and measured == pytest.approx(norm, rel=1e-5)
        assert moved == pytest.approx(clip if expect_clipped else norm, rel=1e-4)

    # a loss that is not a number, and a finite loss with an infinite gradient: nothing moves
    original = shapes_train.mass_weighted_relative_l2
    state, moments = _weights(model), _weights(optimizer.state_dict()["state"][0])
    for broken in (lambda p, t, m: original(p, t, m) * float("nan"),
                   lambda p, t, m: original(p, t, m) + (p.flatten()[0] - p.flatten()[0].detach()).sqrt()):
        monkeypatch.setattr(shapes_train, "mass_weighted_relative_l2", broken)
        value = broken(model(batch), batch["labels"], batch["mass"])
        assert train_step(model, optimizer, batch) == (None, None, False)
        assert _same(state, _weights(model)) and _same(moments, _weights(optimizer.state_dict()["state"][0]))
        assert all(parameter.grad is None for parameter in model.parameters())
    assert math.isfinite(float(value.detach()))                     # the second one: the loss itself is finite


def _loaded(store, weights):
    model = GLNONet(_config(store))
    model.load_state_dict(weights)
    return model


def test_clip_threshold_is_a_multiple_of_the_recent_median():
    assert clip_threshold([1.0] * 50, None) is None
    assert clip_threshold([1.0] * (shapes_train.GRAD_WARMUP - 1), 10.0) is None
    assert clip_threshold([1.0] * shapes_train.GRAD_WARMUP, 10.0) == 10.0
    norms = [1.0] * 10 + [3.0] * 11 + [500.0]                                   # a spike does not move the median
    assert clip_threshold(norms, 10.0) == 10.0 * statistics.median(norms) == 30.0


def _train(store, run_dir, settings=FAST, variant="full", **overrides):
    lines = []
    history = train_shapes(store, run_dir, _config(store, variant, **overrides), settings, lines.append)
    return history, lines


def _comparable(history):
    return [{key: value for key, value in record.items() if key not in ("seconds", "ms_per_sample")}
            for record in history]


def test_two_epochs_are_one_epoch_and_a_resume(store, tmp_path, monkeypatch):
    monkeypatch.setattr(shapes_train, "GRAD_WARMUP", 2)             # so the clip acts within these 6 steps
    settings = dataclasses.replace(FAST, epochs=3, grad_clip=0.9)
    straight, _ = _train(store, tmp_path / "straight", settings)
    assert [record["epoch"] for record in straight] == [0, 1, 2] and sum(r["clipped"] for r in straight) >= 2
    _train(store, tmp_path / "resumed", dataclasses.replace(settings, epochs=1))
    _train(store, tmp_path / "resumed", dataclasses.replace(settings, epochs=2))
    resumed, lines = _train(store, tmp_path / "resumed", settings)
    assert any("Resuming at epoch 2" in line for line in lines)
    assert _comparable(resumed) == _comparable(straight)
    first = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)
    second = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    assert _same(first["model"], second["model"]) and first["grad_norms"] == second["grad_norms"]
    assert first["best_val"] == second["best_val"] and first["epoch"] == 2
    assert _same(torch.load(tmp_path / "straight" / "best.pt", weights_only=False)["model"],
                 torch.load(tmp_path / "resumed" / "best.pt", weights_only=False)["model"])
    assert json.loads((tmp_path / "resumed" / "history.json").read_text()) == resumed
    # asking again for what is done trains nothing
    again, lines = _train(store, tmp_path / "resumed", settings)
    assert again == resumed and not any(line.startswith("epoch ") for line in lines)
    # the gauge and the order of the batches matter: another seed, or no augmentation, is another run
    other, _ = _train(store, tmp_path / "plain", dataclasses.replace(settings, augment=False))
    assert _comparable(other) != _comparable(straight)


def test_history_and_checkpoints(store, tmp_path):
    history, lines = _train(store, tmp_path)
    assert len(history) == 2 and sum(line.startswith("epoch ") for line in lines) == 2
    record = history[-1]
    assert record["steps"] == 2 and record["skipped"] == 0 and record["clipped"] == 0 and record["lr"] == 1e-3
    assert 0 < record["train_loss"] < 10 and 0 < record["val_select"] == record["val_sphere"]
    assert record["val_round"] is None and record["val_torus"] is None      # validation is one sphere
    assert len(record["val_per_time"]["select"]) == 10 and record["val_per_time"]["round"] is None
    assert record["grad_norm_max"] >= record["grad_norm_median"] > 0 and record["ms_per_sample"] > 0
    assert sorted(path.name for path in tmp_path.iterdir()) == ["best.pt", "epoch_1.pt", "history.json", "last.pt",
                                                                "last_prev.pt"]
    config, fingerprint = _config(store), store.fingerprint()
    best_epoch = min(history, key=lambda r: r["val_select"])["epoch"]
    for name, epoch in (("best.pt", best_epoch), ("epoch_1.pt", 0), ("last.pt", 1), ("last_prev.pt", 0)):
        state = torch.load(tmp_path / name, weights_only=False)
        assert state["epoch"] == epoch and state["record"] == history[epoch], name
        assert state["meta"] == run_meta(store, config, FAST) and state["fingerprint"] == fingerprint
        assert "fingerprint" not in state["meta"] and state["config"] == config
    # a checkpoint gives back the network and the validation error it was saved with
    model, state = load_checkpoint(tmp_path / "best.pt", "cpu")
    assert not model.training and "model" not in state and checkpoint_preprocess(state) == PRE
    rows = store.rows(sets="val")
    summary = validation_summary(store, rows, evaluate_rows(model, store, rows, 2))
    assert summary["total"]["select"] == pytest.approx(state["record"]["val_select"], rel=1e-6)
    np.testing.assert_allclose(summary["per_time"]["select"], state["record"]["val_per_time"]["select"], rtol=1e-6)


def test_resume_asks_for_the_same_run(store, tiny, tmp_path):
    _train(store, tmp_path, dataclasses.replace(FAST, epochs=1))
    for change in ({"batch_size": 1}, {"lr": 2e-3}, {"seed": 1}, {"augment": False}, {"step_size": 50}):
        with pytest.raises(ValueError, match="another settings"):
            _train(store, tmp_path, dataclasses.replace(FAST, **change))
    with pytest.raises(ValueError, match="another network"):
        _train(store, tmp_path, variant="periodic")                                     # the branch
    with pytest.raises(ValueError, match="another network"):
        _train(store, tmp_path, gaussian_factor=700.0)
    for change in ({"eval_scale": 700.0}, {"cluster_gap": 3e-4}):                       # the scale and τ
        with pytest.raises(ValueError, match="another preprocess"):
            _train(ShapeStore(tiny, dataclasses.replace(PRE, **change)), tmp_path)
    elsewhere = ShapeStore(tiny, PRE)
    elsewhere.identity = "0" * 64                                                        # another dataset
    with pytest.raises(ValueError, match="another dataset"):
        _train(elsewhere, tmp_path)
    assert torch.load(tmp_path / "last.pt", weights_only=False)["epoch"] == 0           # nothing was overwritten
    # more epochs, another clip, other snapshots: the same run goes on
    history, _ = _train(store, tmp_path, dataclasses.replace(FAST, epochs=2, grad_clip=None, max_skipped=0.5,
                                                             snapshots=(2,)))
    assert [record["epoch"] for record in history] == [0, 1] and (tmp_path / "epoch_2.pt").exists()


def test_resume_checks_the_basis_with_a_tolerance(store, tmp_path):
    settings = dataclasses.replace(FAST, epochs=1)
    _train(store, tmp_path, settings)
    path = tmp_path / "last.pt"
    state = torch.load(path, weights_only=False)
    name = "000_sphere"

    def with_probe_moved(by):
        probes = list(state["fingerprint"]["shapes"][name])
        probes[0] += by
        shapes = state["fingerprint"]["shapes"] | {name: probes}
        torch.save(state | {"fingerprint": state["fingerprint"] | {"shapes": shapes}}, path)

    with_probe_moved(1e-7)                      # another CPU: the last digits differ, the run goes on
    assert len(_train(store, tmp_path, FAST)[0]) == 2
    with_probe_moved(0.5)                       # an eigenvector with the other sign
    with pytest.raises(ValueError, match="Ritz basis"):
        _train(store, tmp_path, dataclasses.replace(FAST, epochs=3))


def test_a_last_checkpoint_cut_short_falls_back_to_the_previous_one(store, tmp_path):
    straight, _ = _train(store, tmp_path / "straight", dataclasses.replace(FAST, epochs=3))
    run = tmp_path / "run"
    _train(store, run)
    last = run / "last.pt"
    last.write_bytes(last.read_bytes()[:1000])                      # as Drive leaves an interrupted copy
    history, lines = _train(store, run, dataclasses.replace(FAST, epochs=3))
    assert any("last.pt cannot be read" in line for line in lines) and any("last_prev.pt" in line for line in lines)
    assert _comparable(history) == _comparable(straight)            # epoch 1 was repeated, identically
    assert _same(torch.load(last, weights_only=False)["model"],
                 torch.load(tmp_path / "straight" / "last.pt", weights_only=False)["model"])
    # both unreadable: stop instead of starting again over them
    for name in ("last.pt", "last_prev.pt"):
        (run / name).write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="cannot be read"):
        _train(store, run, dataclasses.replace(FAST, epochs=4))
    assert (run / "last.pt").read_bytes() == b"not a checkpoint"
    # a file that loads but is something else
    torch.save({"model": {}}, run / "last.pt")
    (run / "last_prev.pt").unlink()
    with pytest.raises(ValueError, match="cannot be read"):
        _train(store, run)


def test_dropped_steps_are_counted_and_too_many_stop_the_run(store, tmp_path, monkeypatch):
    original = shapes_train.mass_weighted_relative_l2
    calls = {"count": 0}

    def sometimes_broken(pred, target, mass):
        calls["count"] += 1
        loss = original(pred, target, mass)
        return loss * float("nan") if calls["count"] == 2 and pred.requires_grad else loss

    monkeypatch.setattr(shapes_train, "mass_weighted_relative_l2", sometimes_broken)
    with pytest.raises(RuntimeError, match="1 of 2 steps"):         # 1 of 2 is more than 2 %
        _train(store, tmp_path / "strict", dataclasses.replace(FAST, epochs=1))
    assert not (tmp_path / "strict" / "last.pt").exists()
    calls["count"] = 0
    history, _ = _train(store, tmp_path / "lenient", dataclasses.replace(FAST, epochs=1, max_skipped=0.5))
    assert history[0]["skipped"] == 1 and history[0]["steps"] == 2 and math.isfinite(history[0]["train_loss"])
    state = torch.load(tmp_path / "lenient" / "last.pt", weights_only=False)
    assert state["history"][0]["skipped"] == 1 and len(state["grad_norms"]) == 1


def test_weights_that_are_not_finite_are_never_saved(store, tmp_path, monkeypatch):
    _train(store, tmp_path, dataclasses.replace(FAST, epochs=1))
    saved = (tmp_path / "last.pt").read_bytes()
    original = shapes_train.train_step

    def poisoned(model, optimizer, batch, clip=None):
        result = original(model, optimizer, batch, clip)
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = float("inf")
        return result

    monkeypatch.setattr(shapes_train, "train_step", poisoned)
    with pytest.raises(RuntimeError, match="steps had a loss"):
        _train(store, tmp_path)
    assert (tmp_path / "last.pt").read_bytes() == saved and not (tmp_path / "last_prev.pt").exists()


def test_evaluation_files_each_packet_under_its_number(store):
    torch.manual_seed(1)
    model = GLNONet(_config(store)).train()
    rows = store.rows(sets=("test", "ood"))[::-1].copy()            # tori and spheres interleaved, backwards
    errors = evaluate_rows(model, store, rows, 4)
    assert errors["per_time"].shape == (12, 10) and errors["total"].shape == (12,) and model.training
    for position, row in enumerate(rows):
        alone = evaluate_rows(model, store, [row], 1)
        np.testing.assert_allclose(errors["per_time"][position], alone["per_time"][0], rtol=1e-5)
        batch = store.batch([row])
        with torch.no_grad():
            direct = mass_weighted_relative_l2(model.eval()(batch), batch["labels"], batch["mass"])
        model.train()
        assert errors["total"][position] == pytest.approx(float(direct), rel=1e-5)
    summary = validation_summary(store, rows, errors)
    shapes = store.table["shape"][rows]
    for name, members in (("select", [3, 4]), ("sphere", [4]), ("torus", [3]), ("round", [5])):
        assert summary["total"][name] == pytest.approx(errors["total"][np.isin(shapes, members)].mean(), rel=1e-12)
        np.testing.assert_allclose(summary["per_time"][name], errors["per_time"][np.isin(shapes, members)].mean(0))
    assert len({round(value, 6) for value in summary["total"].values()}) == 4      # the groups do differ
    json.dumps(summary)


def test_predict_shape_gives_states_as_the_dataset_stores_them(store, tiny):
    torch.manual_seed(2)
    model = GLNONet(_config(store)).eval()
    rows, states = predict_shape(model, store, 3, batch_size=3)         # torus 3: 1006 vertices, padded to 1015
    assert rows.tolist() == store.rows(shapes=3).tolist() and states.shape == (4, 1006, 11)
    assert states.dtype == np.complex128
    inputs = np.concatenate([np.load(tiny / "003_torus" / f"{prefix}inputs.npy") for prefix in ("", "ood_")])
    np.testing.assert_allclose(states[:, :, 0], inputs[..., 0] + 1j * inputs[..., 1], rtol=1e-6, atol=1e-9)
    batch = store.batch(rows)
    with torch.no_grad():
        out = model(batch)[:, :1006].double().numpy().reshape(4, 1006, 10, 2) / PRE.amplitude
    np.testing.assert_allclose(states[:, :, 1:], out[..., 0] + 1j * out[..., 1], rtol=1e-5, atol=1e-8)
    only, _ = predict_shape(model, store, 3, sets="ood")
    assert only.tolist() == store.rows(sets="ood", shapes=3).tolist()
    with pytest.raises(ValueError, match="no packets"):
        predict_shape(model, store, 0, sets="ood")
    # with t = 0 among the labels nothing is prepended
    kept = ShapeStore(tiny, dataclasses.replace(PRE, drop_initial=False), shapes=[0])
    _, states = predict_shape(GLNONet(_config(kept)).eval(), kept, 0)
    assert states.shape == (2, 2562, 11)


def test_calibration_measures_both_families_and_projects_the_run(store):
    before = torch.get_rng_state()
    measured = calibrate(store, _config(store), FAST, warmup=1, steps=2, epochs=(50, 300))
    assert set(measured["families"]) == {"sphere", "torus"} and measured["batch_size"] == 2
    sphere, torus = measured["families"]["sphere"], measured["families"]["torus"]
    assert sphere["train_ms_per_sample"] > 0 and torus["train_ms_per_sample"] > 0
    assert (sphere["train_packets"], sphere["val_packets"], torus["train_packets"], torus["val_packets"]) == (2, 2, 2, 0)
    assert sphere["eval_ms_per_sample"] > 0 and torus["eval_ms_per_sample"] is None and sphere["reserved_gib"] is None
    expected = (2 * sphere["train_ms_per_sample"] + 2 * torus["train_ms_per_sample"]
                + 2 * sphere["eval_ms_per_sample"]) / 1000
    assert measured["epoch_seconds"] == pytest.approx(expected)
    assert measured["hours"] == {"50": pytest.approx(50 * expected / 3600), "300": pytest.approx(300 * expected / 3600)}
    json.dumps(measured)
    assert not torch.equal(before, torch.get_rng_state())           # it seeds torch: train_shapes seeds itself again


def test_training_needs_packets_to_train_and_to_choose_with(tiny, tmp_path):
    with pytest.raises(ValueError, match="no training packets"):
        only_val = ShapeStore(tiny, PRE, sets=("val", "test"))
        train_shapes(only_val, tmp_path, _config(only_val), FAST, _quiet)
    with pytest.raises(ValueError, match="nothing to choose"):
        only_train = ShapeStore(tiny, PRE, sets=("train",))
        train_shapes(only_train, tmp_path, _config(only_train), FAST, _quiet)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
def test_one_batch_is_overfitted_far_below_predicting_the_input(store):
    """The plan's mandatory check, on the tiny meshes: the network and the step do learn."""
    torch.manual_seed(0)
    model = GLNONet(shapes_glno_config(store, "periodic", blocks=2, C_width=32)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = store.batch(store.rows(sets="train", family="torus"))
    psi0 = float(packet_errors(batch["input"].repeat(1, 1, 10), batch["labels"], batch["mass"])[1].mean())
    losses = [train_step(model, optimizer, batch)[0] for _ in range(300)]
    assert all(loss is not None for loss in losses)
    assert losses[-1] < 0.3 * min(psi0, 1.0) and losses[-1] < 0.5 * losses[0], (losses[0], losses[-1], psi0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_training_on_the_gpu_with_the_store_on_either_device(tiny, tmp_path):
    for name, device in (("on_gpu", "cuda"), ("on_cpu", "cpu")):
        held = ShapeStore(tiny, PRE, device=device)
        config = shapes_glno_config(held, "full", device="cuda", **NET)
        history = train_shapes(held, tmp_path / name, config, FAST, _quiet)
        assert len(history) == 2 and history[-1]["reserved_gib"] > 0 and math.isfinite(history[-1]["val_select"])
        model, _ = load_checkpoint(tmp_path / name / "best.pt", "cuda")
        _, states = predict_shape(model, held, 2)
        assert states.shape == (2, 2562, 11) and np.isfinite(states).all()
    first = torch.load(tmp_path / "on_gpu" / "last.pt", weights_only=False)["history"]
    second = torch.load(tmp_path / "on_cpu" / "last.pt", weights_only=False)["history"]
    assert [r["val_select"] for r in first] == pytest.approx([r["val_select"] for r in second], rel=1e-3)




# ---------------------------------------------------------------------------------------------------------
# Added by the mutation review of step 6. They need two more imports at the top of the file:
#     import types
#     from schrodinger_glno.glno.shapes_data import SETS
import types
from schrodinger_glno.glno.shapes_data import SETS


@pytest.fixture(scope="module")
def mixed(tiny):
    """The tiny dataset with its test packets counted as validation: spheres 2 and 4, torus 3 and round torus 5,
    so the four reporting groups hold different packets (in ``store`` validation is one sphere)."""
    held = ShapeStore(tiny, PRE)
    held.table["set"][held.table["set"] == SETS.index("test")] = SETS.index("val")
    return held


def test_best_keeps_the_lowest_val_select_through_resumes(store, tmp_path, monkeypatch):
    """val_select falls at every epoch of the other runs of this file, where ``best`` and ``last`` never differ."""
    script = [3.0, 1.0, 2.0, 0.5]
    values, original = iter(script), shapes_train.validation_summary

    def scripted(*arguments):
        summary = original(*arguments)
        summary["total"]["select"] = next(values)
        return summary

    monkeypatch.setattr(shapes_train, "validation_summary", scripted)
    weights = {}
    for epochs, best_epoch, best_val in ((1, 0, 3.0), (2, 1, 1.0), (3, 1, 1.0), (4, 3, 0.5)):     # one new epoch each
        history, _ = _train(store, tmp_path, dataclasses.replace(FAST, epochs=epochs))
        best = torch.load(tmp_path / "best.pt", weights_only=False)
        last = torch.load(tmp_path / "last.pt", weights_only=False)
        assert [record["val_select"] for record in history] == script[:epochs]
        assert (best["epoch"], best["record"]["val_select"], last["best_val"]) == (best_epoch, best_val, best_val)
        weights[epochs] = last["model"]
        assert _same(best["model"], weights[best_epoch + 1])
    assert not _same(weights[2], weights[3])


def test_the_learning_rate_follows_the_schedule_through_resumes(store, tmp_path):
    settings = dataclasses.replace(FAST, epochs=5, lr=2e-3, weight_decay=3e-4, step_size=2, gamma=0.5, snapshots=())
    straight, _ = _train(store, tmp_path / "straight", settings)
    # "lr" is the rate each epoch trained with: halved after every 2 full epochs
    assert [record["lr"] for record in straight] == pytest.approx([2e-3, 2e-3, 1e-3, 1e-3, 5e-4], rel=1e-12)
    for epochs in range(1, 6):                                      # a session per epoch
        resumed, _ = _train(store, tmp_path / "resumed", dataclasses.replace(settings, epochs=epochs))
    assert _comparable(resumed) == _comparable(straight)
    state = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    group, scheduler = state["optimizer"]["param_groups"][0], state["scheduler"]
    assert (group["weight_decay"], group["initial_lr"], group["betas"]) == (3e-4, 2e-3, (0.9, 0.999))
    assert (scheduler["step_size"], scheduler["gamma"], scheduler["last_epoch"]) == (2, 0.5, 5)
    # and the defaults reach the optimizer
    _train(store, tmp_path / "default", dataclasses.replace(FAST, epochs=1))
    state = torch.load(tmp_path / "default" / "last.pt", weights_only=False)
    group, scheduler = state["optimizer"]["param_groups"][0], state["scheduler"]
    assert (group["lr"], group["weight_decay"], scheduler["step_size"], scheduler["gamma"]) == (1e-3, 1e-4, 100, 0.5)


def test_every_epoch_has_its_own_batches_and_every_step_its_own_gauge(store, tmp_path, monkeypatch):
    """A run that repeated the same batches or gauges would also be 'the same after a resume'."""
    seen, original = [], shapes_train.train_step

    def spy(model, optimizer, batch, clip=None):
        seen.append((batch["rows"].tolist(), batch["evecs"].clone()))
        return original(model, optimizer, batch, clip)

    monkeypatch.setattr(shapes_train, "train_step", spy)
    settings = dataclasses.replace(FAST, epochs=3, batch_size=1, seed=5)            # 4 steps per epoch
    _train(store, tmp_path / "augmented", settings)
    rows = store.rows(sets="train")
    expected = [(epoch, step, members.tolist()) for epoch in range(3)
                for step, members in enumerate(family_batches(store, rows, 1, seed=5, epoch=epoch))]
    assert [members for members, _ in seen] == [members for _, _, members in expected]
    orders = {str([members for members, _ in seen[start:start + 4]]) for start in (0, 4, 8)}
    assert len(orders) == 3 and str([[row] for row in rows.tolist()]) not in orders     # shuffled, anew each epoch
    for (epoch, step, members), (_, evecs) in zip(expected, seen):
        batch = store.batch(members)
        gauged = augment_batch(batch, gauge_generator(5, epoch, step))["evecs"]
        assert torch.allclose(evecs, gauged, atol=1e-6) and not torch.allclose(evecs, batch["evecs"], atol=1e-3)
    seen.clear()                                                                    # augment=False: the canonical basis
    _train(store, tmp_path / "plain", dataclasses.replace(settings, epochs=1, augment=False))
    assert len(seen) == 4 and all(torch.equal(evecs, store.batch(members)["evecs"]) for members, evecs in seen)


def test_history_reports_every_validation_group_under_its_name(mixed, tmp_path):
    history = train_shapes(mixed, tmp_path, _config(mixed), dataclasses.replace(FAST, epochs=1), _quiet)
    model, state = load_checkpoint(tmp_path / "best.pt", "cpu")
    rows = mixed.rows(sets="val")
    errors = evaluate_rows(model, mixed, rows, 2)
    shapes, record = mixed.table["shape"][rows], history[0]
    assert sorted(set(shapes.tolist())) == [2, 3, 4, 5] and state["record"] == record
    for name, members in (("select", [2, 3, 4]), ("sphere", [2, 4]), ("torus", [3]), ("round", [5])):
        chosen = np.isin(shapes, members)
        assert record[f"val_{name}"] == pytest.approx(errors["total"][chosen].mean(), rel=1e-6), name
        np.testing.assert_allclose(record["val_per_time"][name], errors["per_time"][chosen].mean(0), rtol=1e-6)
    assert len({round(record[f"val_{name}"], 6) for name in ("select", "sphere", "torus", "round")}) == 4


def test_the_skip_limit_is_a_fraction_of_the_steps_of_the_epoch(store, tmp_path, monkeypatch):
    original, calls, broken = shapes_train.mass_weighted_relative_l2, {"count": 0}, set()

    def loss(pred, target, mass):                                   # only train_step calls it
        calls["count"] += 1
        value = original(pred, target, mass)
        return value * float("nan") if calls["count"] in broken else value

    monkeypatch.setattr(shapes_train, "mass_weighted_relative_l2", loss)
    lenient = dataclasses.replace(FAST, epochs=1, max_skipped=0.5)
    broken.add(1)                                                   # the first step: 1 of 2 is not more than 50 %
    history, _ = _train(store, tmp_path / "first", lenient)
    assert history[0]["skipped"] == 1 and history[0]["train_loss"] is not None
    calls["count"] = 0
    broken.add(2)                                                   # both: 2 of 2 is
    with pytest.raises(RuntimeError, match="2 of 2 steps"):
        _train(store, tmp_path / "both", lenient)
    calls["count"] = 0                                              # all of them allowed: an epoch without a step
    history, lines = _train(store, tmp_path / "all", dataclasses.replace(lenient, max_skipped=1.0))
    record = history[0]
    assert (record["skipped"], record["train_loss"], record["grad_norm_median"], record["grad_norm_max"]) == (2, None, None, None)
    assert any("train none" in line for line in lines) and json.loads((tmp_path / "all" / "history.json").read_text()) == history


def test_a_gradient_that_is_not_finite_is_dropped_with_a_clip_too(store, monkeypatch):
    """After the warm-up every step of a run has a clip: the drop must not depend on ``clip is None``."""
    torch.manual_seed(0)
    model = GLNONet(_config(store)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = store.batch(store.rows(sets="train", family="torus"))
    _, norm, _ = train_step(model, optimizer, batch)
    original, state = shapes_train.mass_weighted_relative_l2, _weights(model)
    monkeypatch.setattr(shapes_train, "mass_weighted_relative_l2",
                        lambda p, t, m: original(p, t, m) + (p.flatten()[0] - p.flatten()[0].detach()).sqrt())
    for clip in (norm / 100, norm * 100):
        assert train_step(model, optimizer, batch, clip) == (None, None, False)
        assert _same(state, _weights(model)) and all(bool(torch.isfinite(value).all()) for value in state.values())


def test_the_gradient_window_keeps_the_last_norms_and_can_reach_the_warm_up(store, tmp_path, monkeypatch):
    assert (shapes_train.GRAD_WINDOW, shapes_train.GRAD_WARMUP) == (200, 20)        # Plan, step 6
    monkeypatch.setattr(shapes_train, "GRAD_WINDOW", 3)
    seen, original = [], shapes_train.train_step

    def spy(model, optimizer, batch, clip=None):
        result = original(model, optimizer, batch, clip)
        seen.append(result[1])
        return result

    monkeypatch.setattr(shapes_train, "train_step", spy)
    _train(store, tmp_path, dataclasses.replace(FAST, epochs=1))
    _train(store, tmp_path, FAST)                                                   # 2 steps, a resume, 2 more
    state = torch.load(tmp_path / "last.pt", weights_only=False)
    assert len(seen) == 4 and state["grad_norms"] == seen[-3:]


@pytest.mark.parametrize("cut_at", ["best", "last"])
def test_a_save_cut_short_leaves_the_checkpoints_whole(store, tmp_path, monkeypatch, cut_at):
    _train(store, tmp_path, dataclasses.replace(FAST, epochs=1))
    kept = {name: (tmp_path / name).read_bytes() for name in ("last.pt", "best.pt")}
    save = torch.save

    def cut(state, path, *arguments, **options):
        if not path.name.startswith(cut_at):
            return save(state, path, *arguments, **options)
        path.write_bytes(b"half a file")
        raise OSError("the session ended")

    with monkeypatch.context() as patch:
        patch.setattr(shapes_train.torch, "save", cut)
        with pytest.raises(OSError, match="session ended"):
            _train(store, tmp_path)                                 # epoch 1 improves: best.pt, then last.pt
    assert (tmp_path / "last.pt").read_bytes() == kept["last.pt"]
    if cut_at == "best":
        assert (tmp_path / "best.pt").read_bytes() == kept["best.pt"]
    history, lines = _train(store, tmp_path)                        # the run goes on from the end of epoch 0
    assert len(history) == 2 and any("Resuming at epoch 1" in line for line in lines)
    assert sorted(path.name for path in tmp_path.iterdir() if path.suffix != ".tmp") == [
        "best.pt", "epoch_1.pt", "history.json", "last.pt", "last_prev.pt"]


def test_a_checkpoint_is_loaded_on_the_device_asked_for_and_modes_are_given_back(store, tmp_path):
    _train(store, tmp_path, dataclasses.replace(FAST, epochs=1))
    state = torch.load(tmp_path / "best.pt", weights_only=False)
    torch.save(state | {"config": state["config"] | {"device": "cuda:1"}}, tmp_path / "from_gpu.pt")    # as Colab writes it
    model, loaded = load_checkpoint(tmp_path / "from_gpu.pt", "cpu")
    rows = store.rows(sets="val")
    errors = evaluate_rows(model, store, rows, 2)
    assert validation_summary(store, rows, errors)["total"]["select"] == pytest.approx(loaded["record"]["val_select"], rel=1e-6)
    assert not model.training                                       # evaluate_rows and predict_shape give the mode back
    predict_shape(model, store, 2)
    assert not model.training
    predict_shape(model.train(), store, 2)
    assert model.training


def test_dropout_is_off_in_evaluation_and_repeats_after_a_resume(store, tmp_path):
    torch.manual_seed(3)
    model = GLNONet(_config(store, mlp_dropout=0.5)).train()
    rows = store.rows(sets="val")
    assert np.array_equal(evaluate_rows(model, store, rows, 2)["total"], evaluate_rows(model, store, rows, 2)["total"])
    assert np.array_equal(predict_shape(model, store, 2)[1], predict_shape(model, store, 2)[1])
    with torch.no_grad():
        batch = store.batch(rows)
        assert not torch.equal(model(batch), model(batch))          # the dropout is there
    straight, _ = _train(store, tmp_path / "straight", FAST, mlp_dropout=0.5)
    _train(store, tmp_path / "resumed", dataclasses.replace(FAST, epochs=1), mlp_dropout=0.5)
    resumed, _ = _train(store, tmp_path / "resumed", FAST, mlp_dropout=0.5)
    assert _comparable(resumed) == _comparable(straight)


def test_calibration_counts_packets_and_times_whole_training_steps(mixed, monkeypatch):
    """With a clock that advances 1 s per reading, every number of ``calibrate`` is known."""
    steps, original = [], shapes_train.train_step

    def spy(model, optimizer, batch, clip=None):
        steps.append((len(batch["rows"]), bool(torch.equal(batch["evecs"], mixed.batch(batch["rows"].numpy())["evecs"]))))
        return original(model, optimizer, batch, clip)

    monkeypatch.setattr(shapes_train, "train_step", spy)

    def run(batch_size, count):
        readings = iter(range(10**6))
        monkeypatch.setattr(shapes_train, "time", types.SimpleNamespace(perf_counter=lambda: float(next(readings))))
        steps.clear()
        return calibrate(mixed, _config(mixed), dataclasses.replace(FAST, batch_size=batch_size), warmup=1, steps=count)

    measured = run(2, 2)            # per family: 2 timed steps of 2 packets in 2 s; 2 of the 4 val packets in 1 s
    assert steps == [(2, False)] * 6                                                # 2 families x (1 + 2), augmented
    for family in ("sphere", "torus"):
        values = measured["families"][family]
        assert (values["train_ms_per_sample"], values["eval_ms_per_sample"]) == (500.0, 500.0)
        assert (values["train_packets"], values["val_packets"]) == (2, 4)
    assert measured["epoch_seconds"] == pytest.approx(2 * (0.5 * 2 + 0.5 * 4))
    assert measured["hours"]["300"] == pytest.approx(300 * 6.0 / 3600)
    measured = run(1, 4)            # 4 timed steps of 1 packet in 4 s; 2 of the 4 val packets in 1 s
    assert steps == [(1, False)] * 10
    assert measured["families"]["sphere"]["train_ms_per_sample"] == 1000.0
    assert measured["families"]["sphere"]["eval_ms_per_sample"] == 500.0
    assert measured["epoch_seconds"] == pytest.approx(2 * (1.0 * 2 + 0.5 * 4))


def test_tf32_matmuls_are_turned_off():
    matmul = torch.backends.cuda.matmul
    new_api = hasattr(matmul, "fp32_precision")
    before = matmul.fp32_precision if new_api else matmul.allow_tf32
    try:
        if new_api:
            matmul.fp32_precision = "tf32"
        else:
            matmul.allow_tf32 = True
        shapes_train.ieee_matmuls()
        assert (matmul.fp32_precision == "ieee") if new_api else (matmul.allow_tf32 is False)
    finally:
        if new_api:
            matmul.fp32_precision = before
        else:
            matmul.allow_tf32 = before


def test_a_refused_run_leaves_no_directory_behind(store, tmp_path):
    with pytest.raises(ValueError, match="mlp_norm"):
        train_shapes(store, tmp_path / "refused", _config(store) | {"mlp_norm": "batch"}, FAST, _quiet)
    assert not (tmp_path / "refused").exists()


def test_weights_that_turn_not_finite_in_the_last_step_are_not_saved(store, tmp_path, monkeypatch):
    """In test_weights_that_are_not_finite_are_never_saved the run stops one step later, on a loss that is not
    finite (the max_skipped guard, whose message also says "not finite"), and never reaches the check of the
    weights. Here they break in the last step of the epoch, where only that check can stop the save."""
    _train(store, tmp_path, dataclasses.replace(FAST, epochs=1))
    kept = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    original, calls = shapes_train.train_step, {"count": 0}

    def poisoned(model, optimizer, batch, clip=None):
        result = original(model, optimizer, batch, clip)
        calls["count"] += 1
        if calls["count"] == 2:                                     # the last of the epoch's 2 steps
            with torch.no_grad():
                next(model.parameters()).view(-1)[0] = float("inf")
        return result

    monkeypatch.setattr(shapes_train, "train_step", poisoned)
    with pytest.raises(RuntimeError, match="the weights are not finite"):
        _train(store, tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == kept


def test_history_times_an_epoch_per_training_packet(store, tmp_path, monkeypatch):
    readings = iter(range(10**6))                                   # a clock that advances 1 s per reading
    monkeypatch.setattr(shapes_train, "time", types.SimpleNamespace(perf_counter=lambda: float(next(readings))))
    history, _ = _train(store, tmp_path, dataclasses.replace(FAST, epochs=1))
    # three readings: the start, the end of the 2 steps (1 s for 4 packets) and the end of validation
    assert (history[0]["ms_per_sample"], history[0]["seconds"]) == (250.0, 2.0)


def test_after_a_fallback_the_unreadable_file_does_not_replace_the_good_one(store, tmp_path):
    _train(store, tmp_path)                                         # last.pt: epoch 1, last_prev.pt: epoch 0
    last = tmp_path / "last.pt"
    last.write_bytes(last.read_bytes()[:1000])
    _train(store, tmp_path)                                         # falls back and repeats epoch 1, nothing more
    assert [torch.load(tmp_path / name, weights_only=False)["epoch"] for name in ("last.pt", "last_prev.pt")] == [1, 0]
    last.write_bytes(last.read_bytes()[:1000])                      # cut again before another epoch ends
    history, _ = _train(store, tmp_path, dataclasses.replace(FAST, epochs=3))
    assert [record["epoch"] for record in history] == [0, 1, 2]


def test_resume_asks_for_the_same_training_and_validation_shapes(store, tiny, tmp_path):
    _train(store, tmp_path, dataclasses.replace(FAST, epochs=1))
    for shapes in ([0, 2], [1, 2]):                                 # fewer or other training shapes
        with pytest.raises(ValueError, match="another shapes.train"):
            _train(ShapeStore(tiny, PRE, shapes=shapes), tmp_path)
    for same_shapes in ({"sets": ("train", "val")}, {"shapes": [0, 1, 2, 3]}):
        _train(ShapeStore(tiny, PRE, **same_shapes), tmp_path, dataclasses.replace(FAST, epochs=1))
    with pytest.raises(ValueError, match=r"network\.gaussian_factor \(50\.0 in the checkpoint, 700\.0 now\)"):
        _train(store, tmp_path, gaussian_factor=700.0)


def test_after_a_fallback_the_previous_checkpoint_stays_readable(store, tmp_path):
    _train(store, tmp_path)                                         # 2 epochs
    last = tmp_path / "last.pt"
    last.write_bytes(last.read_bytes()[:1000])
    _train(store, tmp_path)                                         # falls back and repeats epoch 1: one save
    assert torch.load(tmp_path / "last_prev.pt", weights_only=False)["epoch"] == 0
    assert (tmp_path / "last_unreadable.pt").stat().st_size == 1000
    last.write_bytes(last.read_bytes()[:1000])                      # cut again before the second save
    history, _ = _train(store, tmp_path, dataclasses.replace(FAST, epochs=3))
    assert [record["epoch"] for record in history] == [0, 1, 2]


def test_resume_checks_best_and_history_and_refuses_a_run_without_its_state(store, tmp_path):
    history, _ = _train(store, tmp_path)
    best = tmp_path / "best.pt"
    best.write_bytes(best.read_bytes()[:1000])
    (tmp_path / "history.json").write_text("[]")                    # a cut between last.pt and history.json
    again, lines = _train(store, tmp_path)                          # nothing left to train
    assert any("best.pt cannot be read" in line for line in lines)
    assert json.loads((tmp_path / "history.json").read_text()) == again == history
    history, _ = _train(store, tmp_path, dataclasses.replace(FAST, epochs=3))
    state = torch.load(best, weights_only=False)                    # the next epoch wrote a best.pt that agrees
    assert state["epoch"] == 2 and state["record"] == history[2]
    assert torch.load(tmp_path / "last.pt", weights_only=False)["best_val"] == history[2]["val_select"]
    for name in ("last.pt", "last_prev.pt"):
        (tmp_path / name).unlink()
    with pytest.raises(ValueError, match="neither last.pt nor last_prev.pt"):
        _train(store, tmp_path)
    assert torch.load(best, weights_only=False)["epoch"] == 2       # the old run's best.pt was not trained over
