import numpy as np
import pytest

torch = pytest.importorskip("torch")

from schrodinger_glno.data_utils import DatasetConfig
from schrodinger_glno.config import SphereConfig
from schrodinger_glno.dataset import generate_sphere_dataset
from schrodinger_glno.glno.data import SphereSplit, glno_config, mass_weighted_relative_l2
from schrodinger_glno.glno.layers import GLNONet

TINY = SphereConfig(subdivisions=2, k0_range=(0.5, 1.0), sigma_range=(0.6, 0.8), ood_k0_range=(1.0, 1.2),
                    final_time=0.1, dt=0.01, output_steps=3, k_eig=20, min_points_per_wavelength=4)


@pytest.fixture(scope="module")
def tiny_split(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sphere")
    generate_sphere_dataset(TINY, DatasetConfig(3, 1, 1, seed=0), directory)
    return SphereSplit(directory, "train")


def test_batch_has_the_shapes_glnonet_expects(tiny_split):
    batch = tiny_split.collate([tiny_split[0], tiny_split[1]])
    n, k = len(tiny_split.mesh["mass"]), TINY.k_eig
    expected = {"input": (2, n, 2), "labels": (2, n, 2 * TINY.output_steps), "vertices": (2, n, 3),
                "mass": (2, n), "evals": (2, k), "evecs": (2, n, k), "geo_feat": (2, n)}
    assert {key: tuple(value.shape) for key, value in batch.items()} == expected
    labels = batch["labels"].reshape(2, n, TINY.output_steps, 2)
    torch.testing.assert_close(labels[:, :, 0], batch["input"])


def test_glnonet_forward_backward_runs_with_finite_gradients(tiny_split):
    torch.manual_seed(0)
    config = glno_config(C_out=tiny_split.n_out, k_eig=TINY.k_eig, blocks=2, C_width=16,
                         encoder_hidden_layers=[16], decoder_hidden_layers=[16], mlp_hidden_layers=[16])
    model = GLNONet(config)
    batch = tiny_split.collate([tiny_split[i] for i in range(3)])
    pred = model(batch)
    assert pred.shape == batch["labels"].shape
    loss = mass_weighted_relative_l2(pred, batch["labels"], batch["mass"])
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert model.block_0.LT.system_poles.grad.abs().sum() > 0


def test_mass_weighted_relative_l2():
    mass = torch.tensor([[1.0, 3.0]])
    target = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    assert mass_weighted_relative_l2(target, target, mass).item() == 0
    assert mass_weighted_relative_l2(torch.zeros_like(target), target, mass).item() == pytest.approx(1)
    # A vertex with three times the area counts three times as much
    error_light = mass_weighted_relative_l2(target + torch.tensor([[[1.0, 0], [0, 0]]]), target, mass)
    error_heavy = mass_weighted_relative_l2(target + torch.tensor([[[0, 0], [1.0, 0]]]), target, mass)
    assert error_heavy.item() == pytest.approx(np.sqrt(3) * error_light.item())


SMALL = {"blocks": 2, "C_width": 16, "encoder_hidden_layers": [16], "decoder_hidden_layers": [16],
         "mlp_hidden_layers": [16]}


def test_training_resumes_and_evaluation_reports_per_time(tiny_split, tmp_path):
    from schrodinger_glno.glno.train import TrainSettings, evaluate_glno, train_glno
    data_dir = tiny_split.inputs.filename and __import__("pathlib").Path(tiny_split.inputs.filename).parents[1]
    first = train_glno(data_dir, tmp_path, TrainSettings(epochs=1, batch_size=2, num_workers=0), SMALL, "cpu",
                       log=lambda m: None)
    resumed = train_glno(data_dir, tmp_path, TrainSettings(epochs=2, batch_size=2, num_workers=0), SMALL, "cpu",
                         log=lambda m: None)
    assert [r["epoch"] for r in first] == [0] and [r["epoch"] for r in resumed] == [0, 1]
    assert resumed[0] == first[0]
    with pytest.raises(ValueError):   # another batch size is another run
        train_glno(data_dir, tmp_path, TrainSettings(epochs=3, batch_size=1, num_workers=0), SMALL, "cpu",
                   log=lambda m: None)
    report = evaluate_glno(data_dir, tmp_path, "test", device="cpu", batch_size=1)
    assert report["samples"] == 1 and len(report["relative_l2_per_time"]) == TINY.output_steps
    assert len(report["reference_norm_per_time"]) == TINY.output_steps
    assert np.isfinite(report["relative_l2_mean"])


def test_glno_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        glno_config(C_out=4, k_eig=10, poles=3)
