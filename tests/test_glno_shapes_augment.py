"""Gauge of the eigenbasis: the aperiodic switch in layers.py and the random signs/rotations of augment.py."""
import types
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from schrodinger_glno.glno.augment import augment_batch, block_haar, gauge_generator, group_starts
from schrodinger_glno.glno.data import glno_config
from schrodinger_glno.glno.layers import GLNONet, Laplace_Transform_Layer
from schrodinger_glno.glno.spectral import eigen_clusters

SMALL = {"blocks": 2, "C_width": 8, "encoder_hidden_layers": [8], "decoder_hidden_layers": [8],
         "mlp_hidden_layers": [8]}
GROUPS = [1, 2, 3, 1, 4, 1, 2, 1, 1, 3, 2, 1]                      # K = 22


def _evals(groups=GROUPS, spread=0.0):
    """Eigenvalues equal (spread 0) or nearly equal inside each group, in [0, ~1] like μ/700."""
    values = []
    for g, size in enumerate(groups):
        base = 0.0 if g == 0 else 0.04 * g
        values += [base * (1 + spread * i) for i in range(size)]
    return np.array(values)


def _batch(batch_size=2, n=150, evals=None, seed=0, vertices=False):
    """Random meshes: positive masses, mass-orthonormal bases, the given eigenvalues."""
    evals = _evals() if evals is None else evals
    k = len(evals)
    rng = np.random.default_rng(seed)
    mass = rng.uniform(0.5, 1.5, (batch_size, n)) * 4 * np.pi / n
    evecs = np.empty((batch_size, n, k))
    for b in range(batch_size):
        q, _ = np.linalg.qr(np.sqrt(mass[b])[:, None] * rng.standard_normal((n, k)))
        evecs[b] = q / np.sqrt(mass[b])[:, None]
    batch = {"input": rng.standard_normal((batch_size, n, 2)), "mass": mass, "evecs": evecs,
             "evals": np.tile(evals, (batch_size, 1)), "geo_feat": rng.uniform(0, 1.2, (batch_size, n))}
    if vertices:
        batch["vertices"] = rng.standard_normal((batch_size, n, 3))
    batch = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in batch.items()}
    batch["clusters"] = torch.as_tensor(np.tile(eigen_clusters(evals, 1e-4), (batch_size, 1)))
    return batch


def _net(aperiodic: bool | None, c_in=2, k=len(_evals()), seed=0):
    config = glno_config(C_out=4, k_eig=k, C_in=c_in, **SMALL)
    if aperiodic is not None:
        config["aperiodic"] = aperiodic
    torch.manual_seed(seed)
    return GLNONet(config).eval()


def _change(a, b, mass):
    w = mass.unsqueeze(-1)
    return (torch.sum(w * (a - b) ** 2) / torch.sum(w * b ** 2)).sqrt().item()


@pytest.fixture
def one_thread():
    """Bit-identity checks: with several threads, reductions may be summed in a different order."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_default_layer_is_the_repository_layer(one_thread):
    """Without the key the network is the stage-1/2 one: same weights and bit-identical outputs."""
    batch = _batch(vertices=True)
    default, explicit = _net(None, c_in=5), _net(True, c_in=5)
    assert all(torch.equal(a, b) for a, b in zip(default.state_dict().values(), explicit.state_dict().values()))
    assert default.state_dict().keys() == _net(False, c_in=5).state_dict().keys()   # no new parameters
    with torch.no_grad():
        assert torch.equal(default(batch), explicit(batch))


def test_periodic_branch_ignores_signs_and_rotations_in_degenerate_groups():
    batch = _batch()
    flipped = {**batch, "evecs": batch["evecs"] * torch.where(torch.rand(2, 1, 22) < 0.5, -1.0, 1.0)}
    rotated = augment_batch(batch, gauge_generator(0, 0, 0))
    periodic, full = _net(False), _net(True)
    with torch.no_grad():
        reference = periodic(batch)
        assert reference.shape == (2, 150, 4)
        assert _change(periodic(flipped), reference, batch["mass"]) < 1e-5
        assert _change(periodic(rotated), reference, batch["mass"]) < 1e-5
        # the aperiodic branch depends on the gauge: this is why training draws random gauges
        assert _change(full(flipped), full(batch), batch["mass"]) > 1e-4
        assert _change(full(rotated), full(batch), batch["mass"]) > 1e-4


def test_block_haar_structure():
    labels = eigen_clusters(_evals(), 1e-4)
    clusters = torch.as_tensor(np.stack([labels, np.arange(22)]))      # sample 1: every mode alone
    Q = block_haar(clusters, gauge_generator(1, 2, 3), dtype=torch.float64)
    starts = group_starts(labels)
    mask = np.zeros((22, 22), bool)
    for s, n in zip(starts, np.diff(np.append(starts, 22))):
        mask[s:s + n, s:s + n] = True
    assert torch.all(Q[0][~torch.as_tensor(mask)] == 0)
    assert torch.allclose(Q[0].T @ Q[0], torch.eye(22, dtype=torch.float64), atol=1e-12)
    assert torch.all(Q[1] == torch.diag(torch.diagonal(Q[1])))                # signs only
    assert set(torch.diagonal(Q[1]).tolist()) <= {-1.0, 1.0}
    singletons = [s for s, n in zip(starts, np.diff(np.append(starts, 22))) if n == 1]
    assert set(Q[0, singletons, singletons].tolist()) <= {-1.0, 1.0}


def test_block_haar_is_reproducible_and_unbiased():
    clusters = torch.as_tensor(eigen_clusters(_evals(), 1e-4))[None]
    a = block_haar(clusters, gauge_generator(0, 4, 7))
    assert torch.equal(a, block_haar(clusters, gauge_generator(0, 4, 7)))
    assert not torch.equal(a, block_haar(clusters, gauge_generator(0, 4, 8)))
    assert not torch.equal(a, block_haar(clusters, gauge_generator(0, 5, 7)))
    draws = 2000
    Qs = torch.cat([block_haar(clusters, gauge_generator(9, 0, i), dtype=torch.float64) for i in range(draws)])
    # E[Q] = 0 for Haar blocks, and E‖mean of n draws‖²_F = K/n (each entry of an s-block has variance 1/s)
    assert torch.linalg.norm(Qs.mean(0)) < 3 * np.sqrt(22 / draws)
    # Haar on O(s), not SO(s): half of the 4×4 blocks are reflections
    reflections = (torch.linalg.det(Qs[:, 6:10, 6:10]) < 0).double().mean().item()
    assert 0.45 < reflections < 0.55


def test_augment_batch_keeps_orthonormality_padding_and_the_input():
    batch = _batch(n=140)
    for key in ("evecs", "mass", "input", "geo_feat"):    # 10 rows of zero padding, as in a family batch
        pad = torch.zeros(batch[key].shape[0], 10, *batch[key].shape[2:])
        batch[key] = torch.cat([batch[key], pad], dim=1)
    before = {key: value.clone() for key, value in batch.items()}
    out = augment_batch(batch, gauge_generator(0, 0, 1))
    assert all(torch.equal(batch[key], before[key]) for key in batch)            # nothing modified in place
    assert torch.all(out["evecs"][:, -10:] == 0)
    for key in ("input", "mass", "evals", "geo_feat", "clusters"):
        assert out[key] is batch[key]
    m = batch["mass"].unsqueeze(-1)
    gram = out["evecs"].transpose(1, 2) @ (m * out["evecs"])
    eye = torch.eye(22).expand_as(gram)
    assert torch.allclose(batch["evecs"].transpose(1, 2) @ (m * batch["evecs"]), eye, atol=1e-5)
    assert torch.allclose(gram, eye, atol=1e-5)


def test_group_labels_are_checked():
    with pytest.raises(ValueError):
        group_starts(np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        group_starts(np.array([1, 1, 2]))
    with pytest.raises(ValueError):
        group_starts(np.array([0, 2, 2]))
    batch = _batch()
    with pytest.raises(ValueError):
        augment_batch({**batch, "clusters": batch["clusters"][:, :-1]}, gauge_generator(0, 0, 0))


UPSTREAM = Path(__file__).resolve().parents[1] / "GLNO-repo" / "model" / "GLNO" / "layers.py"


def _upstream_layers():
    """The repository's layers.py, with only its unusable imports removed."""
    source = UPSTREAM.read_text()
    for line in ("from Benchmark.CortexDiffusion.geometry import norm\n",
                 "from model.GLNO.pointnet import FPSPointNetModule\n",
                 "from torch_geometric.nn.conv.edge_conv import EdgeConv\n"):
        source = source.replace(line, "")
    source = source.replace("from ..geometry import", "from schrodinger_glno.glno.geometry import")
    module = types.ModuleType("upstream_glno_layers")
    exec(compile(source, str(UPSTREAM), "exec"), module.__dict__)
    return module


def _layer_inputs(seed=0, b=2, n=150, k=22, width=8):
    rng = np.random.default_rng(seed)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    return (t(rng.standard_normal((b, n, width))), t(rng.uniform(0.5, 1.5, (b, n)) * 4 * np.pi / n),
            t(np.tile(np.linspace(0, 1, k), (b, 1))), t(rng.standard_normal((b, n, k)) / 3),
            t(rng.uniform(0, 1.2, (b, n))))


def _layer_config(**extra):
    return {**glno_config(C_out=4, k_eig=22, C_in=2, **SMALL), **extra}


@pytest.mark.skipif(not UPSTREAM.exists(), reason="no GLNO-repo checkout")
def test_default_laplace_layer_is_bit_identical_to_the_repository(one_thread):
    upstream = _upstream_layers()
    args = _layer_inputs()
    for norms in ({}, {"glno_period_norm": "layer", "glno_aperiod_norm": "layer", "glno_norm": "layer"}):
        torch.manual_seed(0)
        ours = Laplace_Transform_Layer(_layer_config(**norms))
        torch.manual_seed(0)
        theirs = upstream.Laplace_Transform_Layer(_layer_config(**norms))
        with torch.no_grad():
            assert torch.equal(ours(*args), theirs(*args))


def test_switched_off_layer_is_the_periodic_part_of_the_full_layer(monkeypatch, one_thread):
    args = _layer_inputs()
    torch.manual_seed(0)
    off = Laplace_Transform_Layer(_layer_config(aperiodic=False))
    torch.manual_seed(0)
    full = Laplace_Transform_Layer(_layer_config())
    monkeypatch.setattr(full, "pole_to_operator", lambda *a, **kw: torch.zeros(2, 8, 150))
    with torch.no_grad():
        result = off(*args)
        assert torch.equal(result, full(*args)) and result.abs().max() > 0


def _reference_block_haar(labels, generator, dtype=torch.float64):
    """One block at a time from the same draw stream (sample, then group, row by row)."""
    batch, k = labels.shape
    blocks = []
    for b in range(batch):
        starts = group_starts(labels[b])
        blocks += [(b, s, n) for s, n in zip(starts, np.diff(np.append(starts, k)))]
    draws = torch.randn(sum(n * n for _, _, n in blocks), generator=generator, dtype=torch.float64)
    Q, offset = torch.zeros(batch, k, k, dtype=dtype), 0
    for b, s, n in blocks:
        q, r = torch.linalg.qr(draws[offset:offset + n * n].reshape(n, n))
        Q[b, s:s + n, s:s + n] = (q * torch.sign(torch.diagonal(r))).to(dtype)
        offset += n * n
    return Q


def test_block_haar_matches_a_block_by_block_reference():
    rng = np.random.default_rng(3)
    partitions = []
    for _ in range(5):                                # a different group structure per sample
        sizes = []
        while sum(sizes) < 40:
            sizes.append(int(rng.integers(1, 6)))
        sizes[-1] -= sum(sizes) - 40
        sizes = [s for s in sizes if s > 0]
        partitions.append(np.repeat(np.cumsum([0] + sizes[:-1]), sizes))
    cases = [np.stack(partitions), np.zeros((1, 1), int), np.zeros((3, 1), int),
             np.zeros((2, 7), int), np.tile(np.arange(9), (2, 1))]
    for labels in cases:
        for dtype in (torch.float64, torch.float32):
            fast = block_haar(torch.as_tensor(labels), gauge_generator(2, 1, len(labels)), dtype=dtype)
            reference = _reference_block_haar(labels, gauge_generator(2, 1, len(labels)), dtype)
            assert torch.allclose(fast, reference, atol=1e-12 if dtype == torch.float64 else 1e-6)


def test_group_labels_must_point_at_group_starts():
    for labels in ([0, 1, 1, 2], [-1, 0, 1], [-1, -1, -1], [0, 0.0, 1]):
        with pytest.raises(ValueError):
            group_starts(np.array(labels))


def test_samples_of_a_batch_get_independent_gauges_and_steps_do_not_alias():
    labels = eigen_clusters(np.array([0, 1, 1, 1, 2, 2.0]), 1e-4)
    Q = block_haar(torch.as_tensor(np.stack([labels, labels])), gauge_generator(0, 0, 0), dtype=torch.float64)
    assert not torch.equal(Q[0], Q[1])
    clusters = torch.as_tensor(labels)[None]
    assert not torch.equal(block_haar(clusters, gauge_generator(0, 4, 8)), block_haar(clusters, gauge_generator(0, 5, 7)))


def test_augment_keeps_float64():
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((40, 6)))
    labels = eigen_clusters(np.array([0, 1, 1, 1, 2, 2.0]), 1e-4)
    out = augment_batch({"evecs": torch.as_tensor(q[None]), "clusters": torch.as_tensor(labels[None])},
                        gauge_generator(0, 0, 0))["evecs"]
    assert out.dtype == torch.float64
    assert torch.allclose(out[0].T @ out[0], torch.eye(6, dtype=torch.float64), atol=1e-12)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


@cuda
def test_gauge_on_the_gpu_is_the_cpu_gauge():
    clusters = torch.as_tensor(eigen_clusters(np.array([0, 1, 1, 1, 2, 2, 3.0]), 1e-4))[None]
    cpu = block_haar(clusters, gauge_generator(0, 1, 2), dtype=torch.float64)
    gpu = block_haar(clusters.cuda(), gauge_generator(0, 1, 2), dtype=torch.float64)
    assert gpu.device.type == "cuda" and torch.equal(gpu.cpu(), cpu)


@cuda
def test_augment_refuses_tf32(monkeypatch):
    labels = torch.as_tensor(eigen_clusters(np.array([0, 1, 1, 2.0]), 1e-4))[None]
    batch = {"evecs": torch.zeros(1, 10, 4, device="cuda"), "clusters": labels}
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32")
    else:
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(RuntimeError, match="TF32"):
        augment_batch(batch, gauge_generator(0, 0, 0))


def _filter_layer(stable: bool, factor: float):
    torch.manual_seed(0)
    return Laplace_Transform_Layer(_layer_config(gaussian_factor=factor, gaussian_stable=stable))


def _poles_above_the_spectrum(layer, omega: float):
    with torch.no_grad():
        layer.system_poles.copy_(torch.complex(layer.system_poles.real, torch.full_like(layer.system_poles.real, omega)))


def test_stable_gaussian_filter_matches_the_default_inside_the_spectrum(one_thread):
    """Poles start with ω in [0, 1), inside μ/700 ∈ [0, 1]: softmax gives the same weights."""
    args = _layer_inputs()
    for factor in (1.0, 50.0):
        default, stable = _filter_layer(False, factor), _filter_layer(True, factor)
        with torch.no_grad():
            a, b = default(*args), stable(*args)
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6) and a.abs().max() > 0


def test_stable_gaussian_filter_keeps_gradients_finite_above_the_spectrum(one_thread):
    """ω² = 1.21 is 0.21 above the top eigenvalue: with factor 50 every exp underflows (0/0)."""
    args = _layer_inputs()
    grads = {}
    for stable in (False, True):
        layer = _filter_layer(stable, 50.0)
        _poles_above_the_spectrum(layer, 1.1)
        layer(*args).sum().backward()
        grads[stable] = layer.system_poles.grad
    assert not torch.isfinite(grads[False]).all()           # the repository's layer
    assert torch.isfinite(grads[True]).all()


def test_stable_gaussian_weights_go_to_the_nearest_eigenvalue():
    evals = torch.as_tensor(np.tile(np.linspace(0, 1, 22), (2, 1)), dtype=torch.float32)
    residue = torch.randn(2, 8, 8, 1, dtype=torch.cfloat)
    out = _filter_layer(True, 700.0).gaussian_spectral_filter(torch.full((8, 8, 1), 1.02), evals, residue)
    assert torch.equal(out[..., -1], residue.real) and torch.all(out[..., :-1] == 0)


def test_stable_gaussian_filter_has_the_default_gradients_inside_the_spectrum():
    """Same weights and the same gradients: a detached or rescaled softmax would fail here."""
    args = _layer_inputs()
    for factor in (1.0, 50.0):
        grads = []
        for stable in (False, True):
            layer = _filter_layer(stable, factor)
            (layer(*args) ** 2).sum().backward()
            grads.append({name: p.grad.clone() for name, p in layer.named_parameters()})
        for name, g in grads[0].items():
            assert torch.allclose(grads[1][name], g, rtol=1e-4, atol=1e-7), name
        assert grads[1]["system_poles"].imag.abs().max() > 0
