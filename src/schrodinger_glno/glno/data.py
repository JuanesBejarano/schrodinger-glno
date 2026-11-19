"""Connect the sphere and torus datasets to the paper's ``GLNONet``.

``GLNONet.forward`` takes a dictionary with ``input`` (B, N, C), ``vertices``
(B, N, 3; concatenated to ``input`` inside the network), ``mass`` (B, N),
``evals`` (B, K), ``evecs`` (B, N, K) and ``geo_feat`` (B, N). Here the mesh is
the same for every sample, so those tensors are loaded once and expanded to
the batch without copying.

Shapes for one sample: ``input`` is ψ0 as (N, 2) = (Re, Im); ``labels`` are the
saved states flattened time by time to (N, 2 * n_t) =
(Re t0, Im t0, Re t1, Im t1, ...).
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def glno_config(C_out: int, k_eig: int, device: str = "cpu", **overrides) -> dict:
    """Every key ``GLNONet`` reads (the repository's ``config/`` YAML files are missing).

    blocks, C_width, glno_poles, glno_sigma and the learning rate follow the
    paper's Table 5, Poisson column: one fixed mesh, as in this stage. The rest
    are not in the paper and use the repository's simplest options.
    """
    config = {
        # Table 5 (Poisson)
        "blocks": 4, "C_width": 64, "glno_poles": 1, "glno_sigma": 1,
        # Input and output: ψ0 (2 channels) + vertex coordinates (3) -> 2 channels per saved time
        "C_in": 2 + 3, "vertices_dim": 3, "C_out": C_out, "device": device,
        "k_eig": k_eig, "k_min": 0, "k_max": k_eig, "rotate": False, "geo_preprocess": None,
        "last_activation": None, "outputs_at": "vertices",
        # Encoder and decoder
        "encoder": "mlp", "encoder_hidden_layers": [64], "encoder_dropout": 0.0,
        "encoder_norm": None, "encoder_activation": "gelu",
        "decoder": "mlp", "decoder_hidden_layers": [64], "decoder_dropout": 0.0,
        "decoder_norm": None, "decoder_activation": "gelu",
        # Laplace transform layer (Eq. 17-18: Gaussian filter and Taylor expansion)
        "gaussian": "expx2", "gaussian_factor": 1.0, "exp_x2_appr": True,
        "normalize_pole": None, "glno_scale": None,
        "glno_period_norm": None, "glno_aperiod_norm": None, "glno_norm": None,
        # GLNO block
        "connect_method": "mlp", "mlp_hidden_layers": [64, 64], "mlp_dropout": 0.0,
        "mlp_norm": None, "mlp_activation": "gelu", "norm_feature": "layer", "skip": True,
        "learned_geo_feat": False, "high_fre": False,
    }
    unknown = set(overrides) - set(config)
    if unknown:
        raise KeyError(f"Unknown GLNO config keys: {sorted(unknown)}")
    config.update(overrides)
    return config


class SurfaceSplit(Dataset):
    """One split of a sphere or torus dataset, read lazily from the ``.npy`` memmaps."""

    def __init__(self, data_dir: Path, split: str, dtype: torch.dtype = torch.float32):
        data_dir = Path(data_dir)
        manifest = json.loads((data_dir / "manifest.json").read_text())
        if split not in manifest["splits"]:
            raise ValueError(f"Unknown split {split!r}; available: {manifest['splits']}")
        with np.load(data_dir / "mesh.npz") as mesh:
            self.mesh = {"vertices": torch.as_tensor(mesh["verts"], dtype=dtype),
                         "mass": torch.as_tensor(mesh["mass"], dtype=dtype),
                         "evals": torch.as_tensor(mesh["evals"], dtype=dtype),
                         "evecs": torch.as_tensor(mesh["evecs"], dtype=dtype),
                         "geo_feat": torch.as_tensor(mesh["geo_feat"], dtype=dtype)}
            self.times = mesh["t_out"]
        self.inputs = np.load(data_dir / split / "inputs.npy", mmap_mode="r")
        self.outputs = np.load(data_dir / split / "outputs.npy", mmap_mode="r")
        self.params = np.load(data_dir / split / "params.npy")
        self.dtype = dtype

    @property
    def n_out(self) -> int:
        return 2 * self.outputs.shape[2]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int) -> dict:
        n_vertices = self.outputs.shape[1]
        return {"input": torch.as_tensor(np.array(self.inputs[i]), dtype=self.dtype),
                "labels": torch.as_tensor(np.array(self.outputs[i]).reshape(n_vertices, -1), dtype=self.dtype)}

    def collate(self, samples: list[dict]) -> dict:
        """Stack samples and expand the shared mesh tensors to the batch size."""
        batch = {key: torch.stack([s[key] for s in samples]) for key in ("input", "labels")}
        size = len(samples)
        for key, value in self.mesh.items():
            batch[key] = value.unsqueeze(0).expand(size, *value.shape)
        return batch


SphereSplit = SurfaceSplit    # stage-1 name, kept for existing code


def mass_weighted_relative_l2(pred: torch.Tensor, target: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of ``‖pred - target‖_M / ‖target‖_M``, over all channels.

    ``‖u‖_M² = Σ_i m_i Σ_c u_ic²``: each vertex counts with its area, as in the
    physical norm, instead of the per-vertex Euclidean norm of the repository's ``LpLoss``.
    """
    weights = mass.unsqueeze(-1)
    error = torch.sqrt(torch.sum(weights * (pred - target)**2, dim=(-2, -1)))
    scale = torch.sqrt(torch.sum(weights * target**2, dim=(-2, -1)))
    return torch.mean(error / scale)
