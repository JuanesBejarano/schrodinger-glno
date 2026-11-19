"""Dataset configuration, complex channels and manifest integrity for GLNO."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class DatasetConfig:
    train: int = 800
    val: int = 100
    test: int = 100
    seed: int = 0
    ood: int = 0

    def __post_init__(self) -> None:
        if min(self.train, self.val, self.test) < 1 or self.ood < 0:
            raise ValueError("train, val and test must be positive; ood >= 0")

def psi_to_channels(psi: NDArray, V: NDArray | None = None) -> NDArray:
    if V is not None and V.max() > 0:
        return np.stack([psi.real, psi.imag, V / V.max()], axis=-1)
    return np.stack([psi.real, psi.imag], axis=-1)

def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _manifest_identity(manifest: dict) -> str:
    payload = {key: manifest[key] for key in ("format_version", "preset", "physics", "counts", "splits", "sha256")}
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def dataset_identity(directory: Path, *, verify: bool = True) -> str:
    """Return the manifest identity and optionally verify every data file."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["identity"] != _manifest_identity(manifest):
        raise ValueError("Dataset manifest identity is invalid")
    if verify:
        for relative, expected in manifest["sha256"].items():
            if _digest(directory / relative) != expected:
                raise ValueError(f"Dataset file differs from manifest: {relative}")
    return manifest["identity"]
