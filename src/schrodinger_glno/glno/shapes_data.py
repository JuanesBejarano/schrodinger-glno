"""Stage-3 data for the GLNO: what the network sees of every shape, built once and kept on one device.

``ShapeStore`` reads a dataset of ``shapes_dataset`` (one directory per shape) and builds, shape by
shape, the tensors of Plan_GLNO_etapa3.md §1:

``evecs``     ``E = Φ U``, the lumped-orthonormal Ritz basis of ``spectral.ritz_polar_basis``
``evals``     ``max(μ, 0) / eval_scale``, the blended-mass Ritz values Crank–Nicolson used
``mass``      the lumped mass, the one ``to_basis`` and the loss use
``geo_feat``  ``‖x - c‖ / geo_scale``, ``c`` the area centroid
``input``     ``amplitude · (Re ψ0, Im ψ0)``
``labels``    ``amplitude · (Re ψ(t_1), Im ψ(t_1), Re ψ(t_2), …)``: the saved states without t = 0
``clusters``  the eigenvalue groups of ``spectral.eigen_clusters``, for ``augment.augment_batch``

There is no ``vertices`` key: the network runs with ``C_in = 2``.

Layout. Shapes of one family share their tensors, padded with zeros to the largest mesh of that
family in the whole dataset (10 242 vertices for the spheres and 15 679 for the tori of
``data/shapes``, at most 1 % of padding), so a store of some shapes pads as the store of all of
them. A padded vertex has mass 0 and a zero row in ``evecs``, so it adds nothing to ``to_basis`` or
to a mass-weighted error; the network's output there is meaningless and is dropped with
``n_verts``. The Ritz basis is computed on the CPU, one shape at a time, and copied into tensors
allocated beforehand on ``device``: the host stays under ~1 GB while ``data/shapes`` fills
7.95 GiB of the device. A batch holds packets of one family (``batch`` refuses a mix); there is no
``DataLoader`` and there are no worker processes.

Packets are numbered in the order of the shapes, each shape's in-distribution packets before its
out-of-distribution ones; ``table`` has one entry per packet and ``rows`` selects from it. The
numbers are positions in one store: a store of other sets or shapes numbers its packets
differently, so what is saved per packet is keyed by the table's ``(shape, set, packet)``.

The Ritz basis is recomputed wherever the store is built, and LAPACK may round differently on
another CPU, so ``fingerprint`` keeps a few numbers per shape that pin down the basis (signs
included) and ``fingerprint_mismatch`` compares them with a tolerance; a hash of the bytes would
differ between two machines that agree to 1e-9.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable
import numpy as np
from numpy.typing import NDArray
import torch
from ..mesh import vertex_areas
from ..shapes_dataset import FAMILIES, MANIFEST_FILE, ShapesConfig, shapes_dataset_identity
from .spectral import (RITZ_VERSION, centroid, eigen_clusters, geo_feature, ritz_polar_basis,
                       spectral_oracle_error)

__all__ = ["SETS", "GROUPS", "ShapesPreprocess", "ShapeStore", "shape_arrays", "family_batches", "group_means",
           "fingerprint_mismatch", "training_geo_scale"]

SETS = ("train", "val", "test", "ood")              # packet sets; "ood" exists on test shapes only
GROUPS = ("select", "sphere", "torus", "round")     # reporting groups, see ShapeStore.group_positions
FINGERPRINT_VERSION = 1
_FAMILY_CODE = {family: code for code, family in enumerate(FAMILIES)}
_PROBE_PRIMES = ((2, 3, 5, 7), (11, 13, 17, 19))


@dataclass(frozen=True)
class ShapesPreprocess:
    """How a stored shape becomes network inputs; part of a run's identity.

    ``basis``: ``"ritz"`` (``E = Φ U`` with ``μ``) or ``"lumped"`` (the stored ``Φ`` with ``λ``, an
    ablation: it misses the data by up to 27 %). ``eval_scale`` is one constant for every shape,
    never a shape's own largest eigenvalue. ``geo_scale`` is the largest ``‖x - c‖`` over the
    training shapes of ``data/shapes`` (``training_geo_scale`` gives 1.403019) and stays fixed for
    any other dataset. ``amplitude`` brings a typical ``|ψ|`` per vertex near 1 on a surface of
    area 4π. ``drop_initial`` leaves t = 0 out of the labels: it is the input.
    """
    basis: str = "ritz"
    eval_scale: float = 700.0
    cluster_gap: float = 1e-4
    geo_scale: float = 1.4030
    amplitude: float = math.sqrt(8 * math.pi)
    drop_initial: bool = True

    def __post_init__(self) -> None:
        if self.basis not in ("ritz", "lumped"):
            raise ValueError("basis must be 'ritz' or 'lumped'")
        for name in ("eval_scale", "geo_scale", "amplitude"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and positive")
        if not (math.isfinite(self.cluster_gap) and self.cluster_gap >= 0):
            raise ValueError("cluster_gap must be finite and >= 0")


def _shape_indices(shapes) -> NDArray:
    """Shape indices as an int64 vector; refuses what is not an integer or a list of integers."""
    if np.iterable(shapes) and not isinstance(shapes, (str, np.ndarray)):
        shapes = list(shapes)                                   # a set, a range, a generator
    indices = np.atleast_1d(np.asarray(shapes))
    if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError("shapes must be an integer or a list of integers")
    return indices.astype(np.int64)


def _probes(U: NDArray | None, values: NDArray) -> NDArray:
    """16 numbers ``aᵀ U b`` that move by O(1) when a column of ``U`` changes sign, and 4 of the eigenvalues.

    ``a`` and ``b`` are fixed cosines, not random draws, so they do not depend on a NumPy version.
    """
    k = len(values)
    modes = np.arange(1, k + 1)[:, None]
    a, b = (np.cos(modes * np.sqrt(primes)) for primes in _PROBE_PRIMES)
    core = a.T @ b if U is None else a.T @ U @ b
    return np.concatenate([core.ravel(), [values[min(1, k - 1)], values[k // 2], values[-1], values.mean()]])


def shape_arrays(folder: Path, config: ShapesConfig, preprocess: ShapesPreprocess) -> dict:
    """The network's view of the mesh in ``folder``, unpadded, as NumPy arrays.

    ``evecs`` (N, K) float32, ``evals`` (K,) float32 already divided by ``eval_scale``, ``mass`` (N,)
    and ``geo_feat`` (N,) float32, ``clusters`` (K,) int64 and ``probes`` (20,) float64 for the
    fingerprint. Refuses a mesh whose files disagree with ``config``, whose area is not the
    config's 4π (``amplitude`` and ``geo_scale`` assume it), or whose largest eigenvalue is not
    near ``K`` (Weyl's law on area 4π: ``λ_K ≈ K``; a wide window, against eigenvalues in other
    units).
    """
    with np.load(Path(folder) / "mesh.npz") as stored:
        verts, faces, mass, evals, evecs, t_out = (stored[key] for key in
                                                   ("verts", "faces", "mass", "evals", "evecs", "t_out"))
    k = config.k_eig
    if evecs.shape != (len(verts), k) or evals.shape != (k,) or mass.shape != (len(verts),):
        raise ValueError(f"{folder}: mesh.npz does not hold {k} eigenpairs of its {len(verts)} vertices")
    if t_out.shape != (config.output_steps,) or not np.allclose(t_out, config.output_times(), rtol=0, atol=1e-12):
        raise ValueError(f"{folder}: the saved times differ from the dataset config")
    if not np.allclose(mass, vertex_areas(verts, faces), rtol=1e-12, atol=0):
        raise ValueError(f"{folder}: the stored mass is not the lumped mass of the mesh")
    if abs(mass.sum() / config.area - 1) > 1e-9:
        raise ValueError(f"{folder}: the area of the mesh is {mass.sum():.9g}, not {config.area:.9g}")
    if preprocess.basis == "ritz":
        ritz = ritz_polar_basis(verts, faces, evecs)
        basis, values, U = ritz.E, ritz.mu, ritz.U
    else:
        phi = evecs.astype(np.float64)
        if not np.allclose(phi.T @ (mass[:, None] * phi), np.eye(k), rtol=0, atol=1e-6):   # as ritz_polar_basis
            raise ValueError(f"{folder}: the stored eigenvectors are not orthonormal in the lumped mass")
        if not np.isfinite(evals).all() or np.any(np.diff(evals) < 0):
            raise ValueError(f"{folder}: the stored eigenvalues are not finite and ascending")
        basis, values, U = evecs.astype(np.float32, copy=False), np.maximum(evals, 0.0), None
    if not 0.5 < values[-1] / k < 2:
        raise ValueError(f"{folder}: largest eigenvalue / K = {values[-1] / k:.3g}, expected about 1 "
                         "(Weyl's law on area 4π)")
    return {"evecs": basis, "evals": (values / preprocess.eval_scale).astype(np.float32),
            "mass": mass.astype(np.float32), "geo_feat": geo_feature(verts, mass, preprocess.geo_scale),
            "clusters": eigen_clusters(values, preprocess.cluster_gap), "probes": _probes(U, values)}


class ShapeStore:
    """Every selected packet of a stage-3 dataset as network inputs on ``device``.

    ``sets`` picks the packet sets to load and ``shapes`` (indices) restricts the shapes; a shape
    without a selected set is skipped. ``verify`` checks the SHA-256 of every file of the dataset
    first. ``progress`` receives one line per shape.

    Attributes: ``identity`` (the dataset's), ``config`` (its ``ShapesConfig``), ``preprocess``,
    ``steps`` and ``times`` of the labels, ``dt``, ``k_eig``, ``c_out`` (label channels),
    ``n_max`` (padded vertices per family), ``shapes`` (one dict per loaded shape), ``table`` (one
    entry per packet: ``shape``, ``set`` as an index into ``SETS``, ``family`` as an index into
    ``shapes_dataset.FAMILIES``, ``round``, ``packet``, ``n_verts``, and the rows ``slot`` and
    ``position`` of its shape and of itself in the family tensors), ``params`` and
    ``diagnostics`` (the dataset's columns, per packet) and ``nbytes``.
    """

    def __init__(self, directory: Path, preprocess: ShapesPreprocess | None = None, *,
                 device: str | torch.device = "cpu", sets: Iterable[str] = SETS,
                 shapes: int | Iterable[int] | None = None, verify: bool = False,
                 progress: Callable[[str], None] | None = None) -> None:
        directory = Path(directory)
        self.preprocess = preprocess = ShapesPreprocess() if preprocess is None else preprocess
        self.device = torch.device(device)
        self.identity = shapes_dataset_identity(directory, verify=verify)
        manifest = json.loads((directory / MANIFEST_FILE).read_text())
        self.config = config = ShapesConfig(**manifest["config"])
        sets = (sets,) if isinstance(sets, str) else tuple(sets)
        if not set(sets) <= set(SETS):
            raise ValueError(f"unknown packet sets {sorted(set(sets) - set(SETS))}; there are {SETS}")
        entries = manifest["shapes"]
        if [entry["index"] for entry in entries] != list(range(config.n_shapes)):
            raise ValueError("the manifest does not list the shapes in order")
        wanted = None if shapes is None else set(_shape_indices(shapes).tolist())
        if wanted is not None and not wanted <= set(range(config.n_shapes)):
            raise ValueError(f"shape indices must be in 0..{config.n_shapes - 1}")
        selected = []
        for entry in entries:
            index = entry["index"]
            if (entry["family"], entry["split"], entry["round"]) != (config.family(index), config.split(index),
                                                                     config.is_round(index)):
                raise ValueError(f"{entry['id']}: the manifest entry disagrees with the dataset config")
            packet_sets = [(name, prefix, count) for name, prefix, count, _ in config.packet_sets(index)
                           if name in sets]
            if packet_sets and (wanted is None or index in wanted):
                selected.append((entry, packet_sets))
        if not selected:
            raise ValueError("no packets selected")

        first = 1 if preprocess.drop_initial else 0
        self.steps, self.times = config.output_indices()[first:], config.output_times()[first:]
        self.dt, self.k_eig, self.c_out = config.dt, config.k_eig, 2 * len(self.steps)
        self.n_max = {family: max(entry["n_verts"] for entry in entries if entry["family"] == family)
                      for family in FAMILIES if any(entry["family"] == family for entry in entries)}
        self._families = {}
        for family in FAMILIES:
            members = [packet_sets for entry, packet_sets in selected if entry["family"] == family]
            if not members:
                continue
            n_shapes, n = len(members), self.n_max[family]
            n_packets = sum(count for packet_sets in members for _, _, count in packet_sets)

            def zeros(*shape: int) -> torch.Tensor:
                return torch.zeros(shape, dtype=torch.float32, device=self.device)

            self._families[family] = {
                "evecs": zeros(n_shapes, n, self.k_eig), "evals": zeros(n_shapes, self.k_eig),
                "mass": zeros(n_shapes, n), "geo_feat": zeros(n_shapes, n),
                "clusters": torch.zeros((n_shapes, self.k_eig), dtype=torch.int64),      # CPU, for augment_batch
                "inputs": zeros(n_packets, n, 2), "labels": zeros(n_packets, n, self.c_out)}

        amplitude = np.float32(preprocess.amplitude)
        slots, positions = dict.fromkeys(FAMILIES, 0), dict.fromkeys(FAMILIES, 0)
        columns = {key: [] for key in ("shape", "set", "family", "round", "packet", "n_verts", "slot", "position")}
        params, diagnostics, self.shapes, self._probes = [], [], [], {}
        for entry, packet_sets in selected:
            start = time.perf_counter()
            family, n, folder = entry["family"], entry["n_verts"], directory / entry["id"]
            data, slot = self._families[family], slots[family]
            arrays = shape_arrays(folder, config, preprocess)
            if len(arrays["mass"]) != n:
                raise ValueError(f"{entry['id']}: the mesh has {len(arrays['mass'])} vertices, the manifest says {n}")
            for key in ("evecs", "mass", "geo_feat"):
                data[key][slot, :n] = torch.from_numpy(arrays[key])
            data["evals"][slot] = torch.from_numpy(arrays["evals"])
            data["clusters"][slot] = torch.from_numpy(arrays["clusters"])
            self._probes[entry["id"]] = arrays["probes"]
            for name, prefix, count in packet_sets:
                inputs, outputs = np.load(folder / f"{prefix}inputs.npy"), np.load(folder / f"{prefix}outputs.npy")
                if (inputs.shape != (count, n, 2) or outputs.shape != (count, n, config.output_steps, 2)
                        or inputs.dtype != np.float32 or outputs.dtype != np.float32):
                    raise ValueError(f"{entry['id']}: {prefix}inputs.npy or {prefix}outputs.npy has the wrong shape "
                                     "or is not float32")
                if not np.isfinite(outputs).all():
                    raise ValueError(f"{entry['id']}: {prefix}outputs.npy has values that are not finite")
                if not np.array_equal(inputs, outputs[:, :, 0]):
                    raise ValueError(f"{entry['id']}: {prefix}inputs.npy is not the stored state at t = 0")
                position = positions[family]
                data["inputs"][position:position + count, :n] = torch.from_numpy(amplitude * inputs)
                data["labels"][position:position + count, :n] = torch.from_numpy(
                    amplitude * outputs[:, :, first:].reshape(count, n, self.c_out))
                params.append(np.load(folder / f"{prefix}params.npy"))
                diagnostics.append(np.load(folder / f"{prefix}diagnostics.npy"))
                if len(params[-1]) != count or len(diagnostics[-1]) != count:
                    raise ValueError(f"{entry['id']}: {prefix}params.npy or {prefix}diagnostics.npy has the wrong length")
                for key, value in (("shape", entry["index"]), ("set", SETS.index(name)),
                                   ("family", _FAMILY_CODE[family]), ("round", entry["round"]), ("n_verts", n),
                                   ("slot", slot)):
                    columns[key].append(np.full(count, value))
                columns["packet"].append(np.arange(count))
                columns["position"].append(np.arange(position, position + count))
                positions[family] += count
            slots[family] += 1
            seconds = time.perf_counter() - start
            self.shapes.append({"index": entry["index"], "id": entry["id"], "family": family, "split": entry["split"],
                                "round": entry["round"], "n_verts": n, "slot": slot, "seconds": seconds})
            if progress is not None:
                progress(f"{entry['id']}: {n} vertices, {sum(c for _, _, c in packet_sets)} packets, {seconds:.1f} s")
        self.table = {key: np.concatenate(values) for key, values in columns.items()}
        self.params, self.diagnostics = np.concatenate(params), np.concatenate(diagnostics)
        self.nbytes = sum(tensor.numel() * tensor.element_size()
                          for data in self._families.values() for tensor in data.values())

    def __len__(self) -> int:
        return len(self.table["shape"])

    def rows(self, *, sets: str | Iterable[str] | None = None, family: str | None = None,
             is_round: bool | None = None, shapes: int | Iterable[int] | None = None) -> NDArray:
        """Packet numbers, ascending, of the packets that pass every given filter."""
        keep = np.ones(len(self), dtype=bool)
        if sets is not None:
            sets = (sets,) if isinstance(sets, str) else tuple(sets)
            if not set(sets) <= set(SETS):
                raise ValueError(f"unknown packet sets {sorted(set(sets) - set(SETS))}; there are {SETS}")
            keep &= np.isin(self.table["set"], [SETS.index(name) for name in sets])
        if family is not None:
            if family not in _FAMILY_CODE:
                raise ValueError(f"unknown family {family!r}; there are {FAMILIES}")
            keep &= self.table["family"] == _FAMILY_CODE[family]
        if is_round is not None:
            if not isinstance(is_round, (bool, np.bool_)):
                raise ValueError("is_round must be True, False or None")
            keep &= self.table["round"] == bool(is_round)
        if shapes is not None:
            keep &= np.isin(self.table["shape"], _shape_indices(shapes))
        return np.flatnonzero(keep)

    def _checked(self, rows, *, allow_empty: bool = False) -> NDArray:
        rows = np.asarray(rows)
        if rows.ndim != 1:
            raise ValueError("rows must be a list of packet numbers")
        if len(rows) == 0:
            if allow_empty:
                return np.zeros(0, dtype=np.int64)
            raise ValueError("rows is empty")
        if not np.issubdtype(rows.dtype, np.integer) or rows.min() < 0 or rows.max() >= len(self):
            raise ValueError(f"packet numbers must be integers in 0..{len(self) - 1}")
        return rows.astype(np.int64)

    def batch(self, rows, device: str | torch.device | None = None) -> dict:
        """The packets ``rows`` (all of one family) as one batch for ``GLNONet``, moved to ``device`` if given.

        ``input`` (B, N, 2), ``labels`` (B, N, c_out), ``mass`` and ``geo_feat`` (B, N), ``evals``
        (B, K) and ``evecs`` (B, N, K), with N the family's padded size; and on the CPU ``clusters``
        (B, K), ``n_verts`` (B,) and ``rows`` (B,). The tensors are copies: changing them leaves
        the store as it is.
        """
        rows = self._checked(rows)
        codes = np.unique(self.table["family"][rows])
        if len(codes) != 1:
            raise ValueError("a batch holds packets of one family")
        data = self._families[FAMILIES[codes[0]]]
        slots = torch.from_numpy(self.table["slot"][rows])
        shape, packet = slots.to(self.device), torch.from_numpy(self.table["position"][rows]).to(self.device)
        batch = {"input": data["inputs"][packet], "labels": data["labels"][packet], "mass": data["mass"][shape],
                 "evals": data["evals"][shape], "evecs": data["evecs"][shape], "geo_feat": data["geo_feat"][shape]}
        if device is not None:
            batch = {key: value.to(device) for key, value in batch.items()}
        return batch | {"clusters": data["clusters"][slots], "n_verts": torch.from_numpy(self.table["n_verts"][rows]),
                        "rows": torch.from_numpy(rows)}

    def oracle_error(self, batch: dict) -> NDArray:
        """(B, T) spectral oracle of a batch, in float64: the Crank–Nicolson factor of each mode, applied in
        the batch's own ``evecs`` and ``evals`` to its ``input``, against its ``labels``.

        It is ``spectral.spectral_oracle_error`` computed from what the network is given, so it also
        checks the scales, the padding and the layout of the labels. With the Ritz basis it is the
        error the network could reach (Plan §1.5).
        """
        amplitude, scale = self.preprocess.amplitude, self.preprocess.eval_scale
        errors = np.empty((len(batch["rows"]), len(self.steps)))
        for i, n in enumerate(batch["n_verts"].tolist()):
            evecs, evals, mass, start, labels = (batch[key][i].double().cpu().numpy()
                                                 for key in ("evecs", "evals", "mass", "input", "labels"))
            target = labels[:n].reshape(n, len(self.steps), 2)
            errors[i] = spectral_oracle_error(evecs[:n], evals * scale, mass[:n],
                                              (start[:n, 0] + 1j * start[:n, 1]) / amplitude,
                                              (target[..., 0] + 1j * target[..., 1]) / amplitude, self.steps, self.dt)
        return errors

    def group_positions(self, rows) -> dict:
        """Where each reporting group sits in ``rows``: positions into ``rows``, not packet numbers.

        ``sphere`` and ``torus`` leave the tori of revolution out, ``round`` holds them and
        ``select`` is everything else (their copies in validation and test are near-copies of the
        training ones, so they are reported apart). ``rows`` must be in the order of the values
        the positions will index: for results collected batch by batch that is the batches'
        order, see ``family_batches``.
        """
        rows = self._checked(rows, allow_empty=True)
        family, is_round = self.table["family"][rows], self.table["round"][rows]
        groups = {"select": ~is_round, "round": is_round}
        groups |= {name: (family == _FAMILY_CODE[name]) & ~is_round for name in FAMILIES}
        return {name: np.flatnonzero(groups[name]) for name in GROUPS}

    def fingerprint(self) -> dict:
        """What identifies the basis of the loaded shapes, and only the basis: the scales, the amplitude and
        the rest of ``preprocess`` belong in the run's meta.

        Compare two of them with ``fingerprint_mismatch``, never with ``==``: the numbers change in
        their last digits with the BLAS thread count and the CPU. In a checkpoint it goes beside
        the meta, not inside a dict that is compared for equality.
        """
        return {"version": FINGERPRINT_VERSION, "ritz_version": RITZ_VERSION, "identity": self.identity,
                "basis": self.preprocess.basis, "k_eig": self.k_eig,
                "shapes": {name: probes.tolist() for name, probes in self._probes.items()}}


def family_batches(store: ShapeStore, rows, batch_size: int, *, seed: int | None = None,
                   epoch: int = 0) -> list[NDArray]:
    """Cut ``rows`` into batches of one family each.

    With a ``seed``, the packets are shuffled inside each family, cut, and the batches shuffled,
    all from ``(seed, epoch)`` and the set of ``rows``, not their order: a resumed run sees the
    batches of an uninterrupted one. Without it the packets keep their order inside each family
    and the families follow one another (for evaluation). The last batch of a family may be
    smaller.

    Either way ``np.concatenate(batches)`` is not ``rows`` when ``rows`` mixes families. File
    per-packet results by ``batch["rows"]`` (packet numbers), or give ``np.concatenate(batches)``,
    not ``rows``, to ``group_positions``, ``params`` and ``diagnostics``.
    """
    rows = store._checked(rows, allow_empty=True)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("rows repeats a packet")
    rng = None if seed is None else np.random.default_rng([seed, epoch])
    family = store.table["family"][rows]
    batches = []
    for code in np.unique(family):
        members = rows[family == code]
        if rng is not None:
            members = rng.permutation(np.sort(members))
        batches += [members[start:start + batch_size] for start in range(0, len(members), batch_size)]
    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


def group_means(values, groups: dict) -> dict:
    """The mean over the first axis of ``values`` for each group of ``ShapeStore.group_positions``.

    A group without packets (a set with no torus, say) gives ``None`` instead of a NaN or an error.
    ``values`` holds one entry per row given to ``group_positions``, in the same order.
    """
    values = np.asarray(values)
    count = len(groups["select"]) + len(groups["round"])            # the two split the rows
    if len(values) != count:
        raise ValueError(f"{len(values)} values for groups of {count} rows")
    return {name: values[positions].mean(axis=0) if len(positions) else None for name, positions in groups.items()}


def fingerprint_mismatch(a: dict, b: dict, *, tolerance: float = 1e-3, eval_tolerance: float = 1e-6) -> str | None:
    """Why two fingerprints describe different bases, or ``None`` when they agree on the shapes they share.

    The 16 basis probes of a shape move by O(1) when one eigenvector changes sign and by ~1e-5
    between LAPACK drivers (round tori, whose nearly equal pairs rotate by ~1e-6), so they are
    compared to ``tolerance`` in absolute value; the 4 eigenvalue probes to ``eval_tolerance``,
    relative.
    """
    for fingerprint in (a, b):
        if not (isinstance(fingerprint, dict) and isinstance(fingerprint.get("shapes"), dict)):
            return "not a fingerprint"
    for key in ("version", "ritz_version", "identity", "basis", "k_eig"):
        if a.get(key) != b.get(key):
            return f"{key} differs: {a.get(key)!r} and {b.get(key)!r}"
    shared = [name for name in a["shapes"] if name in b["shapes"]]
    if not shared:
        return "no shape in common"
    for name in shared:
        first, second = np.asarray(a["shapes"][name], dtype=float), np.asarray(b["shapes"][name], dtype=float)
        if (first.shape != second.shape or first.ndim != 1 or len(first) <= 4
                or not (np.isfinite(first).all() and np.isfinite(second).all())):
            return f"{name}: malformed probes"
        change = np.abs(first[:-4] - second[:-4]).max()
        if change > tolerance:
            return f"{name}: the basis differs (probes move by {change:.2e})"
        scale = np.maximum(np.abs(first[-4:]), np.abs(second[-4:]))
        change = np.divide(np.abs(first[-4:] - second[-4:]), scale, out=np.zeros(4), where=scale > 0).max()
        if change > eval_tolerance:
            return f"{name}: the eigenvalues differ (relative change {change:.2e})"
    return None


def training_geo_scale(directory: Path) -> float:
    """Largest ``‖x - c‖`` over the training shapes: the ``geo_scale`` that puts their ``geo_feat`` in [0, 1]."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    largest = 0.0
    for entry in manifest["shapes"]:
        if entry["split"] == "train":
            with np.load(directory / entry["id"] / "mesh.npz") as stored:
                verts, mass = stored["verts"], stored["mass"]
            largest = max(largest, float(np.linalg.norm(verts - centroid(verts, mass), axis=1).max()))
    if largest == 0.0:
        raise ValueError("the dataset has no training shape")
    return largest
