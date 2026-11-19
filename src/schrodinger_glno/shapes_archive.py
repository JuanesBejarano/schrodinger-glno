"""A stage-3 dataset as deterministic tars, for the trip to Google Drive and onto the disk of a Colab machine.

``pack`` turns a complete dataset directory (``shapes_dataset``) into a few uncompressed tars and an index,
``shapes_tars.json``; ``unpack`` rebuilds the directory from them and hands it over only when every SHA-256 of
its manifest holds. The link is slow and Drive, a FUSE mount, can serve a file that is truncated, stale or
still syncing, so every step is checked against a hash and nothing appears under the dataset's name before
the last check has passed.

**What travels.** The manifest decides (``dataset_files``): ``manifest.json``, ``config.json`` and, for every
shape, the files the manifest hashes plus ``shape.json``. Anything else in the directory (a ``.partial`` left
by a run, a log) is not part of the dataset and is not packed.

**Packing** (``pack``) verifies the dataset in full first, and then does not rely on that pass for what it
archives minutes later. Every data file is hashed again while it goes into its tar and must still have the
SHA-256 of the manifest. The JSON files, which the manifest does not hash, are packed from memory, as bytes
that were the same before and after the verification. A file that changes, or is read wrong, in between
stops the pack, instead of being found in Colab after the upload.

**Tars.** One per split for val and test, and the training split in ``train_parts`` tars of whole, consecutive
shapes, so an upload or a copy that is cut costs one tar and each tar is a complete archive of its own (not a
slice of a larger one)::

    shapes_<identity[:16]>_train_<k>of<N>.tar     shapes_<identity[:16]>_val.tar     shapes_<identity[:16]>_test.tar

The identity is in the name because Drive keeps two files of one name side by side: the tars of a dataset
that was generated again can then never be taken for the old ones. Members are paths from the dataset's
root (``000_sphere/mesh.npz``), with no directory entries, so ``tar -xf`` of any tar into a folder works too.

**manifest.json and config.json travel in every tar,** as its first two members. A tar extracted alone then
says which dataset it belongs to and which hash each of its files must have, and there is no extra small
file whose absence would make the others useless. The copies are the same bytes, so extracting several tars
overwrites them with themselves; the cost is 0.66 MB per tar of data/shapes (0.05 % in all).

**Determinism.** Packing twice, on any machine and at any time, gives the same bytes, so the SHA-256 of a tar
identifies an upload. Of a file only its name, its size and its bytes reach the tar: the members go in a
fixed order (the two metadata files, then the shapes by index, the files of a shape by name) with mode 0644,
uid = gid = 0, empty user and group names and time 0. The format is ustar (POSIX.1-1988), which every tool
reads: its header has no field beyond those (pax would add records, GNU its own), and a name or a size that
does not fit in it is an error, not an extension. There is no compression (zstd would save about 9 %), hence no
compressor version either, and the index holds no date and no path.

**Index** (``shapes_tars.json``, in the directory of the tars)::

    format_version   of this module (ARCHIVE_VERSION)
    identity         of the dataset (its manifest's), 64 hex characters
    files, bytes     of the extracted dataset: what the free-space check asks for
    tars             a list, in the order above, of
        name, split, shapes (ids)     the file and what it holds
        size, sha256                  of the tar
        members                       {path: size} of every member

**Unpacking** (``unpack``), in this order:

1. The index is read and checked; an expected identity, or its first 16 characters, is enforced.
2. If the target already holds that dataset, verified (every SHA-256), nothing is done: the tars are not even
   looked for. A target that holds anything else is refused, never deleted: a wrong path would otherwise cost
   a dataset of hours. An empty directory is used; a symbolic link that does not lead to the dataset is
   refused, since the temporary directory of step 5 has to be a sibling of the directory itself.
3. Every tar must exist with exactly its listed size, before anything is extracted; all the missing or
   truncated ones are named at once. ``check_sha256`` also hashes them first, at the price of reading them twice.
4. A ``<target>.unpacking`` left by an interrupted run is removed, and the free space is checked.
5. The tars are extracted into ``<target>.unpacking``, a sibling of the target and so on its file system
   (the caller gives a target on the local disk), with tarfile's ``data`` filter. They are read straight from
   where they are, in one sequential pass that hashes the bytes as tarfile consumes them: the SHA-256 and the
   size of each tar are checked against the index as soon as it is extracted, and a bad one stops the run by
   its name. Copying them to the local disk first would take twice the space (17.7 GB instead of 8.8) and
   one more pass, and would catch nothing this does not. The price is that a failed run keeps nothing, and
   the next one reads every tar again; so a tar that cannot be read, or whose bytes are not the index's, is
   read a second time before the run gives up, and a hiccup of Drive costs one tar instead of all of them.
6. The extracted directory is verified in full (identity, every SHA-256, ``shape.json``, ``config.json``, and
   that the tars held exactly the manifest's files); only then is it renamed to the target. Any failure
   removes ``<target>.unpacking``, so a failed run leaves no target at all.

**What a tar may hold.** Only regular files whose path the index lists with that size, each once, and the
index may only list paths of the dataset's layout (``manifest.json``, ``config.json``, ``<shape id>/<file>`` with
plain names). An absolute path, a ``..``, a link, a device, a directory entry or an unlisted file is refused
with the tar's name, before it is extracted and although the ``data`` filter would also stop most of them:
nothing is skipped or repaired, since a tar that is not what ``pack`` wrote should not be trusted in part.

**One file to upload** (``zip_tars``, ``read_zip_index``, ``unzip_tars``). A folder of several files is awkward to drag into Drive,
so the tars and their index can also travel inside one zip, ``glno_<dataset>.zip`` (``zip_name``). The zip is
only a container: it holds exactly ``shapes_tars.json`` and the tars it lists, each hashed on its way in.
Deflate takes about 9 % off these tars at any level, so the fastest level is used. In Colab ``unzip_tars``
copies the tars out of the zip onto the local disk, where each must again have the size and the SHA-256 of
the index, and ``unpack`` then works on that folder as on one uploaded as it is. The tars are on the local
disk twice for a moment (in the zip's folder and extracted): 17.7 GB for data/shapes, which the free-space
check asks for before anything is written. As with ``unpack``, the folder appears under its name only when
it is complete, a second call does nothing, and a folder that holds something else is refused, not deleted.

Only the standard library and ``shapes_dataset``: the module is copied into the Colab notebook's package.
"""
from contextlib import contextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time
from typing import Callable
import zipfile
import zlib
from .shapes_dataset import (CONFIG_FILE, FORMAT_VERSION, MANIFEST_FILE, SHAPE_FILE, SPLITS, _write_json,
                             shapes_dataset_identity)

ARCHIVE_VERSION = 1
INDEX_FILE = "shapes_tars.json"
METADATA_FILES = (MANIFEST_FILE, CONFIG_FILE)
TEMPORARY_SUFFIX = ".unpacking"
ZIP_TEMPORARY_SUFFIX = ".unzipping"
ZIP_LEVEL = 1                   # deflate leaves 91 % of these tars at level 1 and at level 6
MEMBER_MODE, MEMBER_MTIME = 0o644, 0
BLOCK = 2**20
# Room for what a file system adds to the bytes of the files: each one rounded up to a block, and its own records.
SPACE_MARGIN = 64 * 2**20
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_HASH = re.compile(r"[0-9a-f]{64}")


class _Hashed:
    """A binary file that hashes and counts the bytes passing through it, so a tar is written, or read, only once."""

    def __init__(self, raw) -> None:
        self.raw, self.sha256, self.size = raw, sha256(), 0

    def _count(self, block: bytes) -> bytes:
        self.sha256.update(block)
        self.size += len(block)
        return block

    def write(self, block: bytes) -> int:
        return self.raw.write(self._count(block))

    def read(self, size: int) -> bytes:
        return self._count(self.raw.read(size))

    def tell(self) -> int:
        return self.size

    def drain(self) -> None:
        """Read to the end: tarfile stops at the end-of-archive blocks, before the padding that follows them."""
        while self.read(BLOCK):
            pass


def _in_layout(name) -> bool:
    """Whether ``name`` is a path of a dataset: a metadata file, or ``<shape id>/<file>`` with plain names.

    A plain name starts with a letter, a digit or ``_``, which leaves out ``..``, absolute paths (an empty
    first part), backslashes and drive letters.
    """
    parts = name.split("/") if isinstance(name, str) else []
    return name in METADATA_FILES if len(parts) == 1 else len(parts) == 2 and all(map(_NAME.fullmatch, parts))


def dataset_files(manifest: dict) -> dict[str, list[str]]:
    """``{shape id: its files}`` as sorted paths from the dataset's root: those the manifest hashes, and ``shape.json``.

    A manifest that names a file outside the layout is refused: ``pack`` would write a tar ``unpack`` rejects.
    """
    files = {entry["id"]: [] for entry in manifest["shapes"]}
    for relative in [*manifest["sha256"], *(f"{shape_id}/{SHAPE_FILE}" for shape_id in files)]:
        shape_id = relative.split("/", 1)[0]
        if not _in_layout(relative) or shape_id not in files or relative in files[shape_id]:
            raise ValueError(f"{MANIFEST_FILE} lists a file outside the dataset layout: {relative!r}")
        files[shape_id].append(relative)
    return {shape_id: sorted(names) for shape_id, names in files.items()}


@contextmanager
def _dataset_errors(directory: Path):
    """Turn what a missing file, or a manifest of another form, raises into a ValueError that names the file."""
    try:
        yield
    except FileNotFoundError as error:
        raise ValueError(f"Dataset file is missing: {error.filename}") from error
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{directory / MANIFEST_FILE} is not the manifest of a shapes dataset: {error!r}") from error


def _json(path: Path, data: bytes | None = None):
    """What the JSON file ``path`` says (``data``: its bytes, if already read); a damaged one is named.

    ``json`` alone gives a line and a column, which among the 200 ``shape.json`` of a dataset say nothing.
    """
    try:
        return json.loads(path.read_bytes() if data is None else data)
    except ValueError as error:                         # also the UnicodeDecodeError of bytes that are not text
        raise ValueError(f"{path} is not JSON: {error}") from error


def verified_identity(directory: Path) -> str:
    """The identity of the dataset in ``directory``, after checking every SHA-256, ``shape.json`` and ``config.json``.

    ``shapes_dataset_identity`` does not read ``config.json`` (``dataset_summary`` does), so it is compared
    here with the manifest's copy of the config. The JSON files are parsed first: a damaged one is then
    named, and found before the pass over the data instead of after it.
    """
    directory = Path(directory)
    with _dataset_errors(directory):
        manifest, stored = (_json(directory / name) for name in METADATA_FILES)
        for entry in manifest["shapes"]:
            _json(directory / entry["id"] / SHAPE_FILE)
        identity = shapes_dataset_identity(directory, verify=True)
        if stored != {"format_version": FORMAT_VERSION, "config": manifest["config"]}:
            raise ValueError(f"Dataset file differs from manifest: {CONFIG_FILE}")
    return identity


def _json_files(directory: Path) -> tuple[dict, dict[str, bytes]]:
    """The manifest and ``{path: bytes}`` of a dataset's JSON files: the manifest, the config, every ``shape.json``.

    The manifest hashes none of them, so ``pack`` cannot check them while it archives them, as it does the
    data files: it reads them before the verification and again after it, and packs the bytes it holds.
    """
    held = {MANIFEST_FILE: (directory / MANIFEST_FILE).read_bytes()}
    manifest = _json(directory / MANIFEST_FILE, held[MANIFEST_FILE])
    for name in (CONFIG_FILE, *(f"{entry['id']}/{SHAPE_FILE}" for entry in manifest["shapes"])):
        held[name] = (directory / name).read_bytes()
    return manifest, held


def _tar_groups(manifest: dict, train_parts: int) -> list[tuple[str, str, list[str]]]:
    """``(tar name, split, shape ids)`` of each tar: train in ``train_parts`` runs of consecutive shapes, val, test."""
    shapes = sorted(manifest["shapes"], key=lambda entry: entry["index"])
    ids = {split: [entry["id"] for entry in shapes if entry["split"] == split] for split in SPLITS}
    train = ids["train"]
    if not 1 <= train_parts <= max(1, len(train)):
        raise ValueError(f"train_parts must be between 1 and the number of training shapes ({len(train)})")
    prefix = f"shapes_{manifest['identity'][:16]}"
    cuts = [len(train) * k // train_parts for k in range(train_parts + 1)]
    groups = [(f"{prefix}_train_{k + 1}of{train_parts}.tar", "train", train[cuts[k]:cuts[k + 1]])
              for k in range(train_parts)]
    groups += [(f"{prefix}_{split}.tar", split, ids[split]) for split in SPLITS if split != "train"]
    return [group for group in groups if group[2]]


def _member(name: str, size: int) -> tarfile.TarInfo:
    """The header of a member: a regular file with nothing of the machine or the clock (see the module docstring)."""
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime, info.type = size, MEMBER_MODE, MEMBER_MTIME, tarfile.REGTYPE
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _add(tar: tarfile.TarFile, directory: Path, name: str, hashes: dict, held: dict) -> int:
    """Add the file ``name`` to the tar and return its size: the bytes in ``held``, or else those on the disk.

    The verification read the disk a while ago, and what is archived is read now: a data file is hashed on
    its way into the tar, and one that no longer has the SHA-256 of the manifest stops the pack.
    """
    if name in held:
        tar.addfile(_member(name, len(held[name])), io.BytesIO(held[name]))
        return len(held[name])
    with (directory / name).open("rb") as raw:
        source, size = _Hashed(raw), os.fstat(raw.fileno()).st_size
        tar.addfile(_member(name, size), source)
    if source.sha256.hexdigest() != hashes[name]:
        raise ValueError(f"Dataset file changed while it was packed: {name}")
    return size


def _write_tar(directory: Path, path: Path, names: list[str], hashes: dict, held: dict) -> dict:
    """Write the files ``names`` of ``directory`` as a deterministic tar; return its size, sha256 and members.

    ``hashes`` is the manifest's ``{path: sha256}`` and ``held`` the files to take from memory (``_add``).
    The temporary file is removed if the writing fails: after a full disk it would be what keeps it full.
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as raw:
            stream = _Hashed(raw)
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT, copybufsize=BLOCK) as tar:
                members = {name: _add(tar, directory, name, hashes, held) for name in names}
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"size": stream.size, "sha256": stream.sha256.hexdigest(), "members": members}


def pack(directory: Path, out: Path, *, train_parts: int = 1,
         progress: Callable[[str], None] | None = None) -> dict:
    """Verify the dataset in ``directory`` (every SHA-256) and write its tars and ``shapes_tars.json`` in ``out``.

    Returns the index. ``train_parts`` is the number of tars of the training split (4 for data/shapes: about
    1.75 GB each). Tars of the same names are replaced, each written under a temporary name and renamed; the
    index is removed first and written last, so an interrupted run leaves no index of tars it did not finish.
    A file that is not the verified one when it is archived is a ValueError (see the module docstring).
    ``progress`` gets one line for the verification and one per tar.
    """
    directory, out = Path(directory), Path(out)
    say = progress or (lambda message: None)
    if out.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"{out} is inside the dataset {directory}: the tars would be taken for a part of it")
    with _dataset_errors(directory):
        manifest, held = _json_files(directory)
        # Before the verification, which reads the whole dataset: a wrong train_parts should not cost that pass.
        files, groups = dataset_files(manifest), _tar_groups(manifest, train_parts)
    identity = verified_identity(directory)
    say(f"verified {directory}: identity {identity[:16]}")
    with _dataset_errors(directory):
        changed = [name for name, data in _json_files(directory)[1].items() if held.get(name) != data]
    if changed:
        raise ValueError(f"Dataset file changed while it was verified: {changed[0]}")
    out.mkdir(parents=True, exist_ok=True)
    (out / INDEX_FILE).unlink(missing_ok=True)
    tars = []
    for name, split, shape_ids in groups:
        names = [*METADATA_FILES, *(relative for shape_id in shape_ids for relative in files[shape_id])]
        tars.append({"name": name, "split": split, "shapes": shape_ids}
                    | _write_tar(directory, out / name, names, manifest["sha256"], held))
        say(f"{name}: {len(shape_ids)} {split} shapes, {tars[-1]['size'] / 1e9:.3f} GB")
    extracted = {name: size for tar in tars for name, size in tar["members"].items()}
    index = {"format_version": ARCHIVE_VERSION, "identity": identity, "files": len(extracted),
             "bytes": sum(extracted.values()), "tars": tars}
    _write_json(out / INDEX_FILE, index)
    return index


def _index_problem(index) -> str | None:
    """What is wrong with a parsed index, or None. Its paths are checked here because they end up on the disk."""
    def is_size(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def is_hash(value) -> bool:
        return isinstance(value, str) and _HASH.fullmatch(value) is not None

    try:
        if index["format_version"] != ARCHIVE_VERSION:
            return f"format_version is {index['format_version']!r}, not {ARCHIVE_VERSION}"
        if not is_hash(index["identity"]):
            return "identity is not a SHA-256"
        names = [entry["name"] for entry in index["tars"]]
        if not names or len(set(names)) != len(names):
            return "it lists no tar, or one twice"
        files = {}
        for entry in index["tars"]:
            name, members = entry["name"], entry["members"]
            if not (isinstance(name, str) and _NAME.fullmatch(name) and name.endswith(".tar")):
                return f"{name!r} is not the plain name of a tar"
            if not (is_size(entry["size"]) and is_hash(entry["sha256"])):
                return f"{name} has no valid size and sha256"
            if not set(METADATA_FILES) <= set(members):
                return f"{name} lacks {' or '.join(METADATA_FILES)}"
            for member, size in members.items():
                if not (_in_layout(member) and is_size(size)):
                    return f"{name} lists {member!r}, which is outside the dataset layout"
                if member in files and (member not in METADATA_FILES or files[member] != size):
                    return f"{member} is listed in two tars"
                files[member] = size
        if (index["files"], index["bytes"]) != (len(files), sum(files.values())):
            return "files and bytes are not those of its members"
    except (KeyError, TypeError, AttributeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def _index(path: Path, data: bytes | None = None) -> dict:
    """The index in ``path``, parsed and checked (``data``: its bytes, if already read)."""
    index = _json(path, data)
    problem = _index_problem(index)
    if problem:
        raise ValueError(f"{path} is not a valid index of shapes tars: {problem}")
    return index


def read_index(tars_dir: Path) -> dict:
    """The index ``shapes_tars.json`` of the tars in ``tars_dir``; a missing or malformed one is a ValueError."""
    path = Path(tars_dir) / INDEX_FILE
    try:
        return _index(path)
    except FileNotFoundError:
        raise ValueError(f"Missing index of the tars: {path}") from None


def _check_expected(identity: str, expected: str | None, source: Path) -> None:
    """Refuse an index of another dataset than the ``expected`` identity (all of it, or 16 characters or more)."""
    if expected is None:
        return
    if not (isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{16,64}", expected)):
        raise ValueError(f"The expected identity must be its 64 hex characters or the first 16, not {expected!r}")
    if not identity.startswith(expected):
        raise ValueError(f"{source} is of the dataset {identity[:16]}, not of the expected {expected[:16]}")


def _target_problem(target: Path, identity: str) -> str | None:
    """Why ``target`` is not the verified dataset ``identity``, or None when it is."""
    try:
        found = verified_identity(target)
    except (OSError, ValueError) as error:
        return str(error)
    return None if found == identity else f"it holds the dataset {found[:16]}"


def _tar_digest(path: Path) -> str:
    with path.open("rb") as raw:
        stream = _Hashed(raw)
        stream.drain()
    return stream.sha256.hexdigest()


def _tar_problem(path: Path, entry: dict, check_sha256: bool) -> str | None:
    """What is wrong with a tar before it is extracted, to follow its name in a message; None if nothing is."""
    try:
        if not path.is_file():
            return "is missing"
        size = path.stat().st_size
        if size != entry["size"]:
            return f"has {size} bytes, not {entry['size']} (truncated, or still uploading?)"
        if check_sha256 and _tar_digest(path) != entry["sha256"]:
            return "has another SHA-256 than the index"
    except OSError as error:
        return f"could not be read: {error}"
    return None


def _check_tars(tars_dir: Path, index: dict, check_sha256: bool) -> None:
    """Every tar of the index is there with its exact size (and SHA-256, when asked), or a ValueError names them all.

    All at once because, over a slow link, each upload that has to be repeated costs a while.
    """
    problems = [f"{entry['name']} {problem}" for entry in index["tars"]
                if (problem := _tar_problem(tars_dir / entry["name"], entry, check_sha256))]
    if problems:
        raise ValueError(f"The tars in {tars_dir} are not those of {INDEX_FILE}: " + "; ".join(problems))


def _check_space(folder: Path, needed: int) -> None:
    free = shutil.disk_usage(folder).free
    if free < needed + SPACE_MARGIN:
        raise ValueError(f"Not enough disk space in {folder}: the dataset needs {(needed + SPACE_MARGIN) / 1e9:.2f} GB "
                         f"and {free / 1e9:.2f} GB are free")


def _check_member(path: Path, member: tarfile.TarInfo, pending: dict) -> None:
    """Refuse a member that is not a regular file the index still expects, with that size; take it off ``pending``."""
    if member.type != tarfile.REGTYPE:
        raise ValueError(f"{path} holds {member.name!r}, which is not a regular file (a link, a device, a directory?); "
                         "the tar is refused")
    if not _in_layout(member.name) or pending.pop(member.name, None) != member.size:
        raise ValueError(f"{path} holds {member.name!r} ({member.size} bytes), which is not a file that {INDEX_FILE} "
                         "lists for it, once and with that size; the tar is refused")


class _ReadError(ValueError):
    """A tar that could not be read, or whose bytes are not those of the index: what a second read may cure."""


def _extract_tar(path: Path, entry: dict, folder: Path) -> None:
    """Extract one tar into ``folder`` in a single sequential read, and check it against its index ``entry``.

    Each member is checked before it is extracted (``_check_member``); the size and the SHA-256 of the tar
    are known when its last byte has gone by. Extracting it again overwrites its own files.
    """
    pending = dict(entry["members"])
    try:
        with path.open("rb") as raw:
            stream = _Hashed(raw)
            # Mode "r|" reads forward only, and a plain tar only. tarfile's defaults read the file 10 kB at a
            # time and copy a member 16 kB at a time; one larger block for both keeps the reads from Drive few.
            with tarfile.open(fileobj=stream, mode="r|", bufsize=BLOCK, copybufsize=BLOCK) as tar:
                for member in tar:
                    _check_member(path, member, pending)
                    tar.extract(member, folder, filter="data")
            stream.drain()
    except (tarfile.TarError, OSError) as error:
        raise _ReadError(f"{path} could not be extracted: {error}") from error
    # The bytes before the files: tarfile takes a damaged header, or a tar cut between two members, for the end
    # of the archive, and that is a read that went wrong, not a tar that lacks files.
    if (stream.size, stream.sha256.hexdigest()) != (entry["size"], entry["sha256"]):
        raise _ReadError(f"{path} differs from {INDEX_FILE}: {stream.size} bytes with SHA-256 "
                         f"{stream.sha256.hexdigest()[:16]}, not {entry['size']} with {entry['sha256'][:16]} "
                         "(a stale or damaged copy?)")
    if pending:
        raise ValueError(f"{path} lacks {len(pending)} of the files {INDEX_FILE} lists, e.g. {next(iter(pending))}")


def _extract_with_retry(path: Path, entry: dict, folder: Path, say: Callable[[str], None]) -> None:
    """Extract one tar, and once more if it could not be read; a tar with a member that is refused is not read again."""
    try:
        _extract_tar(path, entry, folder)
    except _ReadError as error:
        say(f"{error}; reading {path.name} again")
        _extract_tar(path, entry, folder)


def _verify_extracted(folder: Path, index: dict) -> None:
    """The extracted dataset is the index's: identity, every SHA-256, and exactly the files of its manifest."""
    identity = verified_identity(folder)
    if identity != index["identity"]:
        raise ValueError(f"The tars hold the dataset {identity[:16]}, not {index['identity'][:16]} of {INDEX_FILE}")
    manifest = _json(folder / MANIFEST_FILE)
    listed = {*METADATA_FILES, *(name for names in dataset_files(manifest).values() for name in names)}
    odd = sorted(listed ^ {name for entry in index["tars"] for name in entry["members"]})
    if odd:
        raise ValueError(f"The tars and {MANIFEST_FILE} differ in {len(odd)} files, e.g. {odd[0]}")


def _clear(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def unpack(tars_dir: Path, target_dir: Path, *, identity: str | None = None, check_sha256: bool = False,
           progress: Callable[[str], None] | None = None) -> dict:
    """Rebuild in ``target_dir`` the dataset packed in ``tars_dir``, verified; do nothing if it is already there.

    ``target_dir`` goes on a local disk (``/content/data/shapes``); ``tars_dir`` may be on Drive. ``identity``
    is the expected one, or its first 16 characters or more. ``check_sha256`` hashes every tar before
    extracting any (they are hashed while they are extracted anyway). Returns ``identity``, ``directory``,
    ``files``, ``bytes``, ``seconds`` and ``extracted``: False when the target already held the verified
    dataset and nothing was done. Every refusal is a ValueError and leaves ``target_dir`` as it was; the
    steps and their reasons are in the module docstring.
    """
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("unpack needs the extraction filters of tarfile (Python 3.12 or later, 3.11.4+ or "
                           f"3.10.12+); this is Python {sys.version.split()[0]}")
    tars_dir, target = Path(tars_dir), Path(target_dir)
    say = progress or (lambda message: None)
    start = time.perf_counter()
    index = read_index(tars_dir)
    _check_expected(index["identity"], identity, tars_dir / INDEX_FILE)
    result = {key: index[key] for key in ("identity", "files", "bytes")} | {"directory": str(target)}
    link = target.is_symlink()                          # also one that leads nowhere, which exists() does not see
    if link or target.exists():
        problem = _target_problem(target, index["identity"])
        if problem is None:
            say(f"{target} already holds the dataset {index['identity'][:16]}, verified: nothing to do")
            return result | {"extracted": False, "seconds": time.perf_counter() - start}
        # A link cannot be replaced by the rename, and the directory it leads to may be on another file system.
        if link or not target.is_dir() or any(target.iterdir()):
            raise ValueError(f"{target} {'is a symbolic link' if link else 'exists'} and is not the dataset "
                             f"{index['identity'][:16]} ({problem}); it is left as it is: delete it, or give "
                             "another target")
    _check_tars(tars_dir, index, check_sha256)
    temporary = target.with_name(target.name + TEMPORARY_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    _clear(temporary)
    _check_space(target.parent, index["bytes"])
    try:
        temporary.mkdir()
        for entry in index["tars"]:
            lap = time.perf_counter()
            _extract_with_retry(tars_dir / entry["name"], entry, temporary, say)
            say(f"{entry['name']}: {entry['size'] / 1e9:.3f} GB extracted in {time.perf_counter() - lap:.1f} s")
        _verify_extracted(temporary, index)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    say(f"{target}: dataset {index['identity'][:16]} verified, {index['files']} files, {index['bytes'] / 1e9:.3f} GB")
    return result | {"extracted": True, "seconds": time.perf_counter() - start}


def zip_name(dataset: str) -> str:
    """The name of the single file that carries the tars of the dataset ``dataset`` (``shapes``, ``shapes_small``)."""
    return f"glno_{dataset}.zip"


def zip_tars(tars_dir: Path, zip_path: Path, *, progress: Callable[[str], None] | None = None) -> dict:
    """Write the tars of ``tars_dir`` and their index into the single file ``zip_path``, to upload instead of the folder.

    Every tar must be there with its listed size before anything is written, and is hashed while it goes in:
    one that does not have the SHA-256 of the index stops the run. The zip is written under a temporary
    name and renamed, so an interrupted run leaves no zip, and an older one stays as it was until then.
    Returns ``identity``, ``path``, ``size`` (of the zip) and ``bytes`` (of the tars). ``progress`` gets one line
    per tar.
    """
    tars_dir, zip_path = Path(tars_dir), Path(zip_path)
    say = progress or (lambda message: None)
    if zip_path.resolve().parent == tars_dir.resolve():
        raise ValueError(f"{zip_path} is inside the folder of the tars: whoever uploads the folder would upload "
                         "them twice")
    try:
        data = (tars_dir / INDEX_FILE).read_bytes()         # the bytes that are checked are the ones that travel
    except FileNotFoundError:
        raise ValueError(f"Missing index of the tars: {tars_dir / INDEX_FILE}") from None
    index = _index(tars_dir / INDEX_FILE, data)
    _check_tars(tars_dir, index, check_sha256=False)
    temporary = zip_path.with_name(zip_path.name + ".partial")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=ZIP_LEVEL) as archive:
            with archive.open(INDEX_FILE, "w") as member:
                member.write(data)
            for entry in index["tars"]:
                with (tars_dir / entry["name"]).open("rb") as raw, \
                        archive.open(entry["name"], "w", force_zip64=True) as member:
                    stream = _Hashed(raw)
                    while block := stream.read(BLOCK):
                        member.write(block)
                if (stream.size, stream.sha256.hexdigest()) != (entry["size"], entry["sha256"]):
                    raise ValueError(f"{tars_dir / entry['name']} is not the tar of {INDEX_FILE}: {stream.size} bytes "
                                     f"with SHA-256 {stream.sha256.hexdigest()[:16]}, not {entry['size']} with "
                                     f"{entry['sha256'][:16]}; pack the dataset again")
                say(f"{entry['name']}: {entry['size'] / 1e9:.3f} GB into {zip_path.name}")
        os.replace(temporary, zip_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"identity": index["identity"], "path": str(zip_path), "size": zip_path.stat().st_size,
            "bytes": sum(entry["size"] for entry in index["tars"])}


def _holds_tars(folder: Path, data: bytes, index: dict) -> bool:
    """Whether ``folder`` holds this very index and each of its tars with its size."""
    try:
        return ((folder / INDEX_FILE).read_bytes() == data
                and all((folder / entry["name"]).stat().st_size == entry["size"] for entry in index["tars"]))
    except OSError:
        return False


def _unzip_tar(archive: zipfile.ZipFile, zip_path: Path, entry: dict, path: Path) -> None:
    """Copy one tar out of the zip to ``path`` in a single read, and check it against its index ``entry``."""
    try:
        with archive.open(entry["name"]) as member, path.open("wb") as out:
            stream = _Hashed(member)
            while block := stream.read(BLOCK):
                out.write(block)
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as error:
        raise _ReadError(f"{entry['name']} could not be read from {zip_path}: {error}") from error
    if (stream.size, stream.sha256.hexdigest()) != (entry["size"], entry["sha256"]):
        raise _ReadError(f"{entry['name']} in {zip_path} differs from {INDEX_FILE}: {stream.size} bytes with SHA-256 "
                         f"{stream.sha256.hexdigest()[:16]}, not {entry['size']} with {entry['sha256'][:16]} "
                         "(a damaged upload?)")


@contextmanager
def _open_zip(zip_path: Path):
    """The zip, open for reading; a missing one, or one that is not a whole zip, is a ValueError."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except FileNotFoundError:
        raise ValueError(f"Missing zip of the tars: {zip_path}") from None
    except (zipfile.BadZipFile, OSError) as error:
        raise ValueError(f"{zip_path} is not a complete zip (truncated, or still uploading?): {error}") from error
    with archive:
        yield archive


def _zip_index(archive: zipfile.ZipFile, zip_path: Path) -> tuple[bytes, dict]:
    """The index inside an open zip, as bytes and parsed; the zip must hold exactly it and its tars, with their sizes."""
    members = [(info.filename, info.file_size) for info in archive.infolist()]
    if INDEX_FILE not in dict(members):
        raise ValueError(f"{zip_path} holds no {INDEX_FILE}: it is not a zip of shapes tars")
    try:
        data = archive.read(INDEX_FILE)
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as error:
        raise ValueError(f"{INDEX_FILE} could not be read from {zip_path}: {error}") from error
    index = _index(zip_path / INDEX_FILE, data)
    listed = [(INDEX_FILE, len(data)), *((entry["name"], entry["size"]) for entry in index["tars"])]
    if sorted(members) != sorted(listed):
        names = [name for name, _ in members]
        odd = sorted({name for name, _ in set(members) ^ set(listed)}
                     | {name for name in names if names.count(name) > 1})
        raise ValueError(f"{zip_path} does not hold exactly {INDEX_FILE} and its tars, each once and with its "
                         f"size: {', '.join(odd)}")
    return data, index


def read_zip_index(zip_path: Path) -> dict:
    """The index of the tars inside the zip of ``zip_tars``: what ``read_index`` gives for a folder. No tar is read."""
    zip_path = Path(zip_path)
    with _open_zip(zip_path) as archive:
        return _zip_index(archive, zip_path)[1]


def unzip_tars(zip_path: Path, tars_dir: Path, *, identity: str | None = None,
               progress: Callable[[str], None] | None = None) -> dict:
    """Copy the tars and the index out of the zip of ``zip_tars`` into ``tars_dir``, for ``unpack``; do nothing if they are there.

    ``tars_dir`` goes on a local disk (``/content/packed/shapes``), the one of the dataset that ``unpack`` will
    extract from it; ``zip_path`` may be on Drive. ``identity`` is the expected one, or its first 16 characters
    or more, and is enforced before any tar is read. In this order:

    1. The zip must be complete and hold exactly the index and the tars it lists, each with its size.
    2. If ``tars_dir`` already holds that index and those tars, nothing is done (``unpack`` hashes the tars when it
       extracts them). A folder that holds anything else is refused, never deleted; an empty one is used.
    3. There must be room for the tars and for the dataset.
    4. Each tar is copied into ``<tars_dir>.unzipping`` and must have the size and the SHA-256 of the index; one
       that reads wrong is read a second time before the run gives up. The folder is renamed last.

    Returns ``identity``, ``directory``, ``bytes`` (of the tars), ``seconds`` and ``extracted`` (False when nothing
    was done). Every refusal is a ValueError and leaves ``tars_dir`` as it was.
    """
    zip_path, target = Path(zip_path), Path(tars_dir)
    say = progress or (lambda message: None)
    start = time.perf_counter()
    with _open_zip(zip_path) as archive:
        data, index = _zip_index(archive, zip_path)
        _check_expected(index["identity"], identity, zip_path)
        result = {"identity": index["identity"], "directory": str(target),
                  "bytes": sum(entry["size"] for entry in index["tars"])}
        link = target.is_symlink()
        if link or target.exists():
            if not link and _holds_tars(target, data, index):
                say(f"{target} already holds the tars of {zip_path.name}: nothing to do")
                return result | {"extracted": False, "seconds": time.perf_counter() - start}
            if link or not target.is_dir() or any(target.iterdir()):
                raise ValueError(f"{target} {'is a symbolic link' if link else 'exists'} and does not hold the tars "
                                 f"of {zip_path.name}; it is left as it is: delete it, or give another folder")
        temporary = target.with_name(target.name + ZIP_TEMPORARY_SUFFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        _clear(temporary)
        needed, free = result["bytes"] + index["bytes"] + SPACE_MARGIN, shutil.disk_usage(target.parent).free
        if free < needed:
            raise ValueError(f"Not enough disk space in {target.parent}: the tars and the dataset that unpack "
                             f"extracts from them need {needed / 1e9:.2f} GB and {free / 1e9:.2f} GB are free")
        try:
            temporary.mkdir()
            for entry in index["tars"]:
                lap = time.perf_counter()
                try:
                    _unzip_tar(archive, zip_path, entry, temporary / entry["name"])
                except _ReadError as error:
                    say(f"{error}; reading {entry['name']} again")
                    _unzip_tar(archive, zip_path, entry, temporary / entry["name"])
                say(f"{entry['name']}: {entry['size'] / 1e9:.3f} GB out of {zip_path.name} in "
                    f"{time.perf_counter() - lap:.1f} s")
            (temporary / INDEX_FILE).write_bytes(data)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return result | {"extracted": True, "seconds": time.perf_counter() - start}
