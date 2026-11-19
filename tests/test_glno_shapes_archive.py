"""Stage-3 archive: deterministic tars with an index; unpacking verifies everything before the dataset appears."""
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
import pytest
from schrodinger_glno import shapes_archive
from schrodinger_glno.shapes_archive import (INDEX_FILE, dataset_files, pack, read_index, read_zip_index, unpack,
                                             unzip_tars, verified_identity, zip_name, zip_tars)
from schrodinger_glno.shapes_dataset import ShapesConfig, _identity, generate_shapes_dataset, shapes_dataset_identity
from scripts import pack_shapes_dataset

# The six coarse shapes of tests/test_glno_shapes_data.py: 0 sphere and 1 round torus (train), 2 sphere (val),
# 3 torus, 4 sphere and 5 round torus (test, with out-of-distribution packets); two packets per set. 8.2 MB.
TINY = ShapesConfig(train_shapes=2, val_shapes=1, test_shapes=3, packets_per_shape=2, ood_packets_per_shape=2,
                    round_torus_every=2, sphere_subdivisions=4, torus_target_edge=0.12, min_wavenumber=5.0,
                    k_eig=100, final_time=0.05, dt=0.005, min_points_per_wavelength=5.0, k0_range=(0.5, 2.0),
                    sigma_range=(0.15, 0.2), ood_k0_range=(2.0, 3.0), ood_sigma_range=(0.18, 0.2),
                    min_tube_sigmas=2.1, max_leak=2e-3, min_spectral_fraction=0.995, min_ood_spectral_fraction=0.99)
SCRIPT = Path(__file__).parents[1] / "scripts" / "pack_shapes_dataset.py"
METADATA = ["manifest.json", "config.json"]
TRAIN_1, TRAIN_2, VAL, TEST = range(4)                  # the tars of the ``packed`` fixture, in the index's order
NOT_REGULAR = {"symlink": tarfile.SYMTYPE, "hard_link": tarfile.LNKTYPE, "character_device": tarfile.CHRTYPE,
               "block_device": tarfile.BLKTYPE, "fifo": tarfile.FIFOTYPE, "directory": tarfile.DIRTYPE}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shapes_archive") / "data"
    generate_shapes_dataset(TINY, directory, workers=2)
    return directory


@pytest.fixture(scope="module")
def packed(tiny, tmp_path_factory):
    """``(directory, index)`` of the tiny dataset's tars, its two training shapes in two parts. Not to be changed."""
    out = tmp_path_factory.mktemp("shapes_packed") / "tars"
    return out, pack(tiny, out, train_parts=2)


@pytest.fixture
def tars(packed, tmp_path):
    """A copy of the tars that a test may damage."""
    shutil.copytree(packed[0], tmp_path / "tars")
    return tmp_path / "tars"


@pytest.fixture(scope="module")
def zipped(packed, tmp_path_factory):
    """The zip of the tiny dataset's tars and their index. Not to be changed."""
    path = tmp_path_factory.mktemp("shapes_zipped") / zip_name("tiny")
    zip_tars(packed[0], path)
    return path


@pytest.fixture
def no_extraction(monkeypatch):
    """Fail the test if a tar is extracted: the refusal has to come before."""
    def extract(path, entry, folder):
        raise AssertionError(f"{path} was extracted")

    monkeypatch.setattr(shapes_archive, "_extract_tar", extract)


def _tree(directory):
    """``{path from directory: bytes}`` of every file below it."""
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flip(path, offset):
    raw = bytearray(path.read_bytes())
    raw[offset] ^= 1
    path.write_bytes(raw)


def _sign(directory):
    """Rewrite the index with the size and SHA-256 each tar has now, as someone who changed a tar on purpose would."""
    index = json.loads((directory / INDEX_FILE).read_text())
    for entry in index["tars"]:
        entry["size"], entry["sha256"] = (directory / entry["name"]).stat().st_size, _sha(directory / entry["name"])
    (directory / INDEX_FILE).write_text(json.dumps(index))


def _edit_index(directory, change):
    index = json.loads((directory / INDEX_FILE).read_text())
    change(index)
    (directory / INDEX_FILE).write_text(json.dumps(index))


def _rewrite_tar(path, first=(), last=(), keep=None):
    """Rewrite a tar: the members ``(TarInfo, bytes or None)`` of ``first``, its own first ``keep`` (all), ``last``."""
    with tarfile.open(path) as tar:
        own = [(member, tar.extractfile(member).read()) for member in tar.getmembers()]
    with tarfile.open(path, "w") as tar:
        for info, data in [*first, *own[:keep], *last]:
            tar.addfile(info, None if data is None else io.BytesIO(data))


def _file(name, data=b"evil"):
    """A regular file as a member for ``_rewrite_tar``."""
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _another_dataset(tiny, directory):
    """A copy of the tiny dataset with one word of its manifest changed and signed: valid, of another identity."""
    shutil.copytree(tiny, directory)
    manifest = json.loads((directory / "manifest.json").read_text()) | {"time_convention": "another one"}
    (directory / "manifest.json").write_text(json.dumps(manifest | {"identity": _identity(manifest)}))
    return verified_identity(directory)


def _nothing_left(target):
    return not target.exists() and not target.with_name(target.name + ".unpacking").exists()


def _rezip(source, path, change):
    """Write at ``path`` the members ``{name: bytes}`` of the zip ``source`` after ``change`` has edited them."""
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    change(members)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _no_tars_left(target):
    return not target.exists() and not target.with_name(target.name + ".unzipping").exists()


def test_index_lists_the_tars_and_exactly_the_files_of_the_manifest(tiny, packed):
    out, index = packed
    manifest = json.loads((tiny / "manifest.json").read_text())
    files, by_shape = _tree(tiny), dataset_files(manifest)
    assert index == json.loads((out / INDEX_FILE).read_text()) == read_index(out)
    assert index["format_version"] == 1 and index["identity"] == manifest["identity"] == shapes_dataset_identity(tiny)
    prefix = f"shapes_{manifest['identity'][:16]}"
    assert [(entry["name"], entry["split"], entry["shapes"]) for entry in index["tars"]] == [
        (f"{prefix}_train_1of2.tar", "train", ["000_sphere"]), (f"{prefix}_train_2of2.tar", "train", ["001_torus"]),
        (f"{prefix}_val.tar", "val", ["002_sphere"]),
        (f"{prefix}_test.tar", "test", ["003_torus", "004_sphere", "005_torus"])]
    assert sorted(path.name for path in out.iterdir()) == sorted([INDEX_FILE, *(e["name"] for e in index["tars"])])
    listed = {}
    for entry in index["tars"]:
        path = out / entry["name"]
        assert (entry["size"], entry["sha256"]) == (path.stat().st_size, _sha(path))
        # the manifest and the config lead every tar; then whole shapes, their files by name
        assert list(entry["members"]) == METADATA + [name for shape in entry["shapes"] for name in by_shape[shape]]
        assert all(size == len(files[name]) for name, size in entry["members"].items())
        assert set(entry["members"]) & set(listed) <= set(METADATA)            # a data file travels in one tar
        listed |= entry["members"]
    assert set(listed) == set(files)                    # every file of the dataset, each data file in one tar
    assert (index["files"], index["bytes"]) == (len(files), sum(map(len, files.values())))
    assert by_shape["000_sphere"] == [f"000_sphere/{name}" for name in (
        "diagnostics.npy", "inputs.npy", "mesh.npz", "norms.npy", "outputs.npy", "params.npy", "shape.json")]
    assert "003_torus/ood_outputs.npy" in index["tars"][TEST]["members"]
    assert "002_sphere/ood_outputs.npy" not in listed


def test_tars_are_plain_ustar_archives_that_extract_on_their_own(tiny, packed, tmp_path):
    out, index = packed
    for entry in index["tars"]:
        raw = (out / entry["name"]).read_bytes()
        assert raw[257:265] == b"ustar\x0000" and len(raw) % tarfile.RECORDSIZE == 0
        with tarfile.open(out / entry["name"], "r:") as tar:                # "r:" takes no compressed file
            members = tar.getmembers()
            assert [member.name for member in members] == list(entry["members"])
            for member in members:
                assert member.type == tarfile.REGTYPE and not member.pax_headers
                assert (member.mode, member.mtime, member.uid, member.gid, member.uname, member.gname) == (
                    0o644, 0, 0, 0, "", "")
                assert tar.extractfile(member).read() == (tiny / member.name).read_bytes()
    # one training part, extracted by hand: the manifest, the config and its whole shape
    part = index["tars"][TRAIN_2]
    with tarfile.open(out / part["name"]) as tar:
        tar.extractall(tmp_path / "alone", filter="data")
    assert sorted(path.name for path in (tmp_path / "alone").iterdir()) == ["001_torus", "config.json", "manifest.json"]
    assert _tree(tmp_path / "alone") == {name: data for name, data in _tree(tiny).items() if name in part["members"]}


def test_packing_twice_gives_identical_tars(tiny, packed, tmp_path):
    out, index = packed
    moved = tmp_path / "another_place" / "another_name"
    shutil.copytree(tiny, moved)
    for path in moved.rglob("*"):                       # other times and permissions than the first dataset's
        os.utime(path, (1_000_000_000, 1_000_000_000))
        path.chmod(0o700 if path.is_dir() else 0o600)
    # what the manifest does not name is not of the dataset, and does not travel
    (moved / "generation.log").write_text("a log\n")
    (moved / "000_sphere" / "notes.txt").write_text("a note\n")
    (moved / "006_sphere.partial").mkdir()
    (moved / "006_sphere.partial" / "mesh.npz").write_bytes(b"half a mesh")
    again = tmp_path / "again"
    assert pack(moved, again, train_parts=2) == index
    assert _tree(again) == _tree(out)                   # the tars and the index, byte for byte
    (again / index["tars"][VAL]["name"]).write_bytes(b"an older tar")
    assert pack(moved, again, train_parts=2) == index and _tree(again) == _tree(out)        # replaced in place


def test_a_name_that_does_not_fit_in_a_ustar_header_is_an_error(tmp_path):
    """No pax or GNU record is written for it: the tar would no longer be the plain one the docstring promises."""
    name = "a" * 101
    (tmp_path / name).write_bytes(b"data")
    hashes = {name: hashlib.sha256(b"data").hexdigest()}
    with pytest.raises(ValueError, match="name is too long"):
        shapes_archive._write_tar(tmp_path, tmp_path / "long.tar", [name], hashes, {})
    assert [path.name for path in tmp_path.iterdir()] == [name]             # no tar, and no long.tar.tmp either


@pytest.mark.parametrize("error", [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()])
def test_an_interrupted_pack_leaves_no_index_and_no_temporary_file(tiny, tars, tmp_path, monkeypatch, error):
    addfile = tarfile.TarFile.addfile

    def stop_in_val(tar, info, fileobj=None):
        if info.name == "002_sphere/outputs.npy":       # half of the val tar is written by now
            raise error
        return addfile(tar, info, fileobj)

    with monkeypatch.context() as patch:
        patch.setattr(tarfile.TarFile, "addfile", stop_in_val)
        with pytest.raises(type(error)):
            pack(tiny, tars, train_parts=1)             # over an older pack, of two training parts
    left = sorted(path.name for path in tars.iterdir())
    assert sum(name.endswith("_train_1of1.tar") for name in left) == 1      # written before the interruption
    assert INDEX_FILE not in left and all(name.endswith(".tar") for name in left)
    with pytest.raises(ValueError, match="Missing index of the tars"):
        unpack(tars, tmp_path / "data")


def test_a_file_that_changes_after_the_verification_stops_the_pack(tiny, packed, tmp_path, monkeypatch):
    """What is archived is read after the verification, minutes after for data/shapes: it has to be what was verified.
    """
    copy, out, verify, write = tmp_path / "copy", tmp_path / "tars", verified_identity, shapes_archive._write_tar
    shutil.copytree(tiny, copy)
    changes = {"001_torus/outputs.npy": lambda raw: raw[:500] + bytes([raw[500] ^ 1]) + raw[501:],
               "002_sphere/mesh.npz": lambda raw: raw[:-100],
               "005_torus/ood_inputs.npy": lambda raw: raw + bytes(100),
               "000_sphere/shape.json": lambda raw: raw + b"\n",          # the same JSON, but not the same bytes
               "config.json": lambda raw: raw + b"\n",
               "manifest.json": lambda raw: raw + b"\n"}
    for name, change in changes.items():
        whole = (copy / name).read_bytes()

        def verify_then_change(directory):
            identity = verify(directory)
            (copy / name).write_bytes(change(whole))
            return identity

        monkeypatch.setattr(shapes_archive, "verified_identity", verify_then_change)
        with pytest.raises(ValueError, match=f"Dataset file changed while it was (packed|verified): {name}"):
            pack(copy, out, train_parts=2)
        assert all(path.name.endswith(".tar") for path in out.glob("*"))    # no index, no temporary file
        (copy / name).write_bytes(whole)

    def change_then_write(directory, *arguments):
        """The JSON files change once the tars are being written: they are packed from memory."""
        for name in ("manifest.json", "config.json", "000_sphere/shape.json"):
            (copy / name).write_text("{}")
        return write(directory, *arguments)

    monkeypatch.setattr(shapes_archive, "verified_identity", verify)
    monkeypatch.setattr(shapes_archive, "_write_tar", change_then_write)
    assert pack(copy, out, train_parts=2) == packed[1] and _tree(out) == _tree(packed[0])


def test_the_training_split_goes_in_as_many_tars_as_asked(tiny, tmp_path, monkeypatch):
    index = pack(tiny, tmp_path / "tars", train_parts=1)
    prefix = f"shapes_{index['identity'][:16]}"
    assert [(entry["name"], entry["shapes"]) for entry in index["tars"]] == [
        (f"{prefix}_train_1of1.tar", ["000_sphere", "001_torus"]), (f"{prefix}_val.tar", ["002_sphere"]),
        (f"{prefix}_test.tar", ["003_torus", "004_sphere", "005_torus"])]
    assert unpack(tmp_path / "tars", tmp_path / "data")["extracted"] and _tree(tmp_path / "data") == _tree(tiny)
    assert [len(shapes) for _, _, shapes in shapes_archive._tar_groups(
        {"identity": "0" * 64, "shapes": [{"index": i, "id": f"{i:03d}", "split": "train"} for i in range(10)]}, 4)
            ] == [2, 3, 2, 3]                           # consecutive, whole shapes; no val or test: no tar for them

    def verify(directory):
        raise AssertionError("the whole dataset was read before train_parts was looked at")

    monkeypatch.setattr(shapes_archive, "verified_identity", verify)
    for parts in (0, 3, -1):
        with pytest.raises(ValueError, match="train_parts"):
            pack(tiny, tmp_path / "refused", train_parts=parts)
    assert not (tmp_path / "refused").exists()


def test_pack_verifies_the_dataset_first(tiny, tmp_path, monkeypatch):
    out, copy = tmp_path / "tars", tmp_path / "copy"
    shutil.copytree(tiny, copy)
    with monkeypatch.context() as patch:                # a cut JSON file is named, before the data is read
        patch.setattr(shapes_archive, "shapes_dataset_identity", None)
        for name in ("003_torus/shape.json", "manifest.json", "config.json"):
            whole = (copy / name).read_bytes()
            (copy / name).write_bytes(whole[:-3])
            with pytest.raises(ValueError, match=f"{name} is not JSON: "):
                pack(copy, out)
            (copy / name).write_bytes(whole)
    stored = (copy / "config.json").read_text()
    (copy / "config.json").write_text(stored.replace('"seed": 0', '"seed": 1'))
    with pytest.raises(ValueError, match="Dataset file differs from manifest: config.json"):
        pack(copy, out)
    (copy / "config.json").write_text(stored)
    _flip(copy / "001_torus" / "mesh.npz", 1000)
    with pytest.raises(ValueError, match="Dataset file differs from manifest: 001_torus/mesh.npz"):
        pack(copy, out)
    (copy / "005_torus" / "ood_params.npy").unlink()
    (copy / "001_torus" / "mesh.npz").write_bytes((tiny / "001_torus" / "mesh.npz").read_bytes())
    with pytest.raises(ValueError, match="Dataset file is missing: .*005_torus/ood_params.npy"):
        pack(copy, out)
    assert not out.exists()
    with pytest.raises(ValueError, match="inside the dataset"):
        pack(tiny, tiny / "tars")
    assert not (tiny / "tars").exists()


def test_a_manifest_that_names_files_outside_the_layout_is_refused(tiny):
    manifest = json.loads((tiny / "manifest.json").read_text())
    for name in ("../evil.npy", "/tmp/evil.npy", "000_sphere/../../evil.npy", "000_sphere/sub/evil.npy",
                 "777_sphere/evil.npy", "evil.npy", "000_sphere/.hidden", "000_sphere/shape.json"):
        with pytest.raises(ValueError, match="outside the dataset layout"):
            dataset_files(manifest | {"sha256": manifest["sha256"] | {name: "0" * 64}})


def test_unpack_restores_the_dataset_and_a_second_unpack_does_nothing(tiny, packed, tmp_path, monkeypatch):
    out, index = packed
    target, messages = tmp_path / "content" / "data" / "shapes", []            # its parent does not exist yet
    result = unpack(out, target, progress=messages.append)
    assert result["extracted"] is True and result["identity"] == index["identity"]
    assert (result["directory"], result["files"], result["bytes"]) == (str(target), index["files"], index["bytes"])
    assert _tree(target) == _tree(tiny) and verified_identity(target) == index["identity"]
    assert [path.name for path in target.parent.iterdir()] == ["shapes"]       # no temporary directory left
    assert len(messages) == len(index["tars"]) + 1 and "verified" in messages[-1]

    # The second time only the index is read: the tars are not needed, and no file is touched.
    only_index = tmp_path / "only_index"
    only_index.mkdir()
    shutil.copy(out / INDEX_FILE, only_index)
    monkeypatch.setattr(shapes_archive, "_extract_tar", None)
    before = {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in [target, *target.rglob("*")]}
    again = unpack(only_index, target, identity=index["identity"][:16], check_sha256=True, progress=messages.append)
    assert again["extracted"] is False and again["identity"] == index["identity"] and "nothing to do" in messages[-1]
    assert {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in [target, *target.rglob("*")]} == before


def test_a_truncated_or_missing_tar_is_named_before_anything_is_extracted(tars, tmp_path, no_extraction):
    index, target = read_index(tars), tmp_path / "data"
    cut, at_a_record, missing = (index["tars"][k] for k in (TRAIN_1, VAL, TEST))
    os.truncate(tars / cut["name"], cut["size"] - 1)
    os.truncate(tars / at_a_record["name"], at_a_record["size"] - tarfile.RECORDSIZE)
    (tars / missing["name"]).unlink()
    with pytest.raises(ValueError) as error:
        unpack(tars, target)
    message = str(error.value)
    assert f"{cut['name']} has {cut['size'] - 1} bytes, not {cut['size']}" in message
    assert f"{at_a_record['name']} has" in message and f"{missing['name']} is missing" in message
    assert index["tars"][TRAIN_2]["name"] not in message and _nothing_left(target)


def test_a_tar_that_ends_early_stops_the_extraction(tars, tmp_path):
    """The size was right when it was looked at (Drive can say one thing and serve another): the read itself fails."""
    entry, target = read_index(tars)["tars"][VAL], tmp_path / "data"
    path, stored = tars / entry["name"], (tars / INDEX_FILE).read_text()
    whole = path.read_bytes()

    def listed_size(index):
        index["tars"][VAL]["size"] = path.stat().st_size

    with tarfile.open(path) as tar:
        last = tar.getmembers()[-1]
    cuts = {"could not be extracted": entry["size"] // 2,                                  # inside a member
            f"differs from {INDEX_FILE}": last.offset_data + -(-last.size // 512) * 512}   # after the last one
    for reason, size in cuts.items():
        path.write_bytes(whole[:size])
        _edit_index(tars, listed_size)
        with pytest.raises(ValueError, match=f"{entry['name']} {reason}"):
            unpack(tars, target)
        assert _nothing_left(target)
    path.write_bytes(whole)
    _rewrite_tar(path, keep=3)                           # a whole tar, without most of its files
    _edit_index(tars, listed_size)
    with pytest.raises(ValueError, match=f"{entry['name']} differs from {INDEX_FILE}"):
        unpack(tars, target)
    _sign(tars)                                          # only a tar that is the index's can lack files
    with pytest.raises(ValueError, match=f"{entry['name']} lacks {len(entry['members']) - 3} of the files"):
        unpack(tars, target)
    assert _nothing_left(target)
    (tars / INDEX_FILE).write_text(stored)               # and with the true index, its size already gives it away
    with pytest.raises(ValueError, match=f"{entry['name']} has .* bytes, not {entry['size']}"):
        unpack(tars, target)


def test_one_changed_byte_fails_the_verification_and_leaves_no_target(tars, tmp_path, monkeypatch):
    entry, target = read_index(tars)["tars"][VAL], tmp_path / "data"
    with tarfile.open(tars / entry["name"]) as tar:
        member = tar.getmember("002_sphere/outputs.npy")
    _flip(tars / entry["name"], member.offset_data + member.size // 2)
    assert (tars / entry["name"]).stat().st_size == entry["size"]
    messages = []
    with pytest.raises(ValueError, match=f"{entry['name']} differs from {INDEX_FILE}"):
        unpack(tars, target, progress=messages.append)  # found while the tar is extracted
    assert _nothing_left(target)
    assert [message.endswith(f"reading {entry['name']} again") for message in messages] == [False, False, True]
    with monkeypatch.context() as patch:
        patch.setattr(shapes_archive, "_extract_tar", None)
        with pytest.raises(ValueError, match=f"{entry['name']} has another SHA-256"):
            unpack(tars, target, check_sha256=True)     # found before any tar is extracted
    assert _nothing_left(target)
    _sign(tars)                                         # the index agrees with the changed tar: the manifest does not
    with pytest.raises(ValueError, match="Dataset file differs from manifest: 002_sphere/outputs.npy"):
        unpack(tars, target)
    assert _nothing_left(target)


@pytest.mark.parametrize("kind", ["read_error", "data", "header"])
def test_a_tar_that_reads_wrong_once_is_read_again(tiny, tars, tmp_path, monkeypatch, kind):
    """Drive fails, or serves other bytes, for a while: the second read of the tar overwrites what the first left."""
    entry, target, messages, failures = read_index(tars)["tars"][TEST], tmp_path / "data", [], []
    path = tars / entry["name"]
    whole, read = path.read_bytes(), shapes_archive._Hashed.read
    with tarfile.open(path) as tar:
        member = tar.getmember("004_sphere/outputs.npy")

    def fail_once(stream, size):
        if stream.raw.name == str(path) and stream.size > member.offset_data and not failures:
            failures.append(stream.size)
            raise OSError(errno.EIO, "Input/output error")
        return read(stream, size)

    def progress(message):
        messages.append(message)
        if message.endswith("again"):
            path.write_bytes(whole)                     # Drive has caught up

    if kind == "read_error":
        monkeypatch.setattr(shapes_archive._Hashed, "read", fail_once)
    else:                                               # tarfile takes a damaged header for the end of the archive
        _flip(path, member.offset_data + member.size // 2 if kind == "data" else member.offset)
    assert unpack(tars, target, progress=progress)["extracted"] and _tree(target) == _tree(tiny)
    reason = "could not be extracted: .*Input/output error" if kind == "read_error" else f"differs from {INDEX_FILE}"
    again = [message for message in messages if message.endswith("again")]
    assert len(again) == 1 and re.search(f"{entry['name']} {reason}.*; reading {entry['name']} again", again[0])


def test_a_tar_that_cannot_be_read_is_named_with_the_others(tars, tmp_path, monkeypatch, no_extraction):
    index, target, digest = read_index(tars), tmp_path / "data", shapes_archive._tar_digest
    unreadable, missing = index["tars"][TRAIN_2]["name"], index["tars"][TEST]["name"]
    (tars / missing).unlink()

    def fail_on_one(path):
        if path.name == unreadable:
            raise OSError(errno.EIO, "Input/output error")
        return digest(path)

    monkeypatch.setattr(shapes_archive, "_tar_digest", fail_on_one)
    with pytest.raises(ValueError, match=f"{unreadable} could not be read: .*Input/output error; {missing} is missing"):
        unpack(tars, target, check_sha256=True)
    assert _nothing_left(target)


def test_the_target_appears_only_after_the_verification_and_by_one_rename(packed, tmp_path, monkeypatch):
    target, seen, renames = tmp_path / "data", [], []
    verify, replace = shapes_archive._verify_extracted, os.replace

    def spy(folder, index):
        seen.append((Path(folder).name, target.exists()))
        return verify(folder, index)

    def rename(source, destination):
        renames.append((Path(source), Path(destination)))
        return replace(source, destination)

    monkeypatch.setattr(shapes_archive, "_verify_extracted", spy)
    monkeypatch.setattr(os, "replace", rename)
    assert unpack(packed[0], target)["extracted"]
    assert seen == [("data.unpacking", False)] and renames == [(tmp_path / "data.unpacking", target)]


def test_the_whole_tar_is_hashed_whatever_the_block_size(tiny, packed, tmp_path, monkeypatch):
    """With small reads tarfile stops at the end-of-archive blocks, and the padding after them is still to be hashed."""
    monkeypatch.setattr(shapes_archive, "BLOCK", 512)
    assert unpack(packed[0], tmp_path / "data")["extracted"] and _tree(tmp_path / "data") == _tree(tiny)


def test_a_python_without_extraction_filters_is_refused_at_once(packed, tmp_path, monkeypatch, no_extraction):
    monkeypatch.delattr(tarfile, "data_filter")
    with pytest.raises(RuntimeError, match=r"3\.11\.4\+ or 3\.10\.12\+\); this is Python 3\.\d+\.\d+"):
        unpack(packed[0], tmp_path / "data")
    assert not list(tmp_path.iterdir())


def test_a_leftover_temporary_directory_and_an_empty_target_are_no_obstacle(tiny, packed, tmp_path):
    target, leftover = tmp_path / "data", tmp_path / "data.unpacking"
    target.mkdir()
    (leftover / "000_sphere").mkdir(parents=True)
    (leftover / "000_sphere" / "mesh.npz").write_bytes(b"half a mesh")
    (leftover / "999_sphere").mkdir()
    (leftover / "999_sphere" / "inputs.npy").write_bytes(b"of another run")
    (leftover / "manifest.json").write_text("{}")
    assert unpack(packed[0], target)["extracted"]
    assert _tree(target) == _tree(tiny) and not leftover.exists()
    leftover.write_text("a file of that name")
    shutil.rmtree(target)
    assert unpack(packed[0], target)["extracted"] and _tree(target) == _tree(tiny) and not leftover.exists()


def test_the_expected_identity_is_enforced(packed, tmp_path, monkeypatch):
    (out, index), target = packed, tmp_path / "data"
    identity = index["identity"]
    other = ("1" if identity[0] == "0" else "0") + identity[1:]
    with monkeypatch.context() as patch:
        patch.setattr(shapes_archive, "_extract_tar", None)
        for wrong in (other, other[:16], identity[:15] + ("1" if identity[15] == "0" else "0")):
            with pytest.raises(ValueError, match=f"not of the expected {wrong[:16]}"):
                unpack(out, target, identity=wrong)
        for malformed in (identity[:15], "", identity + "0", "not an identity!", 16):
            with pytest.raises(ValueError, match="64 hex characters or the first 16"):
                unpack(out, target, identity=malformed)
    assert _nothing_left(target)
    assert unpack(out, target, identity=identity[:16])["extracted"]
    assert unpack(out, target, identity=identity)["extracted"] is False
    with pytest.raises(ValueError, match="not of the expected"):
        unpack(out, target, identity=other)             # also when the dataset is already there


def test_a_target_that_is_not_the_dataset_is_refused_and_left_as_it_is(tiny, packed, tmp_path, no_extraction):
    out, index = packed
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "notes.txt").write_text("not a dataset")
    with pytest.raises(ValueError, match="left as it is"):
        unpack(out, mine)
    for manifest in ("{}", "[1, 2]", "not JSON"):
        (mine / "manifest.json").write_text(manifest)
        with pytest.raises(ValueError, match="left as it is"):
            unpack(out, mine)
    assert _tree(mine) == {"notes.txt": b"not a dataset", "manifest.json": b"not JSON"}

    damaged = tmp_path / "damaged"
    shutil.copytree(tiny, damaged)
    _flip(damaged / "004_sphere" / "inputs.npy", 500)
    before = _tree(damaged)
    with pytest.raises(ValueError, match="Dataset file differs from manifest: 004_sphere/inputs.npy"):
        unpack(out, damaged)
    assert _tree(damaged) == before

    another = tmp_path / "another"
    identity = _another_dataset(tiny, another)
    before = _tree(another)
    with pytest.raises(ValueError, match=f"it holds the dataset {identity[:16]}"):
        unpack(out, another)
    assert _tree(another) == before and identity != index["identity"]

    (tmp_path / "file").write_text("a file")
    with pytest.raises(ValueError, match="left as it is"):
        unpack(out, tmp_path / "file")
    assert (tmp_path / "file").read_text() == "a file"

    # A link to an empty directory, or to nowhere, could not be replaced by the rename: refused before any work.
    (tmp_path / "empty").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "empty", target_is_directory=True)
    (tmp_path / "nowhere").symlink_to(tmp_path / "none", target_is_directory=True)
    for name in ("link", "nowhere"):
        with pytest.raises(ValueError, match=f"{name} is a symbolic link and is not the dataset .* left as it is"):
            unpack(out, tmp_path / name)
        assert (tmp_path / name).is_symlink()
    assert not list((tmp_path / "empty").iterdir())
    (tmp_path / "to_the_dataset").symlink_to(tiny, target_is_directory=True)
    assert unpack(out, tmp_path / "to_the_dataset")["extracted"] is False   # nothing to do, link or not
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "another", "damaged", "empty", "file", "link", "mine", "nowhere", "to_the_dataset"]


def test_free_space_is_checked_before_extracting(packed, tmp_path, monkeypatch, no_extraction):
    (out, index), target = packed, tmp_path / "data"
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda folder: usage._replace(free=index["bytes"]))  # no margin
    with pytest.raises(ValueError, match=re.escape(f"Not enough disk space in {tmp_path}")):
        unpack(out, target)
    assert _nothing_left(target)


@pytest.mark.parametrize("kind", ["parent", "absolute", "through_a_shape", "top_level", "unlisted", "twice",
                                  "other_size"])
def test_files_the_index_does_not_list_are_refused(tars, tmp_path, kind):
    """One file that ``pack`` would not have written, in a tar whose size and SHA-256 the index confirms."""
    index, target, outside = read_index(tars), tmp_path / "data", tmp_path / "outside"
    outside.mkdir()
    entry = index["tars"][TRAIN_1]
    extra = {"parent": _file("../outside/evil.npy"),
             "absolute": _file(str(outside / "evil.npy")),
             "through_a_shape": _file("000_sphere/../../outside/evil.npy"),
             "top_level": _file("evil.npy"),
             "unlisted": _file("000_sphere/evil.npy"),
             "twice": _file("config.json", b"x" * entry["members"]["config.json"]),
             "other_size": _file("000_sphere/shape.json")}[kind]
    # the file of another size takes the place of the real one, the last member, so that only its size tells
    _rewrite_tar(tars / entry["name"], last=[extra], keep=-1 if kind == "other_size" else None)
    _sign(tars)
    messages = []
    with pytest.raises(ValueError, match=f"{entry['name']} holds .* which is not a file that {INDEX_FILE} lists"):
        unpack(tars, target, progress=messages.append)
    assert _nothing_left(target) and not list(outside.iterdir())
    assert not messages                                 # refused at once: the tar is not read a second time
    assert sorted(path.name for path in tmp_path.iterdir()) == ["outside", "tars"]


@pytest.mark.parametrize("kind", [*NOT_REGULAR, "symlink_for_a_shape"])
def test_members_that_are_not_regular_files_are_refused(tars, tmp_path, kind):
    """A link, a device, a pipe or a directory under a listed name and with its listed size: only its type tells.

    The last one is a link in the place of a shape's folder, through which the files after it would be written.
    """
    index, target, outside = read_index(tars), tmp_path / "data", tmp_path / "outside"
    outside.mkdir()
    entry = index["tars"][TRAIN_1]
    name = "000_sphere" if kind == "symlink_for_a_shape" else "000_sphere/params.npy"
    info = tarfile.TarInfo(name)
    info.type, info.size = NOT_REGULAR.get(kind, tarfile.SYMTYPE), entry["members"].get(name, 0)
    info.linkname = {"symlink": str(outside / "evil.npy"), "hard_link": "manifest.json",
                     "symlink_for_a_shape": str(outside)}.get(kind, "")
    _rewrite_tar(tars / entry["name"], first=[(info, None)])
    _sign(tars)
    with pytest.raises(ValueError, match=f"{entry['name']} holds '{name}', which is not a regular file"):
        unpack(tars, target)
    assert _nothing_left(target) and not list(outside.iterdir())


def test_modes_and_owners_of_the_members_do_not_reach_the_disk(tiny, tars, tmp_path):
    """tarfile's ``data`` filter: the listed files with their bytes, but setuid, writable by all and of someone else."""
    path, target = tars / read_index(tars)["tars"][TRAIN_1]["name"], tmp_path / "data"
    with tarfile.open(path) as tar:
        members = [(member, tar.extractfile(member).read()) for member in tar.getmembers()]
    for member, _ in members:
        member.mode, member.uid, member.gid = 0o6777, 12345, 12345
    _rewrite_tar(path, first=members, keep=0)
    _sign(tars)
    assert unpack(tars, target)["extracted"] and _tree(target) == _tree(tiny)
    for file in target.rglob("*"):
        assert stat.S_IMODE(file.stat().st_mode) & 0o7022 == 0 and file.stat().st_uid == os.getuid(), file


def test_a_listed_path_outside_the_layout_is_refused_all_the_same():
    """``read_index`` refuses an index with such a path; the check of the members does not count on it."""
    for name in ("../evil.npy", "/tmp/evil.npy", "000_sphere/../evil.npy", "evil.npy", "000_sphere/sub/evil.npy"):
        info = tarfile.TarInfo(name)
        info.size = 4
        with pytest.raises(ValueError, match="the tar is refused"):
            shapes_archive._check_member(Path("some.tar"), info, {name: 4})


def test_an_index_that_is_not_what_pack_writes_is_refused(tars, tmp_path, no_extraction):
    target, stored = tmp_path / "data", (tars / INDEX_FILE).read_text()
    name = json.loads(stored)["tars"][TRAIN_1]["name"]

    def member(path):
        return lambda index: index["tars"][TRAIN_1]["members"].update({path: 4})

    def tar_name(new):
        return lambda index: index["tars"][TRAIN_1].update(name=new)

    changes = {
        "outside the dataset layout": [member(path) for path in (
            "../evil.npy", "/tmp/evil.npy", "000_sphere/../../evil.npy", "evil.npy", "000_sphere/sub/evil.npy",
            "000_sphere\\evil.npy", "000_sphere/..", "")],
        "is not the plain name of a tar": [tar_name(new) for new in ("../" + name, "sub/" + name, "/tmp/" + name,
                                                                     name[:-4] + ".zip", 7)],
        "format_version": [lambda index: index.update(format_version=2)],
        "identity is not a SHA-256": [lambda index: index.update(identity=index["identity"][:16])],
        "no valid size and sha256": [lambda index: index["tars"][VAL].update(size=-1),
                                     lambda index: index["tars"][VAL].update(size=True),
                                     lambda index: index["tars"][VAL].update(sha256="0" * 63)],
        "lacks manifest.json or config.json": [lambda index: index["tars"][VAL]["members"].pop("manifest.json")],
        "is listed in two tars": [lambda index: index["tars"][VAL]["members"].update({"000_sphere/mesh.npz": 4}),
                                  lambda index: index["tars"][VAL]["members"].update({"config.json": 4})],
        "no tar, or one twice": [lambda index: index.update(tars=[]),
                                 lambda index: index["tars"].append(index["tars"][VAL])],
        "files and bytes": [lambda index: index.update(bytes=index["bytes"] + 1)],
        "KeyError": [lambda index: index["tars"][TEST].pop("members"), lambda index: index.pop("tars")],
        "TypeError": [lambda index: index.update(tars=7), lambda index: index["tars"][TEST].update(members=7)],
    }
    for reason, edits in changes.items():
        for edit in edits:
            (tars / INDEX_FILE).write_text(stored)
            _edit_index(tars, edit)
            with pytest.raises(ValueError, match=f"{INDEX_FILE} is not a valid index of shapes tars: .*{reason}"):
                unpack(tars, target)
    for text in ("[1, 2]", '"a string"'):
        (tars / INDEX_FILE).write_text(text)
        with pytest.raises(ValueError, match="is not a valid index"):
            unpack(tars, target)
    (tars / INDEX_FILE).write_text(stored[:100])
    with pytest.raises(ValueError, match=f"{INDEX_FILE} is not JSON"):
        unpack(tars, target)
    (tars / INDEX_FILE).unlink()
    with pytest.raises(ValueError, match=f"Missing index of the tars: .*{INDEX_FILE}"):
        unpack(tars, target)
    assert _nothing_left(target)


def test_tars_that_do_not_hold_the_manifests_files_are_refused(tars, tmp_path):
    """An index and tars that agree with each other, but not with the manifest they carry."""
    index, target = read_index(tars), tmp_path / "data"
    entry = index["tars"][TEST]
    _rewrite_tar(tars / entry["name"], last=[_file("005_torus/evil.npy")])

    def listed(index):
        index["tars"][TEST]["members"]["005_torus/evil.npy"] = 4
        index.update(files=index["files"] + 1, bytes=index["bytes"] + 4)

    _edit_index(tars, listed)
    _sign(tars)
    with pytest.raises(ValueError, match="differ in 1 files, e.g. 005_torus/evil.npy"):
        unpack(tars, target)
    assert _nothing_left(target)


def test_tars_of_another_dataset_than_the_index_says_are_refused(tiny, packed, tmp_path):
    """The identity of the index, the one the caller expects, has to be that of what was extracted."""
    tars, target, expected = tmp_path / "tars", tmp_path / "data", packed[1]["identity"]
    identity = _another_dataset(tiny, tmp_path / "another")
    assert pack(tmp_path / "another", tars)["identity"] == identity != expected
    _edit_index(tars, lambda index: index.update(identity=expected))
    with pytest.raises(ValueError, match=f"The tars hold the dataset {identity[:16]}, not {expected[:16]}"):
        unpack(tars, target, identity=expected)
    assert _nothing_left(target)


def test_command_line_packs_prints_the_tars_and_roundtrips(tiny, packed, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pack_shapes_dataset, "ROOT", tmp_path)                 # the default --out is under ROOT
    out = tmp_path / "data_packed" / "data"
    (out.parent / "data.roundtrip").mkdir(parents=True)                         # someone's: the copy goes elsewhere
    (out.parent / "data.roundtrip" / "notes.txt").write_text("mine")
    assert pack_shapes_dataset.main(["--data", str(tiny), "--train-parts", "2", "--roundtrip"]) == 0
    printed, index = capsys.readouterr().out, read_index(out)
    assert index == packed[1] and _tree(out) == _tree(packed[0])
    for entry in index["tars"]:
        assert f"{entry['name']}: {len(entry['shapes'])} {entry['split']} shapes, " in printed
    assert f"4 tars, {sum(e['size'] for e in index['tars']) / 1e9:.3f} GB" in printed and "note:" not in printed
    assert f"roundtrip: identity {index['identity'][:16]} verified from the tars" in printed
    assert printed.splitlines()[-1] == (
        f"upload the 5 files of this pack in {out} to a Drive folder of their own; in Colab: "
        f"shapes_archive.unpack(<that folder>, '/content/data/data', identity='{index['identity'][:16]}')")
    assert _tree(out.parent) == {f"data/{name}": data for name, data in _tree(out).items()} | {
        "data.roundtrip/notes.txt": b"mine"}                                    # the copy is removed, and only it

    # other tars in --out are pointed out; a roundtrip that fails gives status 1 and removes its copy
    (out / "shapes_old_val.tar").write_bytes(b"of an older pack")

    def fail(tars_dir, target_dir, **options):
        Path(target_dir).mkdir()
        raise ValueError("no roundtrip today")

    monkeypatch.setattr(pack_shapes_dataset, "unpack", fail)
    assert pack_shapes_dataset.main(["--data", str(tiny), "--out", str(out), "--train-parts", "2", "--roundtrip"]) == 1
    streams = capsys.readouterr()
    assert "shapes_old_val.tar, which are not of this pack" in streams.out
    assert "error: no roundtrip today" in streams.err and "in Colab" not in streams.out
    assert sorted(path.name for path in out.parent.iterdir()) == ["data", "data.roundtrip"]


def test_command_line_exits_with_status_1_on_a_dataset_that_fails(tiny, tmp_path):
    copy, out = tmp_path / "copy", tmp_path / "tars"
    shutil.copytree(tiny, copy)
    _flip(copy / "002_sphere" / "norms.npy", 200)
    command = [sys.executable, str(SCRIPT), "--data", str(copy), "--out", str(out), "--train-parts", "2", "--roundtrip"]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 1, run.stderr
    assert "error: Dataset file differs from manifest: 002_sphere/norms.npy" in run.stderr and not out.exists()
    run = subprocess.run([sys.executable, str(SCRIPT), "--data", str(tmp_path / "none"), "--out", str(out)],
                         capture_output=True, text=True)
    assert run.returncode == 1 and "error: Dataset file is missing" in run.stderr and not out.exists()


def test_the_zip_holds_the_index_and_its_tars_and_unzips_to_the_same_folder(tiny, packed, zipped, tmp_path,
                                                                           monkeypatch):
    (out, index), messages = packed, []
    names, size = [INDEX_FILE, *(entry["name"] for entry in index["tars"])], sum(e["size"] for e in index["tars"])
    result = zip_tars(out, tmp_path / "up" / "glno_tiny.zip", progress=messages.append)   # "up" does not exist yet
    with zipfile.ZipFile(result["path"]) as archive:
        assert archive.namelist() == names and archive.testzip() is None
        assert {name: archive.read(name) for name in names} == _tree(out)
    assert result == {"identity": index["identity"], "path": str(tmp_path / "up" / "glno_tiny.zip"),
                      "size": (tmp_path / "up" / "glno_tiny.zip").stat().st_size, "bytes": size}
    assert result["size"] < size and len(messages) == len(index["tars"])        # deflate takes something off
    assert [path.name for path in (tmp_path / "up").iterdir()] == ["glno_tiny.zip"]    # no temporary file left
    assert zip_name("shapes_small") == "glno_shapes_small.zip"
    assert read_zip_index(zipped) == read_zip_index(result["path"]) == index   # what read_index gives for the folder

    target = tmp_path / "content" / "packed" / "tiny"                          # its parent does not exist yet
    messages.clear()
    unzipped = unzip_tars(zipped, target, identity=index["identity"][:16], progress=messages.append)
    assert unzipped.pop("seconds") >= 0 and unzipped == {
        "identity": index["identity"], "directory": str(target), "bytes": size, "extracted": True}
    assert _tree(target) == _tree(out) and [path.name for path in target.parent.iterdir()] == ["tiny"]
    assert len(messages) == len(index["tars"]) and all("out of glno_tiny.zip" in message for message in messages)
    assert unpack(target, tmp_path / "data")["extracted"] and _tree(tmp_path / "data") == _tree(tiny)

    # The second time nothing is read out of the zip and no file is touched.
    monkeypatch.setattr(shapes_archive, "_unzip_tar", None)
    before = {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in [target, *target.rglob("*")]}
    again = unzip_tars(zipped, target, identity=index["identity"], progress=messages.append)
    assert again["extracted"] is False and again["bytes"] == size and "nothing to do" in messages[-1]
    assert {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in [target, *target.rglob("*")]} == before


def test_only_the_tars_of_the_index_go_into_the_zip(tars, tmp_path, monkeypatch):
    index, path = read_index(tars), tmp_path / "glno_tiny.zip"
    path.write_bytes(b"an older zip")
    changed, missing = index["tars"][VAL], index["tars"][TEST]["name"]
    _flip(tars / changed["name"], changed["size"] // 2)                        # the same size, other bytes
    with pytest.raises(ValueError, match=f"{changed['name']} is not the tar of {INDEX_FILE}.*pack the dataset again"):
        zip_tars(tars, path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["glno_tiny.zip", "tars"]     # no temporary file left
    assert path.read_bytes() == b"an older zip"

    # what can be known without reading the tars is refused before anything is written
    monkeypatch.setattr(zipfile, "ZipFile", None)
    (tars / missing).unlink()
    with pytest.raises(ValueError, match=f"{missing} is missing"):
        zip_tars(tars, path)
    with pytest.raises(ValueError, match="is inside the folder of the tars"):
        zip_tars(tars, tars / "glno_tiny.zip")
    (tars / INDEX_FILE).write_text("{}")
    with pytest.raises(ValueError, match="is not a valid index of shapes tars"):
        zip_tars(tars, path)
    (tars / INDEX_FILE).unlink()
    with pytest.raises(ValueError, match="Missing index of the tars"):
        zip_tars(tars, path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["glno_tiny.zip", "tars"]


@pytest.mark.parametrize("error", [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()])
def test_an_interrupted_zip_leaves_no_file_and_keeps_the_older_zip(packed, tmp_path, monkeypatch, error):
    path, read, reads = tmp_path / "glno_tiny.zip", shapes_archive._Hashed.read, []
    path.write_bytes(b"an older zip")

    def fail_in_the_second_tar(stream, size):
        reads.append(stream.raw.name)
        if len(set(reads)) == 2:
            raise error
        return read(stream, size)

    monkeypatch.setattr(shapes_archive._Hashed, "read", fail_in_the_second_tar)
    with pytest.raises(type(error)):
        zip_tars(packed[0], path)
    assert [p.name for p in tmp_path.iterdir()] == ["glno_tiny.zip"] and path.read_bytes() == b"an older zip"


def test_a_zip_that_is_not_complete_or_not_of_these_tars_is_refused_before_a_tar_is_copied(packed, zipped, tmp_path,
                                                                                         monkeypatch):
    (out, index), target = packed, tmp_path / "packed" / "tiny"
    name = index["tars"][VAL]["name"]
    monkeypatch.setattr(shapes_archive, "_unzip_tar", None)

    def refused(path, match, **options):
        with pytest.raises(ValueError, match=match):
            unzip_tars(path, target, **options)
        if not options:                                 # the identity is the caller's to compare, as with read_index
            with pytest.raises(ValueError, match=match):
                read_zip_index(path)
        assert _no_tars_left(target)

    refused(tmp_path / "none.zip", "Missing zip of the tars: .*none.zip")
    cut = tmp_path / "cut.zip"
    cut.write_bytes(zipped.read_bytes()[:-100])                                # an upload that has not finished
    refused(cut, "cut.zip is not a complete zip .truncated, or still uploading")
    (tmp_path / "notes.zip").write_text("not a zip")
    refused(tmp_path / "notes.zip", "notes.zip is not a complete zip")
    refused(tmp_path, "is not a complete zip")                                 # a folder
    refused(zipped, f"is of the dataset {index['identity'][:16]}, not of the expected {'0' * 16}", identity="0" * 16)
    refused(_rezip(zipped, tmp_path / "a.zip", lambda members: members.pop(INDEX_FILE)), f"a.zip holds no {INDEX_FILE}")
    refused(_rezip(zipped, tmp_path / "b.zip", lambda members: members.update({INDEX_FILE: b"{}"})),
            "is not a valid index of shapes tars")
    refused(_rezip(zipped, tmp_path / "c.zip", lambda members: members.pop(name)),
            f"c.zip does not hold exactly {INDEX_FILE} and its tars, each once and with its size: {name}$")
    refused(_rezip(zipped, tmp_path / "d.zip", lambda members: members.update({name: members[name][:-1]})),
            f"d.zip does not hold exactly .*: {name}$")
    refused(_rezip(zipped, tmp_path / "e.zip", lambda members: members.update({"../notes.tar": b"mine"})),
            "e.zip does not hold exactly .*: ../notes.tar$")
    shutil.copy(zipped, tmp_path / "f.zip")
    with pytest.warns(UserWarning, match="Duplicate name"), zipfile.ZipFile(tmp_path / "f.zip", "a") as archive:
        archive.writestr(name, (out / name).read_bytes())                      # the same tar a second time
    refused(tmp_path / "f.zip", f"f.zip does not hold exactly .*: {name}$")
    assert not target.parent.exists()


def test_a_tar_that_comes_out_of_the_zip_wrong_is_read_again_and_then_refused(packed, zipped, tmp_path, monkeypatch):
    (out, index), messages = packed, []
    entry, read, failures = index["tars"][VAL], shapes_archive._Hashed.read, []

    def fail_once(stream, size):
        if stream.raw.name == entry["name"] and not failures:
            failures.append(size)
            raise OSError(errno.EIO, "Input/output error")
        return read(stream, size)

    def again():
        return [message for message in messages if message.endswith(f"; reading {entry['name']} again")]

    # Drive fails for a moment: the second read of that tar gives it whole
    with monkeypatch.context() as patched:
        patched.setattr(shapes_archive._Hashed, "read", fail_once)
        assert unzip_tars(zipped, tmp_path / "once", progress=messages.append)["extracted"]
    assert _tree(tmp_path / "once") == _tree(out) and len(again()) == 1
    assert re.search(f"{entry['name']} could not be read from .*Input/output error", again()[0])

    # a zip that holds other bytes for a tar, of the same size: read twice, refused by its name, nothing left
    damaged = bytearray((out / entry["name"]).read_bytes())
    damaged[len(damaged) // 2] ^= 1
    bad = _rezip(zipped, tmp_path / "bad.zip", lambda members: members.update({entry["name"]: bytes(damaged)}))
    messages.clear()
    with pytest.raises(ValueError, match=f"{entry['name']} in .*bad.zip differs from {INDEX_FILE}.*damaged upload"):
        unzip_tars(bad, tmp_path / "bad", progress=messages.append)
    assert len(again()) == 1 and _no_tars_left(tmp_path / "bad")

    # a byte of the zip itself changed inside that tar: the zip's own check stops it
    flipped = tmp_path / "flipped.zip"
    shutil.copy(zipped, flipped)
    with zipfile.ZipFile(zipped) as archive:
        info = archive.getinfo(entry["name"])
    header = zipped.read_bytes()[info.header_offset:info.header_offset + 30]
    start = info.header_offset + 30 + int.from_bytes(header[26:28], "little") + int.from_bytes(header[28:30], "little")
    _flip(flipped, start + info.compress_size // 2)
    messages.clear()
    with pytest.raises(ValueError, match=f"{entry['name']} could not be read from .*flipped.zip"):
        unzip_tars(flipped, tmp_path / "flipped", progress=messages.append)
    assert len(again()) == 1 and _no_tars_left(tmp_path / "flipped")


def test_the_folder_of_the_tars_appears_only_when_it_is_complete(packed, zipped, tmp_path, monkeypatch):
    (out, index), target = packed, tmp_path / "packed" / "tiny"
    seen, renames, unzip, replace = [], [], shapes_archive._unzip_tar, os.replace

    def spy(archive, zip_path, entry, path):
        seen.append((path.parent.name, target.exists()))
        if len(seen) == 3 and not renames:
            raise KeyboardInterrupt
        return unzip(archive, zip_path, entry, path)

    def rename(source, destination):
        renames.append((Path(source), Path(destination)))
        return replace(source, destination)

    monkeypatch.setattr(shapes_archive, "_unzip_tar", spy)
    with pytest.raises(KeyboardInterrupt):
        unzip_tars(zipped, target)
    assert seen == [("tiny.unzipping", False)] * 3 and _no_tars_left(target)

    # an empty folder and what an interrupted run left are no obstacle; the folder arrives by one rename
    target.mkdir()
    (tmp_path / "packed" / "tiny.unzipping").mkdir()
    (tmp_path / "packed" / "tiny.unzipping" / "half.tar").write_bytes(b"half")
    monkeypatch.setattr(os, "replace", rename)
    assert unzip_tars(zipped, target)["extracted"] and _tree(target) == _tree(out)
    assert renames == [(tmp_path / "packed" / "tiny.unzipping", target)] and len(seen) == 3 + len(index["tars"])
    assert [path.name for path in target.parent.iterdir()] == ["tiny"]


def test_a_folder_that_does_not_hold_these_tars_is_refused_and_left_as_it_is(packed, zipped, tmp_path, monkeypatch):
    (out, index), target = packed, tmp_path / "packed" / "tiny"
    assert unzip_tars(zipped, target)["extracted"]
    monkeypatch.setattr(shapes_archive, "_unzip_tar", None)
    os.truncate(target / index["tars"][TEST]["name"], 10)                      # no longer the tar of the index
    file, link = tmp_path / "packed" / "file", tmp_path / "packed" / "link"
    file.write_text("mine")
    link.symlink_to(out)                                                       # even one that leads to these very tars
    before = _tree(tmp_path / "packed")
    for folder, reason in ((target, "exists"), (file, "exists"), (link, "is a symbolic link")):
        with pytest.raises(ValueError, match=f"{folder.name} {reason} and does not hold the tars of glno_tiny.zip; "
                                             "it is left as it is"):
            unzip_tars(zipped, folder)
    assert _tree(tmp_path / "packed") == before and link.is_symlink()
    assert sorted(path.name for path in (tmp_path / "packed").iterdir()) == ["file", "link", "tiny"]


def test_room_for_the_tars_and_for_the_dataset_is_asked_before_a_tar_is_copied(packed, zipped, tmp_path, monkeypatch):
    (out, index), target = packed, tmp_path / "packed" / "tiny"
    needed = sum(entry["size"] for entry in index["tars"]) + index["bytes"] + shapes_archive.SPACE_MARGIN
    usage, copied, unzip = shutil.disk_usage(tmp_path), [], shapes_archive._unzip_tar
    monkeypatch.setattr(shapes_archive, "_unzip_tar", lambda *arguments: copied.append(unzip(*arguments)))
    monkeypatch.setattr(shutil, "disk_usage", lambda folder: usage._replace(free=needed - 1))
    with pytest.raises(ValueError, match=re.escape(f"Not enough disk space in {target.parent}: the tars and the "
                                                    f"dataset that unpack extracts from them need {needed / 1e9:.2f} GB")):
        unzip_tars(zipped, target)
    assert not copied and _no_tars_left(target)
    monkeypatch.setattr(shutil, "disk_usage", lambda folder: usage._replace(free=needed))
    assert unzip_tars(zipped, target)["extracted"] and len(copied) == len(index["tars"])


def test_command_line_with_zip_also_writes_the_single_file(tiny, packed, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pack_shapes_dataset, "ROOT", tmp_path)                 # the default --out is under ROOT
    out, identity = tmp_path / "data_packed" / "data", packed[1]["identity"][:16]
    assert pack_shapes_dataset.main(["--data", str(tiny), "--train-parts", "2", "--zip"]) == 0
    lines, path = capsys.readouterr().out.splitlines(), out.parent / "glno_data.zip"
    assert sorted(p.name for p in out.parent.iterdir()) == ["data", "glno_data.zip"] and _tree(out) == _tree(packed[0])
    assert unzip_tars(path, tmp_path / "unzipped", identity=identity)["extracted"]
    assert _tree(tmp_path / "unzipped") == _tree(packed[0])
    assert lines[-1] == (f"or upload the single file {path} ({path.stat().st_size / 1e9:.3f} GB) instead; in Colab: "
                         f"shapes_archive.unzip_tars(<that file>, '/content/packed/data', identity='{identity}'), "
                         "then unpack from that folder")
    assert sum(line.startswith("upload the 5 files of this pack") for line in lines) == 1
    assert sum(line.endswith("into glno_data.zip") for line in lines) == len(packed[1]["tars"])

    # without --zip nothing but the folder is written; a zip that fails gives status 1
    assert pack_shapes_dataset.main(["--data", str(tiny), "--out", str(tmp_path / "plain"), "--train-parts", "2"]) == 0
    assert "glno_" not in capsys.readouterr().out and not (tmp_path / "glno_data.zip").exists()

    def fail(tars_dir, zip_path, **options):
        raise ValueError("no zip today")

    monkeypatch.setattr(pack_shapes_dataset, "zip_tars", fail)
    assert pack_shapes_dataset.main(["--data", str(tiny), "--train-parts", "2", "--zip"]) == 1
    assert "error: no zip today" in capsys.readouterr().err
