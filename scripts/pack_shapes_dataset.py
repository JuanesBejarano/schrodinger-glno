"""Step 7 of Plan_GLNO_etapa3.md: pack a stage-3 dataset into the tars that travel to Google Drive.

Verifies the dataset (its identity and the SHA-256 of every file), writes the deterministic tars of
``schrodinger_glno.shapes_archive`` and their index ``shapes_tars.json`` in --out, and prints each tar
with its size. --out is outside data/, so the tars are never taken for a part of a dataset.

With --roundtrip the tars are then unpacked as the Colab notebook will unpack them (sizes first,
tarfile's data filter, the SHA-256 of every tar and of every file) into a new directory
``<out>.roundtrip.<random>``, next to --out, which is removed afterwards: it needs the size of the
dataset once more on that disk. Only a run that is killed leaves it behind, to be deleted by hand.

Upload everything in --out, the tars and shapes_tars.json, to one Drive folder per dataset; in Colab,
``shapes_archive.unpack(<that folder>, "/content/data/<name>", identity=...)``. Exits with status 1
if anything fails.

With --zip the tars and the index are also written into one file next to --out, ``glno_<name>.zip``
(about 9 % smaller), to upload instead of the folder; in Colab ``shapes_archive.unzip_tars`` copies the
tars out of it onto the local disk, and ``unpack`` goes on from there.

    python scripts/pack_shapes_dataset.py --data data/shapes --train-parts 4 --roundtrip --zip
    python scripts/pack_shapes_dataset.py --data data/shapes_small --train-parts 1 --roundtrip --zip
"""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import time

from schrodinger_glno.shapes_archive import INDEX_FILE, pack, unpack, zip_name, zip_tars

ROOT = Path(__file__).resolve().parents[1]


def say(message: str) -> None:
    print(message, flush=True)


def roundtrip(out: Path, identity: str) -> None:
    """Unpack the tars in ``out`` next to it and remove the copy; ``unpack`` verifies it or raises.

    The copy goes into a directory made for it, so nothing that was already there is deleted, or taken for
    the result of unpacking.
    """
    holder = Path(tempfile.mkdtemp(prefix=out.name + ".roundtrip.", dir=out.parent))
    try:
        result = unpack(out, holder / "dataset", identity=identity, progress=say)
    finally:
        shutil.rmtree(holder, ignore_errors=True)
    print(f"roundtrip: identity {result['identity'][:16]} verified from the tars in {result['seconds']:.1f} s; "
          f"{holder.name} removed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "shapes")
    parser.add_argument("--out", type=Path, help="directory of the tars (default data_packed/<data>)")
    parser.add_argument("--train-parts", type=int, default=4,
                        help="tars of the training split (4 for data/shapes, ~1.75 GB each; 1 for data/shapes_small)")
    parser.add_argument("--roundtrip", action="store_true",
                        help="unpack the tars next to --out, verify every SHA-256 and remove the copy")
    parser.add_argument("--zip", action="store_true",
                        help="also write the tars and the index into one file next to --out, glno_<data>.zip")
    args = parser.parse_args(argv)
    name = args.data.resolve().name
    out = args.out or ROOT / "data_packed" / name
    start = time.perf_counter()
    try:
        index = pack(args.data, out, train_parts=args.train_parts, progress=say)
        tars = index["tars"]
        print(f"{len(tars)} tars, {sum(tar['size'] for tar in tars) / 1e9:.3f} GB, of the dataset "
              f"{index['identity'][:16]} ({index['files']} files, {index['bytes'] / 1e9:.3f} GB) in "
              f"{time.perf_counter() - start:.1f} s")
        print(f"wrote {out / INDEX_FILE}")
        packed = {INDEX_FILE, *(tar["name"] for tar in tars)}
        others = sorted(path.name for path in out.iterdir() if path.name not in packed)
        if others:
            print(f"note: {out} also holds {', '.join(others)}, which are not of this pack: do not upload them")
        if args.roundtrip:
            roundtrip(out, index["identity"])
        print(f"upload the {len(packed)} files of this pack in {out} to a Drive folder of their own; in Colab: "
              f"shapes_archive.unpack(<that folder>, '/content/data/{name}', "
              f"identity='{index['identity'][:16]}')")
        if args.zip:
            zipped = zip_tars(out, out.parent / zip_name(name), progress=say)
            print(f"or upload the single file {zipped['path']} ({zipped['size'] / 1e9:.3f} GB) instead; in Colab: "
                  f"shapes_archive.unzip_tars(<that file>, '/content/packed/{name}', "
                  f"identity='{index['identity'][:16]}'), then unpack from that folder")
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
