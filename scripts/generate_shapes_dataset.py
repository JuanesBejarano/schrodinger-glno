"""Generate the stage-3 dataset: Gaussian packets on deformed spheres and tori (V = 0), one directory per shape.

Shapes are built in parallel by --workers processes, each with single-threaded BLAS, and each
shape is saved as soon as it is complete. Run the same command again to resume an interrupted
run (or one stopped by a failed shape or a dead worker): complete shapes are skipped. --family
builds one family only; run the families one after the other, since a second run on a
directory another run is using is refused. The manifest is written once every shape exists,
by the run that completes the last one. The settings are stored in <data-dir>/config.json and
a run with different settings is refused.

BLAS also runs on one thread in this process, set before NumPy loads, as in
generate_surface_dataset.py.
"""
import os

if __name__ == "__main__":          # must happen before NumPy loads; importing this module changes nothing
    for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_variable] = "1"

import argparse
import json
from pathlib import Path
import time
from schrodinger_glno.shapes_dataset import ShapesConfig, dataset_summary, generate_shapes_dataset

# Command-line flag -> ShapesConfig field; given flags override --config.
FLAGS = {"seed": "seed", "train": "train_shapes", "val": "val_shapes", "test": "test_shapes",
         "packets": "packets_per_shape", "ood_packets": "ood_packets_per_shape"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/shapes"))
    p.add_argument("--workers", type=int, default=3, help="Shapes built at once (default 3)")
    p.add_argument("--family", choices=("sphere", "torus"), help="Build only this family (default: both)")
    p.add_argument("--seed", type=int)
    p.add_argument("--train", type=int, help="Training shapes (default 160)")
    p.add_argument("--val", type=int, help="Validation shapes (default 20)")
    p.add_argument("--test", type=int, help="Test shapes (default 20)")
    p.add_argument("--packets", type=int, help="Packets per shape (default 5)")
    p.add_argument("--ood-packets", type=int, help="Out-of-distribution packets per test shape (default 5)")
    p.add_argument("--config", type=Path, help="JSON file with ShapesConfig fields that override the defaults")
    args = p.parse_args()
    overrides = json.loads(args.config.read_text()) if args.config else {}
    overrides |= {field: getattr(args, flag) for flag, field in FLAGS.items() if getattr(args, flag) is not None}
    config = ShapesConfig(**overrides)
    start = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[{(time.perf_counter() - start) / 60:6.1f} min] {message}", flush=True)

    print(f"{config.n_shapes} shapes ({config.train_shapes} train, {config.val_shapes} val, "
          f"{config.test_shapes} test), {config.packets_per_shape} packets each and "
          f"{config.ood_packets_per_shape} OOD per test shape; "
          f"data in {args.data_dir}", flush=True)
    try:
        manifest = generate_shapes_dataset(config, args.data_dir, workers=args.workers,
                                           families=[args.family] if args.family else None, progress=progress)
    except (FileExistsError, BlockingIOError) as error:     # another config, or another run on the directory
        raise SystemExit(str(error))
    summary = dataset_summary(args.data_dir) | {"complete": manifest is not None,
                                                "minutes": round((time.perf_counter() - start) / 60, 2)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
