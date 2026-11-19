"""Plot |ψ|² of one packet on each of a few shapes of a stage-3 dataset, drawn on the surface at a few times.

Rows are shapes (by default two deformed spheres, a deformed torus and a round torus), each with
one packet of its split (or of --set); columns are saved times. Colours follow ``visualization``:
|ψ|² on a square-root scale from 0 to the row's maximum at t = 0, so a row shows how the packet
spreads, and a dim headlight shades the surface where ψ ≈ 0. The camera follows the packet: it
looks along the normal at the vertex where |ψ|² is largest, from 30°-65° above or below the
equator on a torus so that the tube does not hide the inner side (as ``visualization._torus_view``).
The arrow at t = 0 is the packet's direction t0.

BLAS runs on one thread, set before NumPy loads, as in generate_shapes_dataset.py.
"""
import os

if __name__ == "__main__":          # must happen before NumPy loads; importing this module changes nothing
    for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_variable] = "1"

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from numpy.typing import NDArray
from schrodinger_glno.mesh import vertex_normals
from schrodinger_glno.sampling import direction_from_angle
from schrodinger_glno.shapes_dataset import CONFIG_FILE, SHAPE_FILE, ShapesConfig
from schrodinger_glno.visualization import CMAP, PATH_COLOUR

KINDS = {("sphere", False): "esfera deformada", ("torus", False): "toro deformado",
         ("torus", True): "toro de revolución"}


def default_shapes(directory: Path) -> list[str]:
    """The first two spheres, the first deformed torus and the first round torus among the complete shapes."""
    config = ShapesConfig(**json.loads((directory / CONFIG_FILE).read_text())["config"])
    complete = [i for i in range(config.n_shapes) if (directory / config.shape_id(i)).is_dir()]
    spheres = [i for i in complete if config.family(i) == "sphere"][:2]
    tori = [next((i for i in complete if config.family(i) == "torus" and config.is_round(i) == flag), None)
            for flag in (False, True)]
    return [config.shape_id(i) for i in spheres + tori if i is not None]


def view_direction(normal: NDArray, point: NDArray, torus: bool) -> NDArray:
    """Camera direction that shows a surface point with this outward ``normal``.

    On a torus the elevation is clipped to 30°-65° (below the equator only where the normal points
    clearly down) and, where the normal is nearly vertical, the azimuth is the point's own.
    """
    normal = normal / np.linalg.norm(normal)
    if not torus:
        return normal
    elevation = np.arcsin(np.clip(normal[2], -1, 1))
    below = elevation < -0.1
    elevation = np.clip(abs(elevation), np.radians(30), np.radians(65)) * (-1 if below else 1)
    x, y = normal[:2] if np.hypot(*normal[:2]) > 0.2 else point[:2]
    azimuth = np.arctan2(y, x)
    return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def surface_panel(ax, verts: NDArray, faces: NDArray, density: NDArray, norm: PowerNorm, view: NDArray) -> None:
    """``density`` on the faces that look toward ``view`` (``plot_trisurf`` with face colours), lit by a headlight."""
    corners = verts[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    front = normals @ view > 0                       # back faces culled; the collection depth-sorts the rest
    light = normals[front] @ view
    colours = plt.get_cmap(CMAP)(norm(density[faces[front]].mean(axis=1)))
    colours[:, :3] = colours[:, :3] * (0.7 + 0.3 * light[:, None]) + 0.12 * light[:, None]
    surface = ax.plot_trisurf(*verts.T, triangles=faces[front], shade=False, linewidth=0, antialiased=False)
    surface.set_facecolor(np.clip(colours, 0, 1))
    ax.set_proj_type("ortho")
    ax.view_init(np.degrees(np.arcsin(view[2])), np.degrees(np.arctan2(view[1], view[0])))
    size = np.abs(verts).max()
    ax.set(xlim=(-size, size), ylim=(-size, size), zlim=(-size, size))
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_axis_off()


def load_row(directory: Path, shape_id: str, packet_set: str | None, packet: int, times: list[float]) -> dict:
    """Mesh, |ψ|² at the saved times nearest ``times``, parameters and direction of one packet of a shape."""
    folder = directory / shape_id
    record = json.loads((folder / SHAPE_FILE).read_text())
    name = packet_set or record["split"]
    if name not in record["packets"] or not 0 <= packet < record["packets"][name]["packets"]:
        raise SystemExit(f"{shape_id} has no packet {packet} in the set {name!r}")
    prefix = "ood_" if name == "ood" else ""
    with np.load(folder / "mesh.npz") as mesh:
        verts, faces, saved = mesh["verts"], mesh["faces"], mesh["t_out"]
    indices = [int(np.argmin(np.abs(saved - t))) for t in times]
    states = np.load(folder / f"{prefix}outputs.npy", mmap_mode="r")[packet][:, indices].astype(float)
    center, phi, k0, sigma = np.load(folder / f"{prefix}params.npy")[packet]
    normals = vertex_normals(verts, faces)
    return {"id": shape_id, "kind": KINDS[record["family"], record["round"]], "set": name, "packet": packet,
            "verts": verts, "faces": faces, "normals": normals, "density": np.sum(states**2, axis=-1),
            "times": saved[indices], "center": int(center), "k0": k0, "sigma": sigma,
            "torus": record["family"] == "torus", "direction": direction_from_angle(normals[int(center)], phi)}


def shapes_figure(rows: list[dict], output: Path) -> Path:
    """One row per packet of ``rows`` (``load_row``), one 3D panel per saved time."""
    n_cols = len(rows[0]["times"])
    # at least three panels wide, the width of the two-line title
    fig = plt.figure(figsize=(2.6 * max(n_cols, 3), 2.8 * len(rows) + 0.6), layout="constrained")
    for i, row in enumerate(rows):
        verts, density = row["verts"], row["density"]
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=density[:, 0].max(), clip=True)
        for j, t in enumerate(row["times"]):
            ax = fig.add_subplot(len(rows), n_cols, i * n_cols + j + 1, projection="3d", computed_zorder=False)
            peak = int(np.argmax(density[:, j]))
            surface_panel(ax, verts, row["faces"], density[:, j], norm,
                          view_direction(row["normals"][peak], verts[peak], row["torus"]))
            title = f"t = {t:.3g}"
            if j == 0:
                start = verts[row["center"]] + 0.03 * row["normals"][row["center"]]
                ax.quiver(*start, *(0.45 * row["direction"]), color=PATH_COLOUR, lw=1.6, arrow_length_ratio=0.3)
                title = (f"{row['id']}: {row['kind']}\n{row['set']} {row['packet']}: k0 = {row['k0']:.2f}, "
                         f"σ = {row['sigma']:.3f}\n" + title)
            ax.set_title(title, fontsize=9)
    fig.suptitle("|ψ|² sobre la superficie: escala por fila de 0 al máximo en t = 0, en raíz cuadrada;\n"
                 "la cámara sigue al paquete; flecha: dirección inicial t0", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/shapes"))
    p.add_argument("--shapes", nargs="+", help="Shape ids, one row each (default: two spheres, a deformed and a "
                                                "round torus)")
    p.add_argument("--set", dest="packet_set", help="Packet set: train, val, test or ood (default: the shape's split)")
    p.add_argument("--packet", type=int, default=0, help="Packet of the set to plot (default 0)")
    p.add_argument("--times", type=float, nargs="+", default=[0.0, 0.25, 0.5],
                   help="Times to show; each panel shows the nearest saved time")
    p.add_argument("--output", type=Path, default=Path("artifacts/shapes_samples.png"))
    args = p.parse_args(argv)
    shapes = args.shapes or default_shapes(args.data_dir)
    rows = [load_row(args.data_dir, shape_id, args.packet_set, args.packet, args.times) for shape_id in shapes]
    path = shapes_figure(rows, args.output)
    print(path)
    return path


if __name__ == "__main__":
    main()
