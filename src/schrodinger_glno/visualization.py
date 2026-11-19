"""Figures of |ψ|² for the samples of a surface dataset: sphere (stage 1) and torus (stage 2).

Each row is one sample and the first panel shows the surface at t = 0 with the
packet's direction. A free packet's centre follows, for short times, the geodesic
from its start along t0 at speed k0; the other panels are maps where that
classical motion can be read off.

- Sphere: the geodesic is a great circle. The maps use rotated coordinates where it
  is the equator and the packet moves to the right; the dotted line marks where a
  classical particle would be (``k0 t`` radians along the circle).
- Torus: the maps show the whole periodic (u, v) domain centred on the start
  (u0, v0), in lengths at the start (``torus_map_xy``): ``x = ρ0 (u - u0)``,
  ``y = r (v - v0)``, both wrapped to one period, with ``ρ0 = R + r cos v0``. Near the start this is an
  isometry, so the packet and its direction look as on the surface; away from v0 the
  u direction is stretched by ``ρ0 / ρ(v)`` (up to (R + r)/(R - r) = 3 on the default
  torus). The dashed line is the geodesic (``torus_geodesic``) and the circle the
  classical position at each panel's time.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from .reference import torus_angles, torus_points

CMAP = "magma"


def path_frame(x0: NDArray, t0: NDArray) -> NDArray:
    """Rows ``(a, b, c)``: ``a`` = start point, ``b`` = direction, ``c`` = normal of the great circle."""
    a = x0 / np.linalg.norm(x0)
    b = t0 - (t0 @ a) * a
    b /= np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)])


def rotated_lon_lat(verts: NDArray, frame: NDArray, shift: float) -> tuple[NDArray, NDArray]:
    """Longitude along the packet's great circle (shifted by ``shift``) and latitude from it."""
    local = verts @ frame.T
    lon = np.angle(np.exp(1j * (np.arctan2(local[:, 1], local[:, 0]) - shift)))
    lat = np.arcsin(np.clip(local[:, 2] / np.linalg.norm(local, axis=1), -1, 1))
    return lon, lat


def _sphere_panel(ax, verts, faces, density, frame, travel, norm) -> None:
    offset = min(travel / 2, 0.6)  # look slightly ahead of the start so the packet and its path both show
    view = np.cos(offset) * frame[0] + np.sin(offset) * frame[1]
    corners = verts[faces]
    centres = corners.mean(axis=1)
    front = centres @ view > 0
    colours = plt.get_cmap(CMAP)(norm(density[faces].mean(axis=1)[front]))
    ax.add_collection3d(Poly3DCollection(corners[front], facecolors=colours, linewidths=0))
    angles = np.linspace(0, travel, 200)
    path = np.outer(np.cos(angles), frame[0]) + np.outer(np.sin(angles), frame[1])
    visible = path @ view > 0
    ax.plot(*(1.01 * path[visible]).T, color="#5EC8F2", lw=1.2, ls="--")
    ax.quiver(*(1.02 * frame[0]), *(0.45 * frame[1]), color="#5EC8F2", lw=1.6, arrow_length_ratio=0.3)
    ax.view_init(np.degrees(np.arcsin(view[2])), np.degrees(np.arctan2(view[1], view[0])))
    ax.set(xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), zlim=(-0.8, 0.8))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def sphere_samples_figure(verts: NDArray, faces: NDArray, mass: NDArray, trajectories: NDArray,
                          times: NDArray, params: NDArray, normals_dirs: NDArray, output: Path,
                          time_indices: list[int], vmax: float | None = None,
                          row_labels: list[str] | None = None) -> Path:
    """Plot ``|ψ|²`` for each sample.

    ``trajectories`` is (n, V, n_t) complex; ``params`` rows are (center, phi, k0, sigma);
    ``normals_dirs`` holds each sample's unit direction t0 at its centre. Each row's
    colour scale ends at its own maximum at t = 0, unless ``vmax`` fixes one scale for
    all rows (use it to compare a prediction with its reference).
    """
    n = len(trajectories)
    cols = 1 + len(time_indices)
    fig = plt.figure(figsize=(2.3 * cols, 1.95 * n + 0.6), layout="constrained")
    grid = fig.add_gridspec(n, cols)
    for row, (traj, p, t0) in enumerate(zip(trajectories, params, normals_dirs)):
        center, k0, sigma = int(p[0]), p[2], p[3]
        density = np.abs(traj)**2
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax if vmax is not None else density[:, 0].max(), clip=True)
        frame = path_frame(verts[center], t0)
        travel = k0 * times[-1]
        ax = fig.add_subplot(grid[row, 0], projection="3d", computed_zorder=False)
        _sphere_panel(ax, verts, faces, density[:, 0], frame, travel, norm)
        label = f"{row_labels[row]}\n" if row_labels else ""
        ax.set_title(f"{label}k0 = {k0:.2f}, σ = {sigma:.2f}", fontsize=9)
        lon, lat = rotated_lon_lat(verts, frame, travel / 2)
        span = np.ptp(lon[faces], axis=1)
        triangulation = mtri.Triangulation(lon, lat, faces, mask=span > np.pi)
        half_width = min(np.pi, travel / 2 + 4 * sigma)
        for col, k in enumerate(time_indices, start=1):
            ax = fig.add_subplot(grid[row, col])
            ax.tripcolor(triangulation, density[:, k], cmap=CMAP, norm=norm, shading="gouraud")
            ax.axvline(k0 * times[k] - travel / 2, color="#5EC8F2", lw=1, ls=":")
            ax.set(xlim=(-half_width, half_width), ylim=(-1.3, 1.3), aspect="equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f"t = {times[k]:.2f}", fontsize=9)
            if row == n - 1:
                ax.set_xlabel("ángulo a lo largo del círculo", fontsize=7)
            if col == 1:
                ax.set_ylabel("latitud", fontsize=7)
    fig.suptitle("|ψ|² en la esfera: vista 3D en t = 0 y mapas a lo largo del círculo máximo "
                 "(línea punteada: posición clásica k0·t)", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------- torus

PATH_COLOUR = "#5EC8F2"


def _torus_frame(u0: float, v0: float) -> tuple[NDArray, NDArray, NDArray]:
    """Unit vectors ``e_u``, ``e_v`` and the outward normal of the torus at angles (u0, v0)."""
    e_u = np.array([-np.sin(u0), np.cos(u0), 0.0])
    e_v = np.array([-np.sin(v0) * np.cos(u0), -np.sin(v0) * np.sin(u0), np.cos(v0)])
    return e_u, e_v, np.cross(e_u, e_v)


def torus_geodesic(x0: NDArray, t0: NDArray, speed: float, times: NDArray, major_radius: float = 1.0,
                   minor_radius: float = 0.5) -> NDArray:
    """Geodesic from ``x0`` with initial velocity ``speed · t0``, at ``times`` (≥ 0).

    Returns rows ``(u, v, du/dt, dv/dt)``, with u and v unwrapped (continuous in t).
    For ``ds² = ρ(v)² du² + r² dv²``, ``ρ = R + r cos v``, the geodesic equations are
    ``u'' = 2 r sin v u' v' / ρ`` and ``v'' = -ρ sin v u'² / r``; they conserve the speed
    ``√(ρ² u'² + r² v'²)`` and Clairaut's constant ``ρ² u'``. ``t0`` is projected onto the
    tangent plane of the continuous torus and normalized, so a direction built from a mesh
    vertex normal can be passed as is.
    """
    R, r = major_radius, minor_radius
    times = np.asarray(times, float)
    if np.any(times < 0):
        raise ValueError("times must be >= 0")
    u0, v0 = (float(a) for a in torus_angles(np.asarray(x0, float), R))
    e_u, e_v, _ = _torus_frame(u0, v0)
    a, b = np.asarray(t0, float) @ e_u, np.asarray(t0, float) @ e_v
    size = np.hypot(a, b)
    start = np.array([u0, v0, speed * a / size / (R + r * np.cos(v0)), speed * b / size / r])

    def rhs(_, y):
        _, v, du, dv = y
        rho = R + r * np.cos(v)
        return [du, dv, 2 * r * np.sin(v) * du * dv / rho, -rho * np.sin(v) * du**2 / r]

    if times.size == 0 or times.max() == 0:
        return np.tile(start, (times.size, 1))
    solution = solve_ivp(rhs, (0, times.max()), start, method="DOP853", rtol=1e-12, atol=1e-12,
                         dense_output=True)
    return solution.sol(times).T


def _wrap(x: NDArray, period: float) -> NDArray:
    """``x`` wrapped to [-period/2, period/2)."""
    return (x + period / 2) % period - period / 2


def torus_map_xy(u: NDArray, v: NDArray, u0: float, v0: float, major_radius: float = 1.0,
                 minor_radius: float = 0.5) -> tuple[NDArray, NDArray]:
    """Map coordinates of the torus angles (u, v) on the map centred on (u0, v0).

    ``x = ρ0 (u - u0)`` and ``y = r (v - v0)`` with ``ρ0 = R + r cos v0``, wrapped to the
    periods ``2π ρ0`` and ``2π r``: x grows along e_u and y along e_v, so the map is the
    surface seen from outside.
    """
    rho0 = major_radius + minor_radius * np.cos(v0)
    return (_wrap(rho0 * (np.asarray(u) - u0), 2 * np.pi * rho0),
            _wrap(minor_radius * (np.asarray(v) - v0), 2 * np.pi * minor_radius))


def periodic_triangulation(x: NDArray, y: NDArray, faces: NDArray, period_x: float,
                           period_y: float) -> tuple[mtri.Triangulation, NDArray]:
    """Triangulation of a closed mesh drawn on a doubly periodic map, without gaps at its edges.

    ``x, y`` are vertex coordinates wrapped to one period. A face spanning more than half a
    period crosses a seam: it is unwrapped around its first corner and drawn shifted by -1, 0
    and +1 periods in x and y, so its pieces fill both edges of the map (the axes clip the
    rest). Returns the triangulation and, for each of its points, the vertex it comes from.
    """
    fx, fy = x[faces], y[faces]
    seam = (np.ptp(fx, axis=1) > period_x / 2) | (np.ptp(fy, axis=1) > period_y / 2)
    sx = (fx[seam, :1] + _wrap(fx[seam] - fx[seam, :1], period_x)).ravel()
    sy = (fy[seam, :1] + _wrap(fy[seam] - fy[seam, :1], period_y)).ravel()
    shifts = [(i * period_x, j * period_y) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    xs = np.concatenate([x, *(sx + dx for dx, _ in shifts)])
    ys = np.concatenate([y, *(sy + dy for _, dy in shifts)])
    source = np.concatenate([np.arange(len(x)), *[faces[seam].ravel()] * len(shifts)])
    copies = len(x) + np.arange(len(shifts) * sx.size).reshape(-1, 3)
    return mtri.Triangulation(xs, ys, np.concatenate([faces[~seam], copies])), source


def _periodic_line(x: NDArray, y: NDArray, period_x: float, period_y: float) -> tuple[NDArray, NDArray]:
    """A continuous path in wrapped map coordinates, broken (NaN) where it crosses a seam."""
    jumps = np.flatnonzero((np.abs(np.diff(x)) > period_x / 2) | (np.abs(np.diff(y)) > period_y / 2)) + 1
    return np.insert(x, jumps, np.nan), np.insert(y, jumps, np.nan)


def _torus_view(u0: float, v0: float) -> NDArray:
    """Camera direction that shows the point (u0, v0) of the torus.

    The azimuth of its outward normal and an elevation between 30° and 65° above the
    equatorial plane (below it for points clearly on the lower half): the torus never looks
    edge-on, and a point on the inner side is seen across the hole, over the opposite side
    of the tube (which, for R = 1, r = 0.5, hides the inner equator below 19.5°).
    """
    normal = _torus_frame(u0, v0)[2]
    elevation = np.arcsin(np.clip(normal[2], -1, 1))
    below = elevation < -0.1                         # from above unless the normal points clearly down
    elevation = np.clip(abs(elevation), np.radians(30), np.radians(65)) * (-1 if below else 1)
    azimuth = np.arctan2(normal[1], normal[0])
    return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def _torus_panel(ax, verts, faces, density, geodesic, t0, norm, R, r) -> None:
    """The torus at t = 0 seen from ``_torus_view``, with the direction arrow and the visible start of the geodesic."""
    u0, v0 = geodesic[0, :2]
    view = _torus_view(u0, v0)
    corners = verts[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    front = normals @ view > 0                       # back faces culled; the collection depth-sorts the rest
    light = normals[front] @ view                    # a dim headlight, so the hole reads even where ψ ≈ 0
    colours = plt.get_cmap(CMAP)(norm(density[faces].mean(axis=1)[front]))
    colours[:, :3] = colours[:, :3] * (0.7 + 0.3 * light[:, None]) + 0.12 * light[:, None]
    ax.add_collection3d(Poly3DCollection(corners[front], facecolors=np.clip(colours, 0, 1), linewidths=0))
    u, v = geodesic[:, 0], geodesic[:, 1]
    lift = 1 + 0.03 / r                              # draw slightly off the surface
    path = torus_points(u, v, R, lift * r)
    facing = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], axis=1) @ view > 0
    shown = len(facing) if facing.all() else int(np.argmin(facing))   # up to where it turns away
    ax.plot(*path[:shown].T, color=PATH_COLOUR, lw=1.1, ls="--")
    e_u, e_v, _ = _torus_frame(u0, v0)
    arrow = (t0 @ e_u) * e_u + (t0 @ e_v) * e_v
    ax.quiver(*path[0], *(0.5 * arrow / np.linalg.norm(arrow)), color=PATH_COLOUR, lw=1.6, arrow_length_ratio=0.3)
    ax.set_proj_type("ortho")
    ax.view_init(np.degrees(np.arcsin(view[2])), np.degrees(np.arctan2(view[1], view[0])))
    size = R + r
    ax.set(xlim=(-size, size), ylim=(-size, size), zlim=(-size, size))
    ax.set_box_aspect((1, 1, 1), zoom=1.6)
    ax.set_axis_off()


def torus_samples_figure(verts: NDArray, faces: NDArray, mass: NDArray, trajectories: NDArray,
                         times: NDArray, params: NDArray, directions: NDArray, output: Path,
                         time_indices: list[int], vmax: float | None = None,
                         row_labels: list[str] | None = None, major_radius: float = 1.0,
                         minor_radius: float = 0.5) -> Path:
    """Plot ``|ψ|²`` for each sample on the torus, with the same conventions as ``sphere_samples_figure``.

    ``trajectories`` is (n, V, n_t) complex; ``params`` rows are (center, phi, k0, sigma);
    ``directions`` holds each sample's unit direction t0 at its centre. Each row's colour
    scale ends at its own maximum at t = 0, unless ``vmax`` fixes one scale for all rows.
    The maps are described in the module docstring.
    """
    R, r = major_radius, minor_radius
    n = len(trajectories)
    cols = 1 + len(time_indices)
    # A map is 2πρ0 × 2πr: rows are as tall as their maps need (2.2 in wide), at least as the 3D view.
    rho0 = R + r * np.cos(torus_angles(verts[params[:, 0].astype(int)], R)[1])
    heights = np.maximum(1.6, 2.2 * r / rho0 + 0.5)
    fig = plt.figure(figsize=(2.6 * cols, heights.sum() + 0.8), layout="constrained")
    grid = fig.add_gridspec(n, cols, height_ratios=heights)
    u_all, v_all = torus_angles(verts, R)
    path_times = np.linspace(0, times[-1], 600)
    for row, (traj, p, t0) in enumerate(zip(trajectories, params, directions)):
        center, k0, sigma = int(p[0]), p[2], p[3]
        density = np.abs(traj)**2
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax if vmax is not None else density[:, 0].max(), clip=True)
        geodesic = torus_geodesic(verts[center], t0, k0, path_times, R, r)
        marks = torus_geodesic(verts[center], t0, k0, times[time_indices], R, r)
        u0, v0 = geodesic[0, :2]
        rho0 = R + r * np.cos(v0)
        ax = fig.add_subplot(grid[row, 0], projection="3d", computed_zorder=False)
        _torus_panel(ax, verts, faces, density[:, 0], geodesic, t0, norm, R, r)
        label = f"{row_labels[row]}\n" if row_labels else ""
        below = "\n(vista desde abajo)" if _torus_view(u0, v0)[2] < 0 else ""
        ax.set_title(f"{label}k0 = {k0:.2f}, σ = {sigma:.2f}, v0 = {np.degrees(v0):.0f}°{below}", fontsize=9)
        # Everything on the maps goes through torus_map_xy: one period wide in each direction.
        centre = (u0, v0, R, r)
        px, py = 2 * np.pi * rho0, 2 * np.pi * r
        triangulation, source = periodic_triangulation(*torus_map_xy(u_all, v_all, *centre), faces, px, py)
        line = _periodic_line(*torus_map_xy(geodesic[:, 0], geodesic[:, 1], *centre), px, py)
        mark_x, mark_y = torus_map_xy(marks[:, 0], marks[:, 1], *centre)
        equators = {"ext.": torus_map_xy(u0, 0.0, *centre)[1], "int.": torus_map_xy(u0, np.pi, *centre)[1]}
        for col, k in enumerate(time_indices, start=1):
            ax = fig.add_subplot(grid[row, col])
            ax.tripcolor(triangulation, density[source, k], cmap=CMAP, norm=norm, shading="gouraud")
            for name, height in equators.items():
                ax.axhline(height, color="white", lw=0.6, ls=(0, (1, 3)), alpha=0.5)
                ax.text(0.98 * px / 2, height, name, color="white", alpha=0.7, fontsize=6, ha="right",
                        va="top" if height > py / 4 else "bottom")      # inside the axes
            ax.plot(*line, color=PATH_COLOUR, lw=0.8, ls="--", alpha=0.8)
            ax.plot(mark_x[col - 1], mark_y[col - 1], "o", ms=7, mfc="none", mec=PATH_COLOUR, mew=1.3)
            ax.set(xlim=(-px / 2, px / 2), ylim=(-py / 2, py / 2), aspect="equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f"t = {times[k]:.2f}", fontsize=9)
            if row == n - 1:
                ax.set_xlabel("ρ0·(u − u0): alrededor del eje", fontsize=7)
            if col == 1:
                ax.set_ylabel("r·(v − v0): alrededor\ndel tubo", fontsize=7)
    fig.suptitle("|ψ|² en el toro: vista 3D en t = 0 y mapas (u, v) periódicos centrados en el punto de partida, "
                 "en longitudes medidas allí\n(línea discontinua: geodésica con dirección inicial t0; círculo: "
                 "posición clásica a velocidad k0; ext./int.: ecuadores exterior e interior; v0 = 0° en el exterior)",
                 fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


def surface_samples_figure(surface: str, verts: NDArray, faces: NDArray, mass: NDArray,
                           trajectories: NDArray, times: NDArray, params: NDArray, directions: NDArray,
                           output: Path, time_indices: list[int], vmax: float | None = None,
                           row_labels: list[str] | None = None, **geometry) -> Path:
    """``sphere_samples_figure`` or ``torus_samples_figure``; ``geometry`` holds the torus radii."""
    figures = {"sphere": sphere_samples_figure, "torus": torus_samples_figure}
    if surface not in figures:
        raise ValueError(f"surface must be one of {tuple(figures)}, not {surface!r}")
    return figures[surface](verts, faces, mass, trajectories, times, params, directions, output,
                            time_indices, vmax, row_labels, **geometry)
