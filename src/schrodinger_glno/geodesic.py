"""Geodesic distance on a closed triangle mesh and the geodesic cutoff of the stage-3 packet.

The packet of ``initial_conditions.gaussian_packet`` uses straight-line (chordal) distances.
On a non-convex surface it puts amplitude on parts that are close in R³ but far along the
surface (a limb next to the body, the other face of a thin plate). Stage 3 multiplies it by
a smooth cutoff in the geodesic distance from its centre: 1 up to 7σ, 0 from 9σ. The
distance comes from the heat method (K. Crane, C. Weischedel and M. Wardetzky, "Geodesics
in Heat", ACM TOG 2013) with the cotangent Laplacian and lumped mass of ``mesh``.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as sla
from numpy.typing import NDArray
from .initial_conditions import gaussian_packet, mass_norm
from .mesh import _check_mesh, cotan_laplacian, mean_edge_length, vertex_areas


class HeatGeodesics:
    """Heat-method geodesic distances from single vertices of one closed, connected mesh.

    ``(M + tL) u = δ_source`` with ``t = h²`` (h: mean edge length, M: lumped mass), the
    unit field ``X = -∇u/|∇u|`` on each face, then the least-squares ``φ`` with ``∇φ ≈ X``:
    ``L φ = Gᵀ A X``, where G is the per-face gradient and A the face areas. ``-Gᵀ A X`` is
    the integrated divergence of X at the vertices (the cotangent formula of the paper),
    with the sign of L, the weak ``-Δ``, and ``L = Gᵀ A G``. L is singular on a closed mesh;
    vertex 0 is pinned to 0 (``Gᵀ A X`` sums to zero, so the system stays consistent) and φ
    is shifted to vanish at the source. Both systems are factorized once per mesh, so a
    distance costs two pairs of triangular solves and a few sparse products.

    The mesh must have no negative cotangent weights (a Delaunay mesh, as stage 3 requires):
    then ``M + tL`` is an M-matrix and u > 0. Otherwise u can turn negative and the distance
    is wrong by tens of percent, so a negative u raises. u decays like ``e^{-d/h}`` (≈ 1e-52
    at the far side of the 25 346-vertex torus) and must stay a normal double, which limits d
    to about 700 h (stage-3 meshes reach about 100 h); an underflow raises too.

    Measured beyond 3h from the source: median relative error 0.9 % / 0.7 % on
    ``icosphere(4)`` / ``(5)`` against great circles; along the outer equator of the torus
    R = 1, r = 0.5, mean 2.1 % / 1.7 % / 1.4 % at edge 0.06 / 0.04 / 0.03. On those meshes
    ``d(a, b)`` and ``d(b, a)`` differ by less than one edge. Cost: factorization 0.5 s
    (10 242 vertices) to 1.5 s (25 346), then 0.03-0.04 s per distance.
    """

    def __init__(self, verts: NDArray, faces: NDArray) -> None:
        verts, faces = np.asarray(verts, float), np.asarray(faces)
        _check_mesh(verts, faces)
        n = len(verts)
        edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        adjacency = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        if csgraph.connected_components(adjacency, directed=False)[0] != 1:
            raise ValueError("the mesh must be a single connected piece with every vertex in a face")
        corners = verts[faces]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        double_area = np.linalg.norm(cross, axis=1)
        if np.any(double_area <= 0):
            raise ValueError("the mesh has degenerate faces")
        normal = cross / double_area[:, None]
        # Hat function of corner k: gradient N × e_k / (2A), e_k its opposite edge (k+1 → k+2).
        grads = [np.cross(normal, corners[:, (k + 2) % 3] - corners[:, (k + 1) % 3]) / double_area[:, None]
                 for k in range(3)]
        rows = np.tile((3 * np.arange(len(faces)))[:, None] + np.arange(3), 3).ravel()
        cols = np.repeat(faces, 3, axis=1).ravel()
        vals = np.stack(grads, axis=1).ravel()
        self.gradient = sp.csr_matrix((vals, (rows, cols)), shape=(3 * len(faces), n))
        self.face_areas = 0.5 * double_area
        self.t = mean_edge_length(verts, faces)**2
        # No ε in the cotangents: with it, the distance would depend on the mesh's units.
        L = cotan_laplacian(verts, faces, denom_eps=0.0)
        self._heat = sla.splu(sp.csc_matrix(sp.diags(vertex_areas(verts, faces)) + self.t * L))
        self._poisson = sla.splu(sp.csc_matrix(L[1:, 1:]))
        self._n = n

    def distance(self, source: int) -> NDArray:
        """Geodesic distance from vertex ``source`` to every vertex, exactly 0 at the source."""
        if not 0 <= source < self._n:
            raise ValueError("source must be a vertex index")
        delta = np.zeros(self._n)
        delta[source] = 1.0
        u = self._heat.solve(delta)
        if u.min() < 0:
            raise ValueError("the heat solution is negative somewhere: the mesh has negative "
                             "cotangent weights (it is not Delaunay)")
        if not (u.min() >= np.finfo(float).tiny and np.all(np.isfinite(u))):
            raise ValueError("the heat solution underflowed; the surface is too long for this edge length")
        grad = (self.gradient @ u).reshape(-1, 3)
        scale = np.abs(grad).max(axis=1)
        if not np.all(scale > 0):
            raise ValueError("the heat gradient vanishes on a face")
        grad /= scale[:, None]                        # so that squaring in the norm cannot underflow
        field = -grad / np.linalg.norm(grad, axis=1)[:, None]
        rhs = self.gradient.T @ (self.face_areas[:, None] * field).ravel()
        phi = np.concatenate([[0.0], self._poisson.solve(rhs[1:])])
        return phi - phi[source]


def geodesic_cutoff(distance: NDArray, sigma: float, inner: float = 7.0, outer: float = 9.0) -> NDArray:
    """Weights 1 for ``d ≤ inner·σ``, 0 for ``d ≥ outer·σ`` and a C² smoothstep in between.

    With ``x = (d - inner·σ) / ((outer - inner)·σ)`` the weight is ``1 - (6x⁵ - 15x⁴ + 10x³)``:
    value, slope and curvature match the constants at both ends.
    """
    if not (sigma > 0 and np.isfinite(sigma)) or not 0 <= inner < outer:
        raise ValueError("need sigma > 0 and 0 <= inner < outer")
    x = np.clip((np.asarray(distance, float) - inner * sigma) / ((outer - inner) * sigma), 0.0, 1.0)
    return 1 - x**3 * (10 + x * (6 * x - 15))


def cut_gaussian_packet(verts: NDArray, mass: NDArray | sp.spmatrix, normals: NDArray, center: int,
                        k0: float, sigma: float, direction: NDArray, distance: NDArray,
                        inner: float = 7.0, outer: float = 9.0) -> tuple[NDArray, float]:
    """``gaussian_packet`` times ``geodesic_cutoff(distance, sigma, inner, outer)``, normalized with ``mass``.

    ``distance`` holds the geodesic distances from ``center`` (``HeatGeodesics.distance``).
    Returns the packet, with ``ψᴴ M ψ = 1``, and the fraction of the chordal packet's
    ``‖ψ‖²_M`` that the cutoff removed, ``1 - ‖wψ‖²_M``: the part that had leaked to
    geodesically distant parts of the surface (with a non-diagonal mass it can be a
    round-off below 0 when nothing is cut).
    """
    psi = gaussian_packet(verts, mass, normals, center, k0, sigma, direction)
    distance = np.asarray(distance, float)
    if distance.shape != (len(verts),):
        raise ValueError("distance must hold one value per vertex")
    if abs(distance[center]) > 1e-3 * sigma:
        raise ValueError("distance must be measured from center")
    cut = geodesic_cutoff(distance, sigma, inner, outer) * psi
    kept = mass_norm(cut, mass)
    return cut / kept, 1.0 - kept**2
