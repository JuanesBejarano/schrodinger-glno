"""Exact solutions of the free Schrödinger equation on the continuous unit sphere and torus.

Sphere: with V = 0 and ħ = m = 1, ``ψ(x, t) = Σ_ℓ e^{-iℓ(ℓ+1)t/2} (Π_ℓ ψ0)(x)``, where the
projection onto degree ℓ follows from the addition theorem:
``(Π_ℓ f)(x) = (2ℓ+1)/(4π) ∫ P_ℓ(x·y) f(y) dμ(y)``. The integral uses a
Gauss–Legendre (in cos θ) × uniform (in φ) grid, exact for the band-limited
integrands involved. This is the reference that measures how far a mesh solution
is from the continuous sphere, not just from the mesh's own equation. The torus
reference, ``torus_exact_solution``, is described in the comment before ``torus_points``.
"""
import numpy as np
import scipy.linalg as sla
from numpy.typing import NDArray


def sphere_quadrature(n_theta: int = 100, n_phi: int = 200) -> tuple[NDArray, NDArray]:
    """Points on the unit sphere and weights summing to 4π."""
    z, wz = np.polynomial.legendre.leggauss(n_theta)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    Z, PHI = np.meshgrid(z, phi, indexing="ij")
    r = np.sqrt(1 - Z**2)
    points = np.stack([r * np.cos(PHI), r * np.sin(PHI), Z], axis=-1).reshape(-1, 3)
    return points, np.repeat(wz, n_phi) * (2 * np.pi / n_phi)


def sphere_packet(points: NDArray, x0: NDArray, t0: NDArray, k0: float, sigma: float) -> NDArray:
    """The continuous Gaussian packet of ``initial_conditions.gaussian_packet``, unnormalized."""
    offset = points - x0
    return np.exp(-np.sum(offset**2, 1) / (4 * sigma**2)) * np.exp(1j * k0 * (offset @ t0))


def sphere_exact_solution(targets: NDArray, x0: NDArray, t0: NDArray, k0: float, sigma: float,
                          times: NDArray, l_max: int = 60, chunk: int = 512) -> NDArray:
    """ψ(targets, times) on the unit sphere, with ψ0 normalized so that ∫|ψ0|² dμ = 1.

    ``targets`` must lie on the unit sphere. ``l_max = 60`` reproduces packets with
    k0 + 3/(2σ) up to about 25 to round-off; raise it for sharper packets.
    """
    # Exact for integrands up to degree ~2·(l_max/2 + 30) - 1: P_ℓ (ℓ ≤ l_max) times the packet's band.
    points, weights = sphere_quadrature(l_max // 2 + 30, 2 * (l_max // 2 + 30))
    psi0 = sphere_packet(points, x0, t0, k0, sigma)
    source = weights * psi0 / np.sqrt(np.sum(weights * np.abs(psi0)**2))
    ell = np.arange(l_max + 1)
    phases = np.exp(-1j * np.outer(ell * (ell + 1) / 2, np.asarray(times, float)))   # (ℓ, n_t)
    out = np.zeros((len(targets), len(times)), complex)
    for start in range(0, len(targets), chunk):
        c = np.clip(targets[start:start + chunk] @ points.T, -1, 1)
        p_prev, p = np.ones_like(c), c.copy()
        proj = np.zeros((len(c), l_max + 1), complex)
        proj[:, 0] = (p_prev @ source) / (4 * np.pi)
        proj[:, 1] = 3 * (p @ source) / (4 * np.pi)
        for l in range(1, l_max):                       # Bonnet recurrence for P_{l+1}
            p_prev, p = p, ((2 * l + 1) * c * p - l * p_prev) / (l + 1)
            proj[:, l + 1] = (2 * l + 3) * (p @ source) / (4 * np.pi)
        out[start:start + chunk] = proj @ phases
    return out


# Torus of revolution x(u, v) = ((R + r cos v) cos u, (R + r cos v) sin u, r sin v), R > r > 0.
# Metric ds² = ρ² du² + r² dv² with ρ(v) = R + r cos v, area element dA = r ρ du dv. With
# f = e^{imu} g(v), -Δf = λf becomes the periodic Sturm–Liouville problem
# -(ρ g')'/r² + m² g/ρ = λ ρ g for every integer m, solved by Fourier collocation in v.
# ψ0 is expanded in these eigenfunctions (FFT in u, weighted projection in v), every term
# gets its phase e^{-iλt/2}, and ψ is evaluated anywhere by exact trigonometric
# interpolation in u and v. All steps are spectrally accurate.


def _check_torus(major_radius: float, minor_radius: float) -> None:
    if not (np.isfinite(major_radius) and 0 < minor_radius < major_radius):
        raise ValueError("the torus needs finite radii with major_radius > minor_radius > 0")


def _wrap_angle(angle: NDArray) -> NDArray:
    """Angle in [0, 2π); ``np.mod`` alone returns 2π for tiny negative inputs."""
    angle = np.mod(angle, 2 * np.pi)
    return np.where(angle >= 2 * np.pi, 0.0, angle)


def torus_points(u: NDArray, v: NDArray, major_radius: float = 1.0, minor_radius: float = 0.5) -> NDArray:
    """Points ``x(u, v)`` on the torus, shape ``broadcast(u, v).shape + (3,)``."""
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    rho = major_radius + minor_radius * np.cos(v)
    return np.stack([rho * np.cos(u), rho * np.sin(u), minor_radius * np.sin(v)], axis=-1)


def torus_angles(points: NDArray, major_radius: float = 1.0) -> tuple[NDArray, NDArray]:
    """Angles ``(u, v)`` in [0, 2π) of points on the torus, the inverse of ``torus_points``.

    ``u`` turns around the z axis and ``v`` around the tube, from the outer equator
    (v = 0) through the top (v = π/2) to the inner equator (v = π). A point off the
    surface gets the angles of its nearest surface point, so ``r`` is not needed.
    """
    points = np.asarray(points, float)
    u = np.arctan2(points[..., 1], points[..., 0])
    v = np.arctan2(points[..., 2], np.hypot(points[..., 0], points[..., 1]) - major_radius)
    return _wrap_angle(u), _wrap_angle(v)


def torus_resolution(k0: float, sigma: float, major_radius: float = 1.0,
                     minor_radius: float = 0.5) -> tuple[int, int]:
    """Default odd grid sizes ``(n_u, n_v)`` of ``torus_exact_solution``.

    With ``K = |k0| + 6.5/σ`` (beyond it the packet's spectral weight ``e^{-σ²k²}``
    is below 1e-18), u modes go up to ``|m| ≤ ⌈K (R + r)⌉ + 20`` (a ring of radius
    ρ ≤ R + r carries wavenumber m/ρ) and v modes up to
    ``⌈1.5 K r⌉ + ⌈30 / arccosh(R/r)⌉``. The margin in v keeps the eigenfunctions
    resolved: their Fourier coefficients decay like ``e^{-arccosh(R/r) |k|}`` (1/ρ has
    poles at cos v = -R/r), slowly on fat tori, where the cost grows as r → R. On
    R = 1, r = 0.5 this gives 241 × 147 for k0 = 12, σ = 0.12 and 143 × 99 for
    k0 = 12, σ = 0.3. Margin, measured there for k0 ≤ 12, σ ∈ [0.12, 0.3], t ≤ 0.5:
    20 fewer u modes still give round-off (≤ 3e-12), half the v grid ≤ 1e-9.
    """
    _check_torus(major_radius, minor_radius)
    if not (sigma > 0 and np.isfinite(sigma) and np.isfinite(k0)):
        raise ValueError("sigma must be positive and k0 finite")
    K = abs(k0) + 6.5 / sigma
    m_max = int(np.ceil(K * (major_radius + minor_radius))) + 20
    strip = np.arccosh(major_radius / minor_radius)
    k_max = int(np.ceil(1.5 * K * minor_radius)) + int(np.ceil(30 / strip))
    return 2 * m_max + 1, 2 * k_max + 1


def _tube_grid(n_v: int, major_radius: float, minor_radius: float) -> tuple[NDArray, NDArray]:
    """``v_j = 2πj/n_v`` and ``ρ(v_j)``, with ρ exactly even (``ρ_j = ρ_{n_v - j}``)."""
    j = np.arange(n_v)
    return 2 * np.pi * j / n_v, major_radius + minor_radius * np.cos(2 * np.pi * np.minimum(j, n_v - j) / n_v)


def _fourier_derivative(n: int) -> NDArray:
    """d/dv on ``n`` (odd) equispaced periodic points: ``D_jk = ½ (-1)^{j-k} / sin((j-k)π/n)``.

    Built as an exactly antisymmetric circulant, so it anticommutes exactly with the
    reflection v → -v.
    """
    p = np.arange(1, (n + 1) // 2)
    d = np.zeros(n)
    d[p] = 0.5 * (-1.0)**p / np.sin(p * np.pi / n)
    d[n - p] = -d[p]
    j = np.arange(n)
    return d[(j[:, None] - j[None, :]) % n]


def _tube_operator(n_v: int, major_radius: float, minor_radius: float) -> tuple[NDArray, NDArray, NDArray]:
    """``ρ(v_j)`` and ``diag(ρ)^{-1/2} Dᵀ diag(ρ) D diag(ρ)^{-1/2} / r²`` in the even and odd bases.

    The bases ``e_0, (e_j + e_{n-j})/√2`` and ``(e_j - e_{n-j})/√2`` (j = 1 .. (n_v - 1)/2)
    are orthonormal and block-diagonalize every operator that commutes with v → -v.
    """
    _, rho = _tube_grid(n_v, major_radius, minor_radius)
    D = _fourier_derivative(n_v)
    scale = 1 / np.sqrt(rho)
    A = scale[:, None] * (D.T @ (rho[:, None] * D)) * scale[None, :] / minor_radius**2
    A = 0.5 * (A + A.T)
    j = np.arange(1, (n_v + 1) // 2)
    same = A[np.ix_(j, j)] + A[np.ix_(n_v - j, n_v - j)]
    cross = A[np.ix_(j, n_v - j)] + A[np.ix_(n_v - j, j)]
    even = np.empty((len(j) + 1, len(j) + 1))
    even[0, 0] = A[0, 0]
    even[0, 1:] = even[1:, 0] = (A[0, j] + A[0, n_v - j]) / np.sqrt(2)
    even[1:, 1:] = 0.5 * (same + cross)
    return rho, even, 0.5 * (same - cross)


def _tube_eigenpairs(m: int, rho: NDArray, even: NDArray, odd: NDArray,
                     minor_radius: float) -> tuple[NDArray, NDArray]:
    """``torus_eigenpairs`` from the output of ``_tube_operator``: two half-size ``eigh``."""
    n_v, h = len(rho), len(odd)
    j = np.arange(1, h + 1)
    q = float(m)**2 / rho**2                     # diag(ρ)^{-1/2} diag(m²/ρ) diag(ρ)^{-1/2}
    # MRRR (driver "evr"): as accurate as the default here and several times faster on these
    # small matrices when the BLAS runs multithreaded.
    lam_even, y_even = sla.eigh(even + np.diag(q[:h + 1]), driver="evr", check_finite=False)
    lam_odd, y_odd = sla.eigh(odd + np.diag(q[1:h + 1]), driver="evr", check_finite=False)
    G = np.zeros((n_v, n_v))
    G[0, :h + 1] = y_even[0]
    G[j, :h + 1] = G[n_v - j, :h + 1] = y_even[1:] / np.sqrt(2)
    G[j, h + 1:] = y_odd / np.sqrt(2)
    G[n_v - j, h + 1:] = -y_odd / np.sqrt(2)
    evals = np.concatenate([lam_even, lam_odd])
    order = np.argsort(evals, kind="stable")
    G /= np.sqrt(rho * (2 * np.pi)**2 * minor_radius / n_v)[:, None]
    return evals[order], G[:, order]


def torus_eigenpairs(m: int, n_v: int, major_radius: float = 1.0,
                     minor_radius: float = 0.5) -> tuple[NDArray, NDArray]:
    """Eigenpairs of ``-Δ (e^{imu} g) = λ e^{imu} g`` for one Fourier mode ``m`` in u.

    ``g`` solves ``-(ρ g')'/r² + m² g/ρ = λ ρ g`` on the circle, by Fourier collocation
    on ``v_j = 2πj/n_v`` (``n_v`` odd, so there is no Nyquist mode): the symmetric
    positive semidefinite ``K = Dᵀ diag(ρ) D / r² + diag(m²/ρ)`` against the weight
    ``diag(ρ)``. ρ is even in v, so the even and the odd eigenfunctions are computed
    separately (exact parity, half-size problems). Returns the ``n_v`` eigenvalues in
    ascending order and the real eigenvectors ``g(v_j)`` as columns, normalized so that
    ``∫∫ |e^{imu} g|² r ρ du dv = 1`` by the periodic trapezoid rule:
    ``(2π)² r / n_v · Gᵀ diag(ρ) G = I``. Only m² enters, so ``λ_{-m,n} = λ_{m,n}``.
    Modes whose v wavenumber is well below n_v/2 are accurate to round-off; the top of
    the discrete spectrum is not.
    """
    _check_torus(major_radius, minor_radius)
    if int(n_v) != n_v or n_v < 3 or n_v % 2 == 0:
        raise ValueError("n_v must be an odd integer >= 3")
    return _tube_eigenpairs(m, *_tube_operator(int(n_v), major_radius, minor_radius), minor_radius)


def torus_exact_solution(targets: NDArray, x0: NDArray, t0: NDArray, k0: float, sigma: float,
                         times: NDArray, major_radius: float = 1.0, minor_radius: float = 0.5, *,
                         n_u: int | None = None, n_v: int | None = None, chunk: int = 2048) -> NDArray:
    """ψ(targets, times) on the torus, with ψ0 normalized so that ∫|ψ0|² dA = 1.

    ψ0 is ``sphere_packet`` (the chordal packet of ``initial_conditions.gaussian_packet``)
    with centre ``x0`` on the torus and unit tangent ``t0`` there, used as given. It is
    sampled on an ``n_u × n_v`` (u, v) grid (default ``torus_resolution``; even sizes are
    rounded up to odd), normalized with the periodic trapezoid rule and the exact area
    element, expanded in ``e^{imu} g_{m,n}(v)`` (FFT in u, then the eigenvectors of
    ``torus_eigenpairs`` with weight ρ) and evolved with ``e^{-iλt/2}``. ``targets``
    (N, 3) are mapped to (u, v) by ``torus_angles`` and evaluated by trigonometric
    interpolation, at most ``chunk`` targets at a time (fewer for many times, to bound
    memory). Returns (N, len(times)) complex.

    Measured at the default resolution on R = 1, r = 0.5 (80 packets: outer and inner
    equator, top, generic; four directions; k0 ≤ 12, σ ∈ [0.12, 0.3]; t ≤ 0.5): ψ0 is
    reproduced to ≤ 4e-14 (max |ψ0| ≈ 1.2-3.3), doubling n_u and n_v changes ψ by
    ≤ 3e-12 (round-off in the eigenvalues) and ∫|ψ(t)|² dA = 1 to ≤ 1e-14. 16 000
    targets × 11 times take about 1 s for k0 = 12, σ = 0.12.
    """
    _check_torus(major_radius, minor_radius)
    default_u, default_v = torus_resolution(k0, sigma, major_radius, minor_radius)
    n_u, n_v = (int(n) | 1 for n in (n_u or default_u, n_v or default_v))
    if n_u < 3 or n_v < 3 or chunk < 1:
        raise ValueError("n_u and n_v must be at least 3 and chunk positive")
    targets, x0, t0 = (np.asarray(a, float) for a in (targets, x0, t0))
    if targets.ndim != 2 or targets.shape[1] != 3 or x0.shape != (3,) or t0.shape != (3,):
        raise ValueError("targets must be (N, 3), x0 and t0 3-vectors")
    times = np.atleast_1d(np.asarray(times, float))
    if len(targets) == 0 or len(times) == 0:
        return np.zeros((len(targets), len(times)), complex)
    u = 2 * np.pi * np.arange(n_u) / n_u
    v, rho = _tube_grid(n_v, major_radius, minor_radius)
    grid = torus_points(u[:, None], v[None, :], major_radius, minor_radius).reshape(-1, 3)
    psi0 = sphere_packet(grid, x0, t0, k0, sigma).reshape(n_u, n_v)
    psi0 /= np.sqrt((2 * np.pi)**2 * minor_radius / (n_u * n_v) * np.sum(rho * np.abs(psi0)**2))
    modes = np.fft.fft(psi0, axis=0) / n_u                      # row m: ψ0 = Σ_m modes[m](v) e^{imu}
    weight = (2 * np.pi)**2 * minor_radius / n_v * rho          # coefficients a = Gᵀ (weight ⊙ f)
    operator = _tube_operator(n_v, major_radius, minor_radius)
    evolved = np.empty((n_v, n_u, len(times)), complex)         # ψ(t) at (v_j, mode m)
    for m in range(n_u // 2 + 1):
        lam, G = _tube_eigenpairs(m, *operator, minor_radius)
        rows = [m] if m == 0 else [m, n_u - m]                  # +m and -m share the eigenpairs
        # Coefficients and evolution as real products (complex data viewed as re/im pairs).
        data = np.ascontiguousarray(weight[:, None] * modes[rows].T)
        coeffs = (G.T @ data.view(float)).view(complex)          # (n, len(rows))
        terms = coeffs[:, :, None] * np.exp(-0.5j * np.outer(lam, times))[:, None, :]
        evolved[:, rows] = (G @ terms.reshape(n_v, -1).view(float)).view(complex).reshape(terms.shape)
    coef = (np.fft.fft(evolved, axis=0) / n_v).reshape(n_v, -1)  # ψ = Σ coef[k, m] e^{ikv} e^{imu}
    k_of, m_of = np.fft.fftfreq(n_v, 1.0 / n_v), np.fft.fftfreq(n_u, 1.0 / n_u)
    target_u, target_v = torus_angles(targets, major_radius)
    out = np.empty((len(targets), len(times)), complex)
    step = max(1, min(chunk, (1 << 22) // max(1, n_u * len(times))))   # in_v below: ≤ 64 MB
    for start in range(0, len(targets), step):
        part = slice(start, start + step)
        in_v = (np.exp(1j * np.outer(target_v[part], k_of)) @ coef).reshape(-1, n_u, len(times))
        out[part] = np.matmul(np.exp(1j * np.outer(target_u[part], m_of))[:, None, :], in_v)[:, 0]
    return out
