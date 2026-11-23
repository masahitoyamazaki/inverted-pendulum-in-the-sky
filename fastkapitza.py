"""Fast fixed-step RK4 integrator (numba) for the clean Kapitza model
    V = V0 (1 - cos phi) - kap chi cos phi + m^2 chi^2 / 2
in an FRW background with H = H0 (de Sitter, p = 0) or H = p/t (p = 1/2: RD, p = 2/3: MD).
Used for parameters with a large hierarchy m_chi >> m~, for which the parametric
resonance condition (no-resonance bound in Sec. V B) is satisfied.

Internally the deviation psi = pi - phi from the hilltop is evolved, so that tiny deviations
(delta_i ~ 1e-6) are resolved with full relative precision.
The step is dt = 2 pi / (m * ns). RK4 damps the fast oscillation by ~(2 pi/ns)^6/144 per step,
so that the relative amplitude error after n steps is ~ n (2 pi/ns)^6/144. The default NS = 400 keeps it
below ~1e-3 for the runs in this repository (up to ~1e10 steps; the Airy check uses ns = 800);
checked by comparing ns = 400 and 800, and against scipy DOP853 for smaller m."""
import numpy as np
from numba import njit

NS = 400


@njit(cache=True)
def _rhs(t, u, V0, kap, m, p, H0, out):
    # u = (psi, chi, dpsi, dchi), psi = pi - phi:  sin(phi) = sin(psi), cos(phi) = -cos(psi)
    h = H0 if p == 0.0 else p / t
    s = np.sin(u[0])
    out[0] = u[2]
    out[1] = u[3]
    out[2] = -3 * h * u[2] + V0 * s + kap * u[1] * s
    out[3] = -3 * h * u[3] - kap * np.cos(u[0]) - m * m * u[1]


@njit(cache=True)
def integrate(y0, t0, T1, V0, kap, m, p, H0, ns, nout, tavg, dev=False):
    """Integrate from t0 to T1. y0 = (phi, chi, dphi, dchi), or (psi, chi, dpsi, dchi) if dev.
    Returns (ts, ys) sampled at nout points (in the same convention as y0; the last sample is
    the state at T1), and the average of t^{3/2} * (number density of phi around its nearest
    minimum) over [tavg, T1] (comoving number density in RD, a^3 = t^{3/2});
    tavg >= T1 disables the average."""
    dt = 2 * np.pi / (m * ns)
    nstep = int((T1 - t0) / dt) + 1
    dt = (T1 - t0) / nstep
    stride = max(nstep // (nout - 1), 1)
    ts = np.empty(nout); ys = np.empty((4, nout))
    u = y0.copy()
    if not dev:
        u[0] = np.pi - y0[0]; u[2] = -y0[2]
    k1 = np.empty(4); k2 = np.empty(4); k3 = np.empty(4); k4 = np.empty(4); tmp = np.empty(4)
    w = np.sqrt(V0 + kap * kap / (m * m))
    acc = 0.0; nacc = 0; j = 0
    t = t0
    for i in range(nstep + 1):
        if (i % stride == 0 and j < nout - 1) or i == nstep:
            ts[j] = t
            for l in range(4): ys[l, j] = u[l]
            if not dev:
                ys[0, j] = np.pi - u[0]; ys[2, j] = -u[2]
            j += 1
        if t >= tavg:
            ph = np.pi - u[0]
            ph = ph - 2 * np.pi * np.round(ph / (2 * np.pi))
            acc += t ** 1.5 * (0.5 * u[2] ** 2 + 0.5 * w * w * ph * ph) / w
            nacc += 1
        if i == nstep:
            break
        _rhs(t, u, V0, kap, m, p, H0, k1)
        for l in range(4): tmp[l] = u[l] + 0.5 * dt * k1[l]
        _rhs(t + 0.5 * dt, tmp, V0, kap, m, p, H0, k2)
        for l in range(4): tmp[l] = u[l] + 0.5 * dt * k2[l]
        _rhs(t + 0.5 * dt, tmp, V0, kap, m, p, H0, k3)
        for l in range(4): tmp[l] = u[l] + dt * k3[l]
        _rhs(t + dt, tmp, V0, kap, m, p, H0, k4)
        for l in range(4): u[l] += dt * (k1[l] + 2 * k2[l] + 2 * k3[l] + k4[l]) / 6
        t += dt
    navg = acc / nacc if nacc > 0 else np.nan
    return ts[:j], ys[:, :j], navg
