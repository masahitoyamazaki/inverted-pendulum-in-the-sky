"""Fast fixed-step RK4 integrator (numba) for the clean Kapitza model
    V = V0 (1 - cos phi) - kap chi cos phi + m^2 chi^2 / 2
in an FRW background with H = H0 (de Sitter, p = 0) or H = p/t (p = 1/2: RD, p = 2/3: MD).
Used for parameters with a large hierarchy m_chi >> m~, for which the parametric
resonance condition (no-resonance bound in Sec. V B) is satisfied.

The step is dt = 2 pi / (m * ns); ns = 200 gives relative errors < 1e-3 (checked by
comparing ns = 100 and 200, and against scipy DOP853 for smaller m)."""
import numpy as np
from numba import njit


@njit(cache=True)
def _rhs(t, u, V0, kap, m, p, H0, out):
    h = H0 if p == 0.0 else p / t
    ph, c, dph, dc = u[0], u[1], u[2], u[3]
    s = np.sin(ph)
    out[0] = dph
    out[1] = dc
    out[2] = -3 * h * dph - V0 * s - kap * c * s
    out[3] = -3 * h * dc + kap * np.cos(ph) - m * m * c


@njit(cache=True)
def integrate(y0, t0, T1, V0, kap, m, p, H0, ns, nout, tavg):
    """Integrate from t0 to T1. Returns (ts, ys) sampled at nout points and the
    average of t^{3/2} * (number density of phi around its nearest minimum) over [tavg, T1]
    (comoving number density in RD, a^3 = t^{3/2}); tavg >= T1 disables the average."""
    dt = 2 * np.pi / (m * ns)
    nstep = int((T1 - t0) / dt) + 1
    dt = (T1 - t0) / nstep
    stride = max(nstep // (nout - 1), 1)
    ts = np.empty(nout); ys = np.empty((4, nout))
    u = y0.copy(); k1 = np.empty(4); k2 = np.empty(4); k3 = np.empty(4); k4 = np.empty(4); tmp = np.empty(4)
    w = np.sqrt(V0 + kap * kap / (m * m))
    acc = 0.0; nacc = 0; j = 0
    t = t0
    for i in range(nstep + 1):
        if i % stride == 0 and j < nout:
            ts[j] = t
            for l in range(4): ys[l, j] = u[l]
            j += 1
        if t >= tavg:
            ph = u[0] - 2 * np.pi * np.round(u[0] / (2 * np.pi))
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
