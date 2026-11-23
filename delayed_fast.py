"""Delayed release (H_c << m~) in RD with the fast integrator (fastkapitza.py).
Late-time number density of phi, relative to the standard misalignment from theta_i = 1.
Default: kap/m = 0.5 (m~^2 = 0.75), m_chi = 3e4, H0 = 3, so that both q(A0) < 0.91 and the
no-resonance condition 2 m~^2 e^{3 Delta N} < m_chi H_i hold for all H_c/m~ >= 0.003."""
import sys, numpy as np
from fastkapitza import integrate, NS
V0, p, H0 = 1., 0.5, 3.
def params(m):
    k = 0.5 * m; mt2 = V0 - k * k / m**2
    return k, np.sqrt(mt2), np.sqrt(2 * m * m * mt2) / k, 0.91 * m * m / (2 * k)
def delayed(x, m=3e4, d=0.1, ns=NS):
    k, mt, Ath, Amax = params(m)
    Hc = x * mt; A0 = Ath * (H0 / Hc)**0.75; t0 = p / H0; tc = p / Hc; Tend = 8 * tc
    assert A0 < Amax
    res = 2 * mt**2 * (A0 / Ath)**2 / (m * H0)          # no-resonance ratio, should be < 1
    qh = k * A0 / m**2
    y0 = np.array([np.pi - d * (1 - qh * np.cos(m * t0)), -k / m**2 + A0 * np.cos(m * t0),
                   -(k * A0 / m) * np.sin(m * t0) * np.sin(d), -m * A0 * np.sin(m * t0)])
    _, _, n = integrate(y0, t0, Tend, V0, k, m, p, H0, ns, 2, 0.9 * Tend)
    return x, n, res
def standard(m=3e4, theta=1.0, ns=NS):
    k, mt, _, _ = params(m)
    t0 = p / H0; Tend = 8 * p / (0.01 * mt)
    y0 = np.array([theta, k / m**2 * np.cos(theta), 0., 0.])
    _, _, n = integrate(y0, t0, Tend, V0, k, m, p, H0, ns, 2, 0.9 * Tend)
    return n
