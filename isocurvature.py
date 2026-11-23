"""Isocurvature test for the hilltop attractor (clean model, full nonlinear, radiation domination).
V = V0(1-cos phi) - kap chi cos phi + m^2 chi^2/2 with kap = 5e3, m = 1e4 (m~^2 = 0.75), H0 = 0.1, A0 = 50,
which satisfy the no-resonance condition 2 m~^2 e^{3 Delta N} < m H_i (ratio 0.63). Fast integrator fastkapitza.py.
Initial phi = pi - delta_i (slow mode dressed with the fast mode), chi oscillating.
Measures (a) the map delta_i -> delta at t_c + 2 tau (linearity), (b) the final comoving number density
n = a^3 E/omega of phi oscillating around a minimum at late times, and its log-derivative d ln n / d ln delta_i,
compared with the 'tuned' scenario: no oscillation (A=0) and phi starting at the hilltop with deviation delta at t_c."""
import numpy as np
from fastkapitza import integrate, NS
V0, k, m = 1., 5e3, 1e4
chibar = -k/m**2; mt2 = V0 + k*chibar; mt = np.sqrt(mt2)
Ath = np.sqrt(2*m*m*mt2)/k
p, H0 = 0.5, 0.1; t0 = p/H0
A0 = 50.
dN = (2/3)*np.log(A0/Ath); tc = t0*np.exp(dN/p); Hc = p/tc; tau = (3*mt2*Hc)**(-1/3)
T1 = tc + 2*tau

def init(d, A0=A0):
    """(psi, chi, dpsi, dchi) with psi = pi - phi: slow mode psi = d dressed with the fast mode."""
    qh = k*A0/m**2
    return np.array([d*(1 - qh*np.cos(m*t0)), -k/m**2 + A0*np.cos(m*t0),
                     (k*A0/m)*np.sin(m*t0)*np.sin(d), -m*A0*np.sin(m*t0)])

def deviation_at_T1(d):
    _, y, _ = integrate(init(d), t0, T1, V0, k, m, p, H0, NS, 2, 1e300, True)
    return y[0, -1]

def run(delta_i, tend_factor=6.):
    _, y, _ = integrate(init(delta_i), t0, T1, V0, k, m, p, H0, NS, 2, 1e300, True)
    d2 = y[0, -1]
    Tend = tc*tend_factor
    _, _, n = integrate(y[:, -1].copy(), T1, Tend, V0, k, m, p, H0, NS, 2, 0.9*Tend, True)
    return d2, n

def run_tuned(delta, tend_factor=6.):
    # no stabilizer: chi sits at its minimum; phi starts at pi - delta, at rest, at t = tc
    y0 = np.array([delta, -k*np.cos(delta)/m**2, 0., 0.])
    Tend = tc*tend_factor
    _, _, n = integrate(y0, tc, Tend, V0, k, m, p, H0, NS, 2, 0.9*Tend, True)
    return n

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    ds = [0.05, 0.1, 0.2, 0.4, 0.8]
    with ProcessPoolExecutor(8) as ex: res = list(ex.map(run, ds))
    print(f"m~/H_c = {mt/Hc:.0f}, no-resonance ratio = {k**2*A0**2/(m**3*H0):.2f}")
    print("delta_i   delta(tc+2tau)/delta_i    n_late/n_late(0.05)")
    for d, (d2, n) in zip(ds, res): print(f"{d:6.2f}   {d2/d:12.4f}          {n/res[0][1]:.4f}")
