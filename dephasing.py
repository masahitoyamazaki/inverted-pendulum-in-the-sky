"""Dephasing test: ratio delta(t_c + 2 tau)/delta_i vs delta_i for the full nonlinear clean model (RD),
compared with the linearized equation. Linear theory => constant ratio."""
import numpy as np
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
from isocurvature import rhs, V0, k, m, mt2, Ath, p, t0, a
A0 = 50.
dN = (2/3)*np.log(A0/Ath); tc = t0*np.exp(dN/p); Hc = p/tc; tau = (3*mt2*Hc)**(-1/3)
T1 = tc + 2*tau
def init(d):
    qh = k*A0/m**2
    return [np.pi - d*(1 - qh*np.cos(m*t0)), -k/m**2 + A0*np.cos(m*t0), -(k*A0/m)*np.sin(m*t0)*np.sin(d), -m*A0*np.sin(m*t0)]
def nonlin(d):
    s = solve_ivp(rhs, [t0, T1], init(d), rtol=1e-11, atol=1e-14, method="DOP853")
    return (np.pi - s.y[0, -1])/d
def lin(d=1e-6):
    def f(t, u):
        dl, c, ddl, dc = u; h = p/t
        return [ddl, dc, -3*h*ddl + (V0 + k*c)*dl, -3*h*dc - k - m*m*c]
    y = init(d); y0 = [d*(1 - k*A0/m**2*np.cos(m*t0)), y[1], (k*A0/m)*np.sin(m*t0)*d, y[3]]
    s = solve_ivp(f, [t0, T1], y0, rtol=1e-11, atol=1e-14, method="DOP853")
    return s.y[0, -1]/d
if __name__ == "__main__":
    ds = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2]
    with ProcessPoolExecutor() as ex: r = list(ex.map(nonlin, ds))
    print(f"m~/H_c = {np.sqrt(mt2)/Hc:.0f}, linear ratio = {lin():.4f}")
    for d, x in zip(ds, r): print(f"delta_i={d:7.0e}  ratio={x:9.4f}")
