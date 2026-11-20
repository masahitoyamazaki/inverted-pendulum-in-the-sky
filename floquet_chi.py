"""Floquet multiplier of delta-phi around phi=pi, with chi oscillating in
V = V0(1-cos phi) - b w cos(chi/g) cos phi + m^2(1-cos chi) at phi=pi (H=0).
Linearized: dd(dphi) = (V0 + b w cos(chi/g)) dphi,
            dd(chi)  = (b w/g) sin(chi/g) - m^2 sin chi."""
import numpy as np
from scipy.integrate import solve_ivp
V0, b, w, g, m2 = 1., 2., 2., 2., 3.4
def f(t, u):
    c, dc, x, dx, y, dy = u
    B = b*w*np.cos(c/g)
    return [dc, (b*w/g)*np.sin(c/g) - m2*np.sin(c), dx, (V0+B)*x, dy, (V0+B)*y]
def multiplier(chi0, tmax=400.):
    """Max |Floquet multiplier| over one period of chi (chi starts at rest => turning point).
    Returns (period, mult); period=None if chi does not return (e.g. separatrix)."""
    ev = lambda t, u: u[1]
    ev.direction = 1 if (b*w/g)*np.sin(chi0/g) - m2*np.sin(chi0) > 0 else -1
    s = solve_ivp(f, [0, tmax], [chi0, 0, 1, 0, 0, 1], rtol=1e-11, atol=1e-12,
                  events=ev, dense_output=True)
    te = [t for t in s.t_events[0] if t > 1e-6]
    if not te: return None, np.nan
    Tp = te[0]; u = s.sol(Tp)
    if abs(u[0]-chi0) > 1e-4: return None, np.nan
    M = np.array([[u[2], u[4]], [u[3], u[5]]])
    return Tp, np.max(np.abs(np.linalg.eigvals(M)))
if __name__ == "__main__":
    for c0 in np.linspace(0.05, 4*np.pi-0.05, 60):
        T, mu = multiplier(c0)
        print(f"chi0={c0:6.3f}  T={T if T is None else round(T,3)}  |mult|={mu:.4g}")
