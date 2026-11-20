"""Isocurvature test for the hilltop attractor (clean model, full nonlinear, radiation domination).
V = V0(1-cos phi) - kap chi cos phi + m^2 chi^2/2.  Initial phi = pi - delta_i (slow mode, dressed), chi oscillating.
Measures (a) the map delta_i -> delta at t_c + 2 tau (linearity), (b) the final comoving number density
n = a^3 E/omega of phi oscillating around phi=0 at late times, and its log-derivative d ln n / d ln delta_i,
compared with the 'tuned' scenario: no oscillation (A=0) and phi starting at the hilltop with deviation delta at t_c."""
import numpy as np
from scipy.integrate import solve_ivp
V0, k, m = 1., 50., 100.
chibar = -k/m**2; mt2 = V0 + k*chibar; mt = np.sqrt(mt2)
Ath = np.sqrt(2*m*m*mt2)/k
p, H0 = 0.5, 0.1; t0 = p/H0
a = lambda t: (t/t0)**p

def rhs(t, u):
    ph, c, dph, dc = u; h = p/t
    return [dph, dc, -3*h*dph - V0*np.sin(ph) - k*c*np.sin(ph), -3*h*dc + k*np.cos(ph) - m*m*c]

def late_n(sol_t, y):
    # comoving number density of phi around phi=0: n = a^3 (E/omega), E = 1/2 dphi^2 + V0(1-cos phi) + ..., omega^2 = V0 + k^2/m^2
    w = np.sqrt(V0 + k*k/m**2)
    E = 0.5*y[2]**2 + 0.5*w*w*(y[0]-2*np.pi*np.round(y[0]/(2*np.pi)))**2
    return np.mean(a(sol_t)**3*E/w)

def run(delta_i, A0=50., tend_factor=6.):
    dN = (2/3)*np.log(A0/Ath); tc = t0*np.exp(dN/p); Hc = p/tc; tau = (3*mt2*Hc)**(-1/3)
    qh = k*A0/m**2
    ph0 = np.pi - delta_i*(1 - qh*np.cos(m*t0)); dph0 = -(k*A0/m)*np.sin(m*t0)*np.sin(delta_i)
    y0 = [ph0, chibar*np.cos(ph0)/(-1) * (-1) if False else -k/m**2*1 + A0*np.cos(m*t0) + 0*ph0, dph0, -m*A0*np.sin(m*t0)]
    y0[1] = (k*np.cos(np.pi)/m**2) + A0*np.cos(m*t0)
    T1 = tc + 2*tau; Tend = tc*tend_factor
    s = solve_ivp(rhs, [t0, Tend], y0, rtol=1e-10, atol=1e-12, method="DOP853", dense_output=True)
    d2 = np.pi - s.sol(T1)[0]
    tl = np.linspace(Tend*0.9, Tend, 4000); yl = s.sol(tl)
    return d2, late_n(tl, yl), tc, tau

def run_tuned(delta, tc, tend_factor=6.):
    # no stabilizer: chi sits at its minimum; phi starts at pi - delta, at rest, at t = tc
    y0 = [np.pi - delta, k*np.cos(np.pi - delta)/m**2, 0., 0.]
    Tend = tc*tend_factor
    s = solve_ivp(rhs, [tc, Tend], y0, rtol=1e-10, atol=1e-12, method="DOP853", dense_output=True)
    tl = np.linspace(Tend*0.9, Tend, 4000); return late_n(tl, s.sol(tl))

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    ds = [0.05, 0.1, 0.2, 0.4, 0.8]
    eps = 0.01
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, ds))
        extra = {d: list(ex.map(run, [d*(1+eps), d*(1-eps)])) for d in [0.1, 0.4]}
    print("delta_i   delta(tc+2tau)/delta_i    n_late")
    for d, (d2, n, tc, tau) in zip(ds, res): print(f"{d:6.2f}   {d2/d:12.4f}          {n:.5g}")
    for d in [0.1, 0.4]:
        (d2p, np_, tc, _), (d2m, nm, _, _) = extra[d]
        slope = (np.log(np_)-np.log(nm))/(2*eps)
        dr = abs(res[ds.index(d)][0])
        ntp = run_tuned(dr*(1+eps), tc); ntm = run_tuned(dr*(1-eps), tc)
        slope_t = (np.log(ntp)-np.log(ntm))/(2*eps)
        print(f"delta_i={d}: dln n/dln delta_i = {slope:.4f};  tuned scenario at delta={dr:.4f}: dln n/dln delta = {slope_t:.4f}")
