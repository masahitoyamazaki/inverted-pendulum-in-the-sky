"""Delayed release (H_c << m~) in RD: late-time number density vs standard misalignment from theta_i=1.
m_chi=1000, kap=500 (kap/m = 0.5, m~^2=0.75), H0=3, so that q(A0) < 0.91 for all runs."""
import numpy as np
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
V0, k, m = 1., 500., 1000.
mt2 = V0 - k*k/m**2; mt = np.sqrt(mt2); Ath = np.sqrt(2*m*m*mt2)/k; Amax = 0.91*m*m/(2*k)
p, H0 = 0.5, 3.
def rhs(t, u):
    ph, c, dph, dc = u; h = p/t
    return [dph, dc, -3*h*dph - V0*np.sin(ph) - k*c*np.sin(ph), -3*h*dc + k*np.cos(ph) - m*m*c]
def late_n(t, y):
    w = np.sqrt(V0 + k*k/m**2); ph = y[0] - 2*np.pi*np.round(y[0]/(2*np.pi))
    return np.mean(t**1.5*(0.5*y[2]**2 + 0.5*w*w*ph**2)/w)
def delayed(x, d=0.1):
    Hc = x*mt; A0 = Ath*(H0/Hc)**0.75; t0 = p/H0; tc = p/Hc; Tend = 8*tc
    assert A0 < Amax, (x, A0, Amax)
    qh = k*A0/m**2
    y0 = [np.pi - d*(1 - qh*np.cos(m*t0)), -k/m**2 + A0*np.cos(m*t0), -(k*A0/m)*np.sin(m*t0)*np.sin(d), -m*A0*np.sin(m*t0)]
    s = solve_ivp(rhs, [t0, Tend], y0, rtol=1e-9, atol=1e-11, method="DOP853", dense_output=True)
    tl = np.linspace(0.9*Tend, Tend, 4000); return x, late_n(tl, s.sol(tl))
def standard(theta=1.0):
    t0 = p/H0; Tend = 8*p/(0.01*mt)
    s = solve_ivp(rhs, [t0, Tend], [theta, k/m**2*np.cos(theta), 0., 0.], rtol=1e-9, atol=1e-11, method="DOP853", dense_output=True)
    tl = np.linspace(0.9*Tend, Tend, 4000); return late_n(tl, s.sol(tl))
if __name__ == "__main__":
    xs = [0.1, 0.03, 0.01, 0.003]
    with ProcessPoolExecutor() as ex:
        f = ex.submit(standard); dl = list(ex.map(delayed, xs)); n0 = f.result()
    print("H_c/m~   n/n_std(theta=1)   (n/n_std)*(H_c/m~)^{3/2}")
    for x, n in dl: print(f"{x:7}  {n/n0:12.2f}   {n/n0*x**1.5:.3f}")
