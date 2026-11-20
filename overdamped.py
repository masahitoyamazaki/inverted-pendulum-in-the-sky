"""(i) Overdamped regime: release at H_c >~ m~ in RD -> attraction factor ~ exp(-int omega^2/(3H) dt) = O(1).
(ii) Delayed release: release at H_c << m~ in RD -> late-time number density enhanced ~ (m~/H_c)^{3/2},
compared with standard misalignment from theta_i = 1 (no oscillating chi).
Model: V = V0(1-cos phi) - kap chi cos phi + m^2 chi^2/2, V0=1, kap=50, m=100 (m~^2 = 0.75)."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from concurrent.futures import ProcessPoolExecutor
V0, k, m = 1., 50., 100.
mt2 = V0 - k*k/m**2; mt = np.sqrt(mt2); Ath = np.sqrt(2*m*m*mt2)/k
p = 0.5
def rhs(t, u):
    ph, c, dph, dc = u; h = p/t
    return [dph, dc, -3*h*dph - V0*np.sin(ph) - k*c*np.sin(ph), -3*h*dc + k*np.cos(ph) - m*m*c]
def init(t0, A0, d):
    qh = k*A0/m**2
    return [np.pi - d*(1 - qh*np.cos(m*t0)), -k/m**2 + A0*np.cos(m*t0), -(k*A0/m)*np.sin(m*t0)*np.sin(d), -m*A0*np.sin(m*t0)]
def overdamped(Hc_over_mt, H0=10., d=0.1):
    Hc = Hc_over_mt*mt; A0 = Ath*(H0/Hc)**0.75; t0 = p/H0; tc = p/Hc
    s = solve_ivp(rhs, [t0, tc], init(t0, A0, d), rtol=1e-10, atol=1e-13, method="DOP853")
    ratio = (np.pi - s.y[0, -1])/d
    # prediction: exp(-int omega_eff^2/(3H) dt), omega_eff^2 = mt2((A/Ath)^2 - 1), A = A0 (t0/t)^{3/4}
    I = quad(lambda t: mt2*((A0*(t0/t)**0.75/Ath)**2 - 1)/(3*p/t), t0, tc)[0]
    return Hc_over_mt, ratio, np.exp(-I)
def late_n(t, y):
    w = np.sqrt(V0 + k*k/m**2)
    ph = y[0] - 2*np.pi*np.round(y[0]/(2*np.pi))
    return np.mean((t/t[0])**1.5*t[0]**1.5*(0.5*y[2]**2 + 0.5*w*w*ph**2)/w)
def delayed(Hc_over_mt, H0=10., d=0.1):
    Hc = Hc_over_mt*mt; A0 = Ath*(H0/Hc)**0.75; t0 = p/H0; tc = p/Hc; Tend = 8*tc
    s = solve_ivp(rhs, [t0, Tend], init(t0, A0, d), rtol=1e-9, atol=1e-11, method="DOP853", dense_output=True)
    tl = np.linspace(0.9*Tend, Tend, 4000); return Hc_over_mt, late_n(tl, s.sol(tl))
def standard(theta=1.0, H0=10.):
    t0 = p/H0; Tend = 8*p/(0.01*mt)
    s = solve_ivp(rhs, [t0, Tend], [theta, k/m**2*np.cos(theta), 0., 0.], rtol=1e-9, atol=1e-11, method="DOP853", dense_output=True)
    tl = np.linspace(0.9*Tend, Tend, 4000); return late_n(tl, s.sol(tl))
if __name__ == "__main__":
    with ProcessPoolExecutor() as ex:
        od = list(ex.map(overdamped, [3., 2., 1., 0.5]))
        dl = list(ex.map(delayed, [0.1, 0.03, 0.01, 0.003]))
        n0 = ex.submit(standard).result()
    print("(i) release at H_c/m~ : delta(t_c)/delta_i  vs  exp(-int w^2/3H)")
    for x, r, pr in od: print(f"   {x:5}:  {r:.3f}   {pr:.3f}")
    print("(ii) delayed release: n/n_std(theta=1)   and ratio / (m~/H_c)^{3/2}")
    for x, n in dl: print(f"   H_c/m~={x:6}:  {n/n0:10.2f}   {n/n0*x**1.5:.3f}")
