"""Check of the Airy matching at the end of the stabilized phase (clean model, linearized).
 dd(delta) + 3H d(delta) - (V0 + kap*chi(t)) delta = 0,
 chi(t) = chibar + A(t) cos(m t),  A = A0 (a0/a)^{3/2}, radiation or matter domination.
Prediction: envelope of x = a^{3/2} delta over initial phases at s=(t-tc)/tau:
   x_env(s) = sqrt(pi) * calA * sqrt(Ai(s)^2 + Bi(s)^2),  calA = X_i (omega_i tau)^{1/2},
   tau = (3 mt^2 H_c)^{-1/3};  delta_rel = calA / a_c^{3/2} = delta_i e^{-3dN/4} (mt tau)^{1/2} (1-e^{-3dN})^{1/4}."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import airy
V0, k, m = 1., 200., 400.
chibar = -k/m**2; mt2 = V0 + k*chibar; mt = np.sqrt(mt2)
Ath = np.sqrt(2*m*m*mt2)/k

def run(A0, p=0.5, H0=0.1, s_eval=(-3, -2, 2, 3), delta_i=1e-3):
    t0 = p/H0
    a = lambda t: (t/t0)**p
    H = lambda t: p/t
    A = lambda t: A0*a(t)**-1.5
    dN = (2/3)*np.log(A0/Ath)
    tc = t0*np.exp(dN/p); Hc = p/tc; tau = (3*mt2*Hc)**(-1/3)
    wi = np.sqrt(k*k*A0*A0/(2*m*m) - mt2)
    def f(t, u):
        d, dd = u
        return [dd, -3*H(t)*dd + (V0 + k*(chibar + A(t)*np.cos(m*t)))*d]
    tev = tc + np.array(s_eval)*tau
    xs = []
    # slow-mode initial data dressed with the fast mode: delta = ds(1 - q/2 cos mt), d(delta) = dds + (kap A/m) sin(mt) ds
    Ai0 = A(t0); qh = k*Ai0/m**2
    def dress(ds, dds): return [ds*(1 - qh*np.cos(m*t0)), dds + (k*Ai0/m)*np.sin(m*t0)*ds]
    for y0 in (dress(delta_i, 0.), dress(0., wi*delta_i)):   # two quadrature phases, same adiabatic amplitude
        s = solve_ivp(f, [t0, tev[-1]], y0, t_eval=tev, rtol=1e-10, atol=1e-14, method="DOP853")
        xs.append(s.y[0]*a(s.t)**1.5)
    xs = np.array(xs); env_num = np.sqrt(xs[0]**2 + xs[1]**2)
    calA = delta_i*a(t0)**1.5*np.sqrt(wi*tau)
    Ai, _, Bi, _ = airy(np.array(s_eval, float))
    env_pred = np.sqrt(np.pi)*calA*np.sqrt(Ai**2 + Bi**2)
    c_pred = (mt*tau)**0.5*(1-np.exp(-3*dN))**0.25
    return dN, Hc, tau, env_num/env_pred, c_pred

if __name__ == "__main__":
    for p, name in [(0.5, "RD"), (2/3, "MD")]:
        for A0 in [10., 40., 160.]:
            dN, Hc, tau, r, c = run(A0, p)
            print(f"{name} A0={A0:5}: dN={dN:.2f} H_c={Hc:.2e} H_c*tau={Hc*tau:.3f}  numeric/Airy at s=-3,-2,2,3: "
                  + " ".join(f"{x:.3f}" for x in r) + f"   c_pred={(mt/(3*Hc))**(1/6):.2f} (exact {c:.2f})")
