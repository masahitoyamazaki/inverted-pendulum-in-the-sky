"""Check of the Airy matching at the end of the stabilized phase (clean model, kap = 5e4, m = 1e5,
which satisfy the no-resonance condition for all A0 used; full equations with tiny delta_i, fast integrator).
 dd(delta) + 3H d(delta) - (V0 + kap*chi(t)) delta = 0,
 chi(t) = chibar + A(t) cos(m t),  A = A0 (a0/a)^{3/2}, radiation or matter domination.
Prediction: envelope of x = a^{3/2} delta over initial phases at s=(t-tc)/tau:
   x_env(s) = sqrt(pi) * calA * sqrt(Ai(s)^2 + Bi(s)^2),  calA = X_i (omega_i tau)^{1/2},
   tau = (3 mt^2 H_c)^{-1/3};  delta_rel = calA / a_c^{3/2} = delta_i e^{-3dN/4} (mt tau)^{1/2} (1-e^{-3dN})^{1/4}."""
import numpy as np
from scipy.special import airy
from fastkapitza import integrate
V0, k, m = 1., 5e4, 1e5
chibar = -k/m**2; mt2 = V0 + k*chibar; mt = np.sqrt(mt2)
Ath = np.sqrt(2*m*m*mt2)/k

def _one(args):
    y0, t0, tev, p, H0 = args
    out = []; t = t0; y = y0.copy()
    for te in tev:
        _, ys, _ = integrate(y, t, te, V0, k, m, p, H0, 800, 2, 1e300, True)
        y = ys[:, -1].copy(); t = te; out.append(y[0])
    return np.array(out)

def run(A0, p=0.5, H0=0.1, s_eval=(-3, -2, 2, 3), delta_i=1e-6, ex=None):
    t0 = p/H0
    a = lambda t: (t/t0)**p
    dN = (2/3)*np.log(A0/Ath)
    tc = t0*np.exp(dN/p); Hc = p/tc; tau = (3*mt2*Hc)**(-1/3)
    wi = np.sqrt(k*k*A0*A0/(2*m*m) - mt2)
    tev = tc + np.array(s_eval)*tau
    qh = k*A0/m**2
    chi0 = [chibar + A0*np.cos(m*t0), -m*A0*np.sin(m*t0)]
    # slow-mode initial data dressed with the fast mode (psi = pi - phi), two quadrature phases
    def dress(ds, dds): return np.array([ds*(1 - qh*np.cos(m*t0)), chi0[0], dds + (k*A0/m)*np.sin(m*t0)*ds, chi0[1]])
    jobs = [(dress(delta_i, 0.), t0, tev, p, H0), (dress(0., wi*delta_i), t0, tev, p, H0)]
    xs = np.array([r*a(tev)**1.5 for r in (ex.map(_one, jobs) if ex else map(_one, jobs))])
    env_num = np.sqrt(xs[0]**2 + xs[1]**2)
    calA = delta_i*a(t0)**1.5*np.sqrt(wi*tau)
    Ai, _, Bi, _ = airy(np.array(s_eval, float))
    env_pred = np.sqrt(np.pi)*calA*np.sqrt(Ai**2 + Bi**2)
    c_pred = (mt*tau)**0.5*(1-np.exp(-3*dN))**0.25
    res = 2*mt2*(A0/Ath)**2/(m*H0)
    return dN, Hc, tau, env_num/env_pred, c_pred, res

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    cases = [(p, name, A0) for p, name in [(0.5, "RD"), (2/3, "MD")] for A0 in [10., 40., 160.]]
    with ProcessPoolExecutor(8) as ex, ThreadPoolExecutor(6) as tx:
        outs = list(tx.map(lambda c: run(c[2], c[0], ex=ex), cases))
        for (p, name, A0), (dN, Hc, tau, r, c, res) in zip(cases, outs):
                print(f"{name} A0={A0:5}: dN={dN:.2f} H_c={Hc:.2e} H_c*tau={Hc*tau:.3f}  numeric/Airy at s=-3,-2,2,3: "
                      + " ".join(f"{x:.3f}" for x in r) + f"   c_pred={(mt/(3*Hc))**(1/6):.2f} (exact {c:.2f})  no-res ratio={res:.2f}", flush=True)
