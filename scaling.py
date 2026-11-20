"""General scaling of the hilltop attractor (averaged slow-mode equation).
 d'' + 3H d' + mt^2 (K(a) - 1) d = 0,  K = K_c (a_c/a)^{2 sigma} with K_c = 1 at the end of the stabilized phase a_c,
 H = H_c (a_c/a)^gamma.  We take H_c = 2 mt (field frozen after the release) and measure the envelope of delta at a_c
 (over initial phases), compared with  delta_*/delta_i = Q_i^{-(3-sigma)/(2(sigma-gamma))},  Q_i = omega_eff,i/H_i."""
import numpy as np
from scipy.integrate import solve_ivp
mt = 1.0
def run(sigma, gamma, Qi, Hc=2.0):
    # choose a_i such that omega_i/H_i = Qi with omega ~ mt sqrt(K)
    # omega/H = (mt/Hc) (a_c/a)^{sigma-gamma}  ->  (a_c/a_i) = (Qi Hc/mt)^{1/(sigma-gamma)}
    r = (Qi*Hc/mt)**(1/(sigma-gamma)); Ni = -np.log(r)   # N measured from a_c (N=0 at release)
    H = lambda N: Hc*np.exp(-gamma*N)
    w2 = lambda N: mt**2*(np.exp(-2*sigma*N) - 1)
    # integrate in N: d'' (in t) -> use variables (d, dd/dt), dN/dt = H
    def f(N, u):
        d, v = u; h = H(N)
        return [v/h, (-3*h*v - w2(N)*d)/h]
    w0 = np.sqrt(w2(Ni))
    ends = [solve_ivp(f, [Ni, 0.0], y0, rtol=1e-10, atol=1e-13, method="LSODA").y[0, -1] for y0 in ([1., 0.], [0., w0])]
    return np.hypot(*ends), Qi**(-(3-sigma)/(2*(sigma-gamma)))
if __name__ == "__main__":
    print(" sigma gamma   Q_i   numerical   predicted")
    for sigma, gamma in [(1.5, 0.), (2.5, 0.), (2.5, 2.), (2.75, 2.), (2.5, 1.5)]:
        for Qi in [10., 30., 100.]:
            n, p = run(sigma, gamma, Qi)
            print(f"{sigma:5} {gamma:5} {Qi:6.0f}   {n:9.4f}   {p:9.4f}")
