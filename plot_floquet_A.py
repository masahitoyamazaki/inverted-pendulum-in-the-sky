"""Floquet exponent of delta-phi at phi=pi vs oscillation amplitude A for the clean model
V = V0(1-cos phi) - kap chi cos phi + m^2 chi^2/2, chi = chibar + A cos(m t), H=0."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams.update({"font.size": 9})
V0, k, m = 1., 50., 100.
chibar = -k/m**2; mt2 = V0 + k*chibar
Ath = np.sqrt(2*m*m*mt2)/k; Amax = 0.91*m*m/(2*k)
def mu(A):
    T = 2*np.pi/m
    f = lambda t, u: [u[1], (V0 + k*(chibar + A*np.cos(m*t)))*u[0]]
    M = np.array([solve_ivp(f, [0, T], y0, rtol=1e-11, atol=1e-12).y[:, -1] for y0 in ([1, 0], [0, 1])]).T
    return np.log(np.abs(np.linalg.eigvals(M)).max())/T
As = np.logspace(-1, np.log10(4*Amax), 400)
mus = np.array([mu(A) for A in As])
st = As[mus < 1e-6]; print("stable A in [%.3f, %.2f]; A_th=%.3f A_max~%.2f" % (st.min(), st.max(), Ath, Amax))
f, ax = plt.subplots(figsize=(3.4, 2.3))
ax.axvspan(Ath, Amax, color="#2ca02c", alpha=0.15, lw=0)
ax.semilogx(As, mus/np.sqrt(mt2), color="#1f77b4", lw=1)
ax.axvline(Ath, color="gray", ls=":", lw=0.8); ax.axvline(Amax, color="gray", ls=":", lw=0.8)
ax.text(Ath*1.1, 2.6, r"$A_{\rm th}$", fontsize=8); ax.text(Amax*1.1, 2.6, r"$A_{\rm max}$", fontsize=8)
ax.set_xlabel(r"amplitude $A$"); ax.set_ylabel(r"$\mu/\tilde m$")
ax.set_xlim(As[0], As[-1]); ax.set_ylim(0, 3)
f.tight_layout(); f.savefig(sys.argv[1] if len(sys.argv) > 1 else "FloquetA.pdf")
