"""Fig. Backgrounds: phi vs N in dS / RD / MD for the clean Kapitza model
V = V0(1-cos phi) - kap chi cos phi + m^2 chi^2/2,  V0=1, kap=5e3, m=1e4, A0=50, H0=0.1.
These parameters satisfy the no-resonance condition 2 m~^2 e^{3 Delta N} < m H_i (ratio 0.63).
Uses the fast integrator fastkapitza.py."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastkapitza import integrate, NS
plt.rcParams.update({"font.size": 9})
V0, kap, m, A0, H0 = 1., 5e3, 1e4, 50., 0.1
chibar = -kap / m**2; V0n = V0 + kap * chibar
Ath = np.sqrt(2 * m**2 * V0n) / kap; Npred = (2 / 3) * np.log(A0 / Ath)
print(f"A_th={Ath:.3f}, predicted release N={Npred:.3f}, no-resonance ratio={kap**2*A0**2/(m**3*H0):.3f}")
def weff(N):
    A = A0 * np.exp(-1.5 * N); return np.sqrt(np.maximum(kap**2 * A * A / (2 * m**2) - V0n, 1e-12))
def run(bg, Nend=3.2, nout=120000):
    y0 = np.array([np.pi - 0.5, A0 + chibar, 0., 0.])
    if bg == "dS":
        t, y, _ = integrate(y0, 0., Nend / H0, V0, kap, m, 0., H0, NS, nout, 1e300); return H0 * t, y
    p = {"RD": 0.5, "MD": 2 / 3}[bg]; t0 = p / H0
    t, y, _ = integrate(y0, t0, t0 * np.exp(Nend / p), V0, kap, m, p, H0, NS, nout, 1e300)
    return p * np.log(t / t0), y
if __name__ == "__main__":
    f, ax = plt.subplots(figsize=(3.4, 2.5))
    cols = {"dS": "#1f77b4", "RD": "#ff7f0e", "MD": "#2ca02c"}
    for bg in ["dS", "RD", "MD"]:
        N, y = run(bg)
        d = np.abs(y[0] - np.pi); i = np.argmax(d > 1.0)
        print(f"  {bg}: release N = {N[i]:.3f}")
        ax.plot(N, y[0], color=cols[bg], lw=0.7, label={"dS": "de Sitter", "RD": "radiation dom.", "MD": "matter dom."}[bg])
    Ns = np.linspace(0.05, Npred - 0.05, 200)
    env = 0.5 * np.exp(-1.5 * Ns) * np.sqrt(weff(0) / weff(Ns))
    ax.plot(Ns, np.pi + env, "k--", lw=0.6); ax.plot(Ns, np.pi - env, "k--", lw=0.6)
    ax.axvline(Npred, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$N=\ln a$"); ax.set_ylabel(r"$\phi$")
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlim(0, 3.2); ax.set_ylim(-0.2, 2 * np.pi + 0.2)
    ax.legend(fontsize=6.5, loc="lower left", frameon=False)
    f.tight_layout(); f.savefig(sys.argv[1] if len(sys.argv) > 1 else "Backgrounds.pdf")
