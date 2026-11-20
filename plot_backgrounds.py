"""Fig. Backgrounds: phi vs N in dS / RD / MD for the clean Kapitza model
V = V0(1-cos phi) - kap chi cos phi + m^2 chi^2/2,  V0=1, kap=50, m=100, A0=50, H0=0.1."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backgrounds import run, BG
plt.rcParams.update({"font.size": 9})
prm = dict(V0=1., kap=50., m=100.); A0 = 50.; chibar = -prm["kap"]/prm["m"]**2
V0n = prm["V0"] + prm["kap"]*chibar
Ath = np.sqrt(2*prm["m"]**2*V0n)/prm["kap"]; Npred = (2/3)*np.log(A0/Ath)
def weff(N):
    A = A0*np.exp(-1.5*N); return np.sqrt(np.maximum(prm["kap"]**2*A*A/(2*prm["m"]**2) - V0n, 1e-12))
f, ax = plt.subplots(figsize=(3.4, 2.5))
cols = {"dS": "#1f77b4", "RD": "#ff7f0e", "MD": "#2ca02c"}
for bg in ["dS", "RD", "MD"]:
    H0 = 0.1; p = BG[bg]; Nend = 3.2
    T = Nend/H0 if p is None else (p/H0)*(np.exp(Nend/p)-1)
    t, N, y = run("massive", bg, H0, T, [np.pi-0.5, A0+chibar, 0, 0], prm, npts=120000)
    ax.plot(N, y[0], color=cols[bg], lw=0.7, label={"dS": "de Sitter", "RD": "radiation dom.", "MD": "matter dom."}[bg])
Ns = np.linspace(0.05, Npred-0.05, 200)
env = 0.5*np.exp(-1.5*Ns)*np.sqrt(weff(0)/weff(Ns))
ax.plot(Ns, np.pi+env, "k--", lw=0.6); ax.plot(Ns, np.pi-env, "k--", lw=0.6)
ax.axvline(Npred, color="gray", ls=":", lw=0.8)
ax.set_xlabel(r"$N=\ln a$"); ax.set_ylabel(r"$\phi$")
ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_xlim(0, 3.2); ax.set_ylim(-0.2, 2*np.pi+0.2)
ax.legend(fontsize=6.5, loc="lower left", frameon=False)
f.tight_layout(); f.savefig(sys.argv[1] if len(sys.argv) > 1 else "Backgrounds.pdf")
print("Npred", Npred)
