"""Floquet growth rate of delta-phi at phi=pi vs chi(0) (Fig. FloquetChi)."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from floquet_chi import multiplier
plt.rcParams.update({"font.size": 9})
c0 = np.linspace(0.02, 4*np.pi-0.02, 400)
res = [multiplier(c) for c in c0]
mu = np.array([np.log(m)/T if T else np.nan for T, m in res])
stable = c0[mu < 1e-6]; print("linearly stable window: chi0 in [%.3f, %.3f]" % (stable.min(), stable.max()))
f, ax = plt.subplots(figsize=(3.4, 2.3))
ax.axvspan(stable.min(), stable.max(), color="#2ca02c", alpha=0.15, lw=0)
ax.plot(c0, mu, color="#1f77b4", lw=1)
ax.axvline(3.5, color="#d62728", lw=0.6, ls="--")
ax.set_xlabel(r"$\chi(0)$"); ax.set_ylabel(r"Floquet exponent $\mu$")
ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
ax.set_xlim(0, 4*np.pi); ax.set_ylim(0, None)
f.tight_layout(); f.savefig(sys.argv[1] if len(sys.argv) > 1 else "FloquetChi.pdf")
