"""Make SpinningTop.pdf for the paper (theta vs e-folds)."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spinning_top import solve
out = sys.argv[1] if len(sys.argv) > 1 else "SpinningTop.pdf"
H = 0.05
fig, ax = plt.subplots(figsize=(3.4, 2.7))
cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for p0, c in zip([5, 20, 100, 1000], cols):
    N, th, Nc = solve(p0, H=H, Nmax=2.8, npts=6000)
    ax.plot(N, th, color=c, lw=1, label=rf"$p_\psi(0)={p0}$")
    ax.axvline(Nc, color=c, ls=":", lw=0.8)
N, th, _ = solve(1e-6, H=H, Nmax=2.8, npts=3000)
ax.plot(N, th, color="gray", lw=1, ls="--", label="no spin")
ax.set_xlabel(r"$N = Ht$"); ax.set_ylabel(r"$\vartheta$")
ax.set_xlim(0, 2.8); ax.set_ylim(0, np.pi*1.05)
ax.set_yticks([0, np.pi/2, np.pi]); ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(out)
