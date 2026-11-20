"""Fig. 1: effective potential V_eff = V0(1-cos phi) + (b^2/4) sin^2 phi, V0=1."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9})
p = np.linspace(0, 2*np.pi, 600)
f, ax = plt.subplots(figsize=(3.4, 2.4))
for b2, c in zip([0, 1, 2, 4], ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]):
    ax.plot(p, (1-np.cos(p)) + b2/4*np.sin(p)**2, color=c, lw=1,
            label=rf"$b^2={b2}$" + (r" $(=2V_0)$" if b2 == 2 else ""))
ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$V_{\rm eff}(\phi)$")
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2.6)
ax.legend(fontsize=6.5, loc="lower center", frameon=False)
f.tight_layout(); f.savefig(sys.argv[1] if len(sys.argv) > 1 else "Poteff.pdf")
