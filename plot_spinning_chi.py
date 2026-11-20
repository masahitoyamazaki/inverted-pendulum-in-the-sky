"""Remake Figs. CoupledAxionTE/SH: phi vs e-folds for the spinning-chi model
V = V0(1-cos phi) - b w cos(chi/g) cos phi, V0=1, b=2, w=100, g=100.
Shaded band: predicted stabilized phase Omega_- < Omega(t) < Omega_+,
Omega_+ = b w/sqrt(2 V0), Omega_- = sqrt(2 b w)."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coupled_axions import evolve
plt.rcParams.update({"font.size": 9})
V0, b, w, g = 1., 2., 100., 100.
Op, Om = b*w/np.sqrt(2*V0), np.sqrt(2*b*w)
def make(H, Omega0, T, out):
    t, y = evolve(0.9*np.pi, 0., 0., Omega0*g, T, V0=V0, b=b, w=w, g=g, H=H,
                  npts=8000, rtol=1e-9, atol=1e-10)
    N = H*t
    Nstart = max(0., np.log(Omega0/Op)/(3*H)*H); Nend = np.log(Omega0/Om)/3
    f, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axvspan(Nstart, Nend, color="#2ca02c", alpha=0.15, lw=0)
    ax.axhline(np.pi, color="gray", lw=0.6, ls=":")
    ax.plot(N, y[0], color="#1f77b4", lw=0.8)
    ax.set_xlabel(r"$N = Ht$"); ax.set_ylabel(r"$\phi$")
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlim(0, N[-1]); ax.set_ylim(-0.2, 2*np.pi+0.2)
    f.tight_layout(); f.savefig(out)
    dev = np.abs(y[0]-np.pi); i = np.argmax(dev > 1.0)
    print(out, "window N=[%.3f, %.3f]" % (Nstart, Nend), "leaves at N=%.3f" % N[i])
outdir = sys.argv[1] if len(sys.argv) > 1 else "."
which = sys.argv[2] if len(sys.argv) > 2 else "both"
if which in ("SH", "both"): make(0.01, 1e2, 80., f"{outdir}/CoupledAxionSH2.pdf")
if which in ("TE", "both"): make(1.0, 1e6, 5.0, f"{outdir}/CoupledAxionTE2.pdf")
