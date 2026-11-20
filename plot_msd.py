"""Remake Figs. MSDforPhiI20 / MSDforChI20: Delta_phi(T=30) vs phi(0), chi(0).
Parameters: V0=1, b=2, w=2, g=2, m_chi^2=3.4, H=0, zero initial velocities."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coupled_axions import Delta
outdir = sys.argv[1] if len(sys.argv) > 1 else "."
kw = dict(V0=1., b=2., w=2., g=2., mchi2=3.4, H=0.)
plt.rcParams.update({"font.size": 9})
def fig(xs, ys, xlabel, xticks, xticklabels, out):
    f, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(xs, ys, ".", ms=2.5, color="#1f77b4")
    ax.set_xlabel(xlabel); ax.set_ylabel(r"$\Delta_\phi(T=30)$")
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(0, None)
    f.tight_layout(); f.savefig(out)
p0 = np.linspace(0, 2*np.pi, 241)
d = [Delta(p, 3.5, **kw) for p in p0]
fig(p0, d, r"$\phi(0)$", [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"], f"{outdir}/MSDforPhi.pdf")
c0 = np.linspace(0, 8*np.pi, 401)
d = [Delta(0.9*np.pi, c, **kw) for c in c0]
fig(c0, d, r"$\chi(0)$", [0, 2*np.pi, 4*np.pi, 6*np.pi, 8*np.pi],
    ["0", r"$2\pi$", r"$4\pi$", r"$6\pi$", r"$8\pi$"], f"{outdir}/MSDforChi.pdf")
