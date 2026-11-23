"""Fig. Delayed: late-time number density relative to standard misalignment (theta_i=1)
vs H_c/m~ for the delayed release in RD (delayed_fast.py: m_chi=3e4, kap=1.5e4, H0=3,
which satisfy the no-resonance condition for all points). Saves data to delayed_data.txt."""
import sys, os, numpy as np
from concurrent.futures import ProcessPoolExecutor
from delayed_fast import delayed, standard
xs = [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003]
if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "Delayed.pdf"
    if not os.path.exists("delayed_data.txt"):
        with ProcessPoolExecutor() as ex:
            f = ex.submit(standard); dl = list(ex.map(delayed, xs)); n0 = f.result()
        np.savetxt("delayed_data.txt", [(x, n/n0) for x, n, r in dl], header="H_c/m~   n/n_std(theta_i=1)")
    d = np.loadtxt("delayed_data.txt")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    f, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.loglog(d[:, 0], d[:, 1], "o", ms=4, color="#1f77b4", label="numerical")
    xx = np.logspace(np.log10(d[:, 0].min())-0.1, np.log10(d[:, 0].max())+0.1, 50)
    ax.loglog(xx, 2.5*xx**-1.5, "--", color="gray", lw=0.8, label=r"$2.5\,(\tilde m/H_c)^{3/2}$")
    ax.set_xlabel(r"$H_c/\tilde m$"); ax.set_ylabel(r"$n_\phi/n_{\phi,\rm std}(\theta_i=1)$")
    ax.legend(fontsize=7, frameon=False)
    f.tight_layout(); f.savefig(out)
    print(d)
