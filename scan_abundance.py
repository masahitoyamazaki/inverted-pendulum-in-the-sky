"""Scan of the late-time abundance n(delta_i) over 0.05 <= delta_i <= 0.85 (49 values; model of isocurvature.py)."""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from isocurvature import run
ds = list(np.round(np.geomspace(0.05, 0.85, 40), 5)) + list(np.round(np.linspace(0.78, 0.82, 9), 5))
if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: r = list(ex.map(run, ds))
    n0 = r[0][1]
    for d, (d2, n) in sorted(zip(ds, r)): print(f"{d:8.5f}  delta_rel/delta_i={d2/d:9.3f}  n/n(0.05)={n/n0:.4f}")
