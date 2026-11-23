"""Dephasing test: ratio delta(t_c + 2 tau)/delta_i vs delta_i for the full nonlinear clean model (RD),
compared with the linear limit (delta_i = 1e-7). Linear theory => constant ratio. Parameters of isocurvature.py."""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from isocurvature import deviation_at_T1, mt, Hc
if __name__ == "__main__":
    ds = [1e-7, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2]
    with ProcessPoolExecutor(8) as ex: r = [x/d for x, d in zip(ex.map(deviation_at_T1, ds), ds)]
    print(f"m~/H_c = {mt/Hc:.0f}, linear ratio = {r[0]:.4f}")
    for d, x in zip(ds[1:], r[1:]): print(f"delta_i={d:7.0e}  ratio={x:9.4f}   rel. diff from linear = {abs(x/r[0]-1):.2e}")
