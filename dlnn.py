"""d ln n / d ln delta_i at several delta_i (delayed-release regime, RD, model of isocurvature.py),
by central differences with two step sizes eps to check convergence."""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from isocurvature import run
ds = [0.05, 0.1, 0.2, 0.4, 0.8]; epss = [1e-3, 1e-4]
jobs = [(d, e, sgn) for d in ds for e in epss for sgn in (+1, -1)]
def job(x):
    d, e, sgn = x
    return x, run(d*(1+sgn*e))[1]
if __name__ == "__main__":
    with ProcessPoolExecutor() as ex:
        base = dict(zip(ds, [r[1] for r in ex.map(run, ds)]))
        res = dict(ex.map(job, jobs))
    print("delta_i   n/n(0.05)   dln n/dln delta_i (eps=1e-3)   (eps=1e-4)")
    for d in ds:
        sl = [(np.log(res[(d, e, 1)]) - np.log(res[(d, e, -1)]))/(2*e) for e in epss]
        print(f"{d:6.2f}   {base[d]/base[0.05]:8.4f}   {sl[0]:14.4f}   {sl[1]:14.4f}")
