"""Inverted spinning top in the sky: homogeneous sigma model on squashed S^3
with V = V0 (1 + cos theta), in de Sitter with constant H.
Squashed S^3 with metric I1(dth^2+sin^2 th dph^2)+I3(dpsi+cos th dph)^2 is the
configuration space of a symmetric rigid body, so we integrate
  dq/dt = 1/2 (0, w) * q,      w = R I^{-1} R^T L      (space frame)
  dL/dt = -V0 n x z - 3 H L,   n = R e3,  cos theta = n.z
which avoids the Euler-angle coordinate singularity at theta = 0."""
import numpy as np
from scipy.integrate import solve_ivp

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def solve(p0, I1=1.0, I3=1.0, V0=1.0, H=0.05, th0=0.05, Nmax=None, npts=4000):
    Iinv = np.diag([1/I1, 1/I1, 1/I3]); zhat = np.array([0, 0, 1.0])
    def f(t, u):
        q, L = u[:4], u[4:]
        R = rot(q); n = R[:, 2]
        w = R @ Iinv @ R.T @ L
        dq = 0.5 * qmul(np.r_[0, w], q)
        dL = -V0 * np.cross(n, zhat) - 3*H*L
        return np.r_[dq, dL]
    q0 = np.array([np.cos(th0/2), np.sin(th0/2), 0, 0])     # tilt by th0 about x
    n0 = rot(q0)[:, 2]
    L0 = p0 * n0                                            # sleeping: L along axis (p_phi = p_psi)
    Nc = np.log(p0/(2*np.sqrt(I1*V0)))/3
    if Nmax is None: Nmax = Nc + 1.5
    T = Nmax/H; ts = np.linspace(0, T, npts)
    s = solve_ivp(f, [0, T], np.r_[q0, L0], t_eval=ts, rtol=1e-9, atol=1e-11, method='LSODA')
    th = np.array([np.arccos(np.clip(rot(s.y[:4, i])[2, 2], -1, 1)) for i in range(s.y.shape[1])])
    return H*s.t, th, Nc

if __name__ == "__main__":
    H = 0.05
    print(" p_psi(0)   N_c(pred)   N_fall(theta>1)   max theta before N_c")
    for p0 in [5, 20, 100, 1000]:
        N, th, Nc = solve(p0, H=H)
        fall = N[np.argmax(th > 1.0)] if np.any(th > 1.0) else np.nan
        pre = th[N < Nc].max() if np.any(N < Nc) else np.nan
        print(f"{p0:8} {Nc:10.3f} {fall:14.3f} {pre:18.3f}")
    # control: no spin
    N, th, _ = solve(1e-6, H=H, Nmax=1.0)
    print("no spin: N_fall =", N[np.argmax(th > 1.0)])
