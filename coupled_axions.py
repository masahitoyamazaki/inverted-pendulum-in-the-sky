"""Coupled axions:
 V = V0(1-cos phi) - b w cos(chi/g) cos phi + mchi2 (1-cos chi),  constant H."""
import numpy as np
from scipy.integrate import solve_ivp

def rhs(V0, b, w, g, mchi2, H):
    def f(t, u):
        p, c, dp, dc = u
        B = b*w*np.cos(c/g)
        return [dp, dc,
                -3*H*dp - (V0 + B)*np.sin(p),
                -3*H*dc - (b*w/g)*np.sin(c/g)*np.cos(p) - mchi2*np.sin(c)]
    return f

def evolve(phi0, chi0, dphi0, dchi0, T, V0=1., b=2., w=100., g=100., mchi2=0., H=0.,
           npts=4000, rtol=1e-9, atol=1e-10):
    ts = np.linspace(0, T, npts)
    s = solve_ivp(rhs(V0, b, w, g, mchi2, H), [0, T], [phi0, chi0, dphi0, dchi0],
                  t_eval=ts, rtol=rtol, atol=atol, method='DOP853')
    return s.t, s.y

def Delta(phi0, chi0, T=30., **kw):
    t, y = evolve(phi0, chi0, 0., 0., T, npts=3001, **kw)
    return np.sqrt(np.trapz((y[0]-phi0)**2, t)/T)
