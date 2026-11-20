"""Hilltop stabilization in different backgrounds: de Sitter (H const),
radiation (a ~ t^{1/2}) and matter (a ~ t^{2/3}) domination.
Model 'massive' (clean Kapitza):  V = V0(1-cos phi) - kap chi cos phi + m^2 chi^2/2
Model 'spin'   (Sec. III A):      V = V0(1-cos phi) - b w cos(chi/g) cos phi
Time variable: N = ln a; we integrate in t with H(t) = H0 (dS) or p/t."""
import numpy as np
from scipy.integrate import solve_ivp

BG = {"dS": None, "RD": 0.5, "MD": 2/3}

def Hfun(bg, H0):
    p = BG[bg]
    if p is None: return (lambda t: H0), 0.0, (lambda t: H0*t)
    t0 = p/H0
    return (lambda t: p/(t0+t)), 0.0, (lambda t: p*np.log((t0+t)/t0))

def run(model, bg, H0, T, y0, prm, npts=20000, rtol=1e-9, atol=1e-10):
    H, _, Nof = Hfun(bg, H0)
    V0 = prm["V0"]
    if model == "massive":
        kap, m = prm["kap"], prm["m"]
        def f(t, u):
            p, c, dp, dc = u; h = H(t)
            return [dp, dc, -3*h*dp - V0*np.sin(p) - kap*c*np.sin(p),
                    -3*h*dc + kap*np.cos(p) - m*m*c]
    else:
        b, w, g = prm["b"], prm["w"], prm["g"]
        def f(t, u):
            p, c, dp, dc = u; h = H(t); B = b*w*np.cos(c/g)
            return [dp, dc, -3*h*dp - (V0+B)*np.sin(p),
                    -3*h*dc - (b*w/g)*np.sin(c/g)*np.cos(p)]
    ts = np.linspace(0, T, npts)
    s = solve_ivp(f, [0, T], y0, t_eval=ts, rtol=rtol, atol=atol, method="DOP853")
    return s.t, Nof(s.t), s.y

def release_N(N, phi, thr=1.0):
    d = np.abs(phi-np.pi); i = np.argmax(d > thr)
    return N[i] if d.max() > thr else np.nan

if __name__ == "__main__":
    # --- massive model: kap/m = 0.5 (no static sign flip), A0 = 50
    prm = dict(V0=1., kap=50., m=100.)
    A0 = 50.; chibar = -prm["kap"]/prm["m"]**2
    Ath = np.sqrt(2*prm["m"]**2*(prm["V0"]+prm["kap"]*chibar))/prm["kap"]
    Npred = (2/3)*np.log(A0/Ath)
    print(f"[massive] threshold A={Ath:.3f}, predicted release N={Npred:.3f}")
    for bg in ["dS", "RD", "MD"]:
        H0 = 0.1
        p = BG[bg]; Nend = Npred + 0.6
        T = Nend/H0 if p is None else (p/H0)*(np.exp(Nend/p)-1)
        t, N, y = run("massive", bg, H0, T, [np.pi-1.0, A0+chibar, 0, 0], prm, npts=60000)
        print(f"  {bg}: release N = {release_N(N, y[0]):.3f}   (t_release = {np.interp(release_N(N,y[0]),N,t):.1f})")
    # --- spin model (Sec III A), Omega0 = 100: predicted N_- = ln(100/20)/3
    prm2 = dict(V0=1., b=2., w=100., g=100.)
    print(f"[spin] predicted release N={np.log(100/np.sqrt(400))/3:.3f}")
    for bg in ["dS", "RD", "MD"]:
        H0 = 0.01; p = BG[bg]; Nend = 0.9
        T = Nend/H0 if p is None else (p/H0)*(np.exp(Nend/p)-1)
        t, N, y = run("spin", bg, H0, T, [0.9*np.pi, 0, 0, 100*prm2["g"]], prm2, npts=40000)
        print(f"  {bg}: release N = {release_N(N, y[0]):.3f}")
