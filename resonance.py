"""Parametric resonance of inhomogeneous modes across the first Mathieu band in de Sitter (Eq. (13)).
Compares ln(1+n_k) with (pi/2) b^2/(Omega H)."""
import numpy as np
from scipy.integrate import solve_ivp
V0=1.0
def X(b,Om,H,kstart=0.7,kend=0.3):
    # y'' + (k^2/a^2 - V0 - 9H^2/4 + b Om sin(Om t)) y = 0,  k/a(0)=kstart*Om
    k=kstart*Om; T=np.log(kstart/kend)/H
    w2=lambda t:(k*np.exp(-H*t))**2 - V0 - 2.25*H*H
    def f(t,u):
        yr,yi,vr,vi=u; m=w2(t)+b*Om*np.sin(Om*t)
        return [vr,vi,-m*yr,-m*yi]
    w0=np.sqrt(w2(0))
    u0=[1/np.sqrt(2*w0),0,0,-w0/np.sqrt(2*w0)]   # positive-frequency mode
    s=solve_ivp(f,[0,T],u0,rtol=1e-10,atol=1e-12,method='RK45',dense_output=True)
    # average n over last driving period
    ts=np.linspace(T-2*np.pi/Om,T,50); ns=[]
    for t in ts:
        yr,yi,vr,vi=s.sol(t); w=np.sqrt(w2(t))
        ns.append((vr**2+vi**2 + w*w*(yr**2+yi**2))/(2*w) - 0.5)
    n=np.mean(ns)
    return 0.5*np.arccosh(1+2*max(n,0)), n
if __name__ == "__main__":
    print(" b2/OmH   ln(1+n)   (pi/2) b2/OmH")
    b, Om = 2, 50
    for H in [0.4, 0.1, 0.025, 0.0125, 0.00625]:
        x, n = X(b, Om, H, 0.6, 0.4)
        print(f"{b*b/(Om*H):7.2f} {np.log(1+n):9.3f} {np.pi/2*b*b/(Om*H):12.3f}")
