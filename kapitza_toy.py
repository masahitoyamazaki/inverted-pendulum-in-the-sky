"""Kapitza toy model (Eq. (3)): cos(phi) vs sin(phi) coupling, and Floquet threshold b = sqrt(2 V0) (Sec. II)."""
import numpy as np
from scipy.integrate import solve_ivp
V0=1.0; Om=50.0
def run(b, coup, phi0=np.pi-0.3, T=60, H=0.0):
    # coup='cos': V = V0(1-cos) + b Om sin(Om t) cos(phi);  'sin': ... sin(phi)
    def f(t,y):
        p,v=y
        if coup=='cos': F = -V0*np.sin(p) + b*Om*np.sin(Om*t)*np.sin(p)
        else:           F = -V0*np.sin(p) - b*Om*np.sin(Om*t)*np.cos(p)
        return [v, F-3*H*v]
    s=solve_ivp(f,[0,T],[phi0,0],max_step=0.01,rtol=1e-8,atol=1e-10)
    return np.max(np.abs(s.y[0]-np.pi))
for coup in ['sin','cos']:
    for b in [1.0,1.3,1.5,2.0,3.0]:
        print(coup,b,'max|phi-pi| =',round(run(b,coup),3))
# Floquet multipliers of linearized eq at phi=pi (cos coupling): x'' + (-V0 + b Om sin Om t) x = 0
def floq(b,k2=0.0):
    Tp=2*np.pi/Om
    M=[]
    for y0 in ([1,0],[0,1]):
        s=solve_ivp(lambda t,y:[y[1],-(k2-V0+b*Om*np.sin(Om*t))*y[0]],[0,Tp],y0,rtol=1e-11,atol=1e-12)
        M.append(s.y[:,-1])
    M=np.array(M).T
    return np.max(np.abs(np.linalg.eigvals(M)))
for b in [1.3,1.4,1.42,1.45,1.6]:
    print('b',b,'|floquet mult| max',floq(b))
# k-mode resonance near k ~ Om/2
ks=np.linspace(0.3,0.7,41)*Om
mus=[np.log(floq(2.0,k*k))*Om/(2*np.pi) for k in ks]
i=np.argmax(mus); print('max growth rate (b=2):',mus[i],'at k/Om=',ks[i]/Om)
