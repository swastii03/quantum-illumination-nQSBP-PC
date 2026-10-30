#Entanglement entropy for nQSBP BiPC and nQSBP SiPC
import numpy as np
import matplotlib.pyplot as plt

Ns0= 0.05                   
lam= np.tanh(np.arcsinh(np.sqrt(Ns0)))
g0, n=1.5, 2             

def f(j, T):
    return T**((j-1)/2)*(T-j*(1-T))

def entropy(p):
    p= p/p.sum(axis=0)
    p= np.clip(p, 1e-300, None)
    return -(p*np.log2(p)).sum(axis=0)

jj= np.arange(200)
pt= lam**(2*jj)
pt/=pt.sum()
E_tmsv = -(pt*np.log2(pt)).sum()
N=600
x =np.linspace(0.003, 0.997, N)
T1,T2= np.meshgrid(x, x)

#biside
p_bi= np.array([lam**(2*j) * f(j, T1)**2 * f(j, T2)**2 * g0**(4*j) for j in range(n + 1)])
rE_bi= entropy(p_bi)/E_tmsv-1

#single side
g= np.sqrt(T2/(1 - T2))
p_si=np.array([lam**(2*j)*f(j, T1)**2 * g**(2*j) for j in range(n + 1)])
rE_si=entropy(p_si)/E_tmsv-1

def heat(rE, title, ylab, fname):
    fig, ax= plt.subplots(figsize=(5.2, 4.6))
    m=ax.pcolormesh(T1, T2, rE, cmap='RdBu_r', shading='auto')
    cb= fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$r_E$', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r'$T_1=\cos^2\theta_1$', fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.show()
    plt.close(fig)

heat(rE_bi, 'Entanglement Enhancement for nQSBP Biside PC-TMSV',
     r'$T_2=\cos^2\theta_2$','fig9EntanglementBiside.png')
heat(rE_si, 'Entanglement Enhancement for nQSBP Single-side PC-TMSV',
     r'$\tau=\cos^2\theta_2$','fig10EntanglementSingle.png')

tau0=g0**2/(1 + g0**2)
k, kt= np.argmin(abs(x-0.2)), np.argmin(abs(x - tau0))
print('biside  E ratio at T1=T2=0.2 :', round(1 + rE_bi[k, k], 2))
print('single  E ratio at T1=0.2, g=1.5:', round(1 + rE_si[kt, k], 2))
print('max r_E: biside %.2f, single-side %.2f' % (rE_bi.max(), rE_si.max()))
