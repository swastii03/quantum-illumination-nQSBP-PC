import numpy as np
import matplotlib.pyplot as plt
from math import factorial
#paramet
lam= np.tanh(np.arcsinh(np.sqrt(0.05)))
g, n= 1.5, 2
J= np.arange(80)

def f(j, T):
    return T**((j-1)/2)*(T-j*(1-T))
def coherence(c2):
    return np.sqrt(c2).sum()**2 - 1
pref=(n + 1)*factorial(n)/((n + 1)**n * (1 + g**2)**n)
def pc_state(T, m):
    w=np.array([lam**(2*j)* f(j, T)**(2*m) for j in J])
    return (1 -lam**2)*w.sum(), w/w.sum()
def probe(T, m, amplified):
    P, c2=pc_state(T, m)
    if not amplified:
        return P, coherence(c2)
    q=np.array([c2[j]*g**(2*m*j) for j in range(n+1)])
    return P*pref**m*q.sum(), coherence(q/q.sum())

C_tmsv= coherence(np.array([(1 - lam**2)*lam**(2*j) for j in J]))
Ts=np.arange(0.02, 1.0, 0.02)
styles = [('PC single-side', 1, False, 'C0', 'o', '-'),
          ('PC two-side', 2, False, 'C1', 's', '-'),
          ('nQSBP single-side', 1, True,  'C2', '^', '--'),
          ('nQSBP two-side', 2, True, 'C3', 'v', '--')]

fig, ax = plt.subplots(1, 2, figsize=(8.6, 6.4))
ax[0].axhline(1, color='k', ls='--', lw=1, label='Original TMSV')
for name, m, amp, c, mk, ls in styles:
    S = np.array([probe(T, m, amp) for T in Ts])
    ax[0].plot(Ts, S[:, 1]/C_tmsv, color=c, marker=mk, ms=2.8, ls=ls, lw=1, label=name)
    ax[1].plot(Ts, S[:, 0], color=c, marker=mk, ms=2.8, ls=ls, lw=1, label=name + ' success prob.')

ax[0].set_title(r'Coherence vs transmittance($\lambda$ = 0.218)', fontsize=8)
ax[1].set_title(r'Success prob. vs T($\lambda$ = 0.218)', fontsize=8)
ax[0].set_ylabel(r'$C/C_{\mathrm{TMSV}}$', fontsize=8)
ax[1].set_ylabel('Success prob.', fontsize=8)
for a in ax:
    a.set_xlabel('T', fontsize=8)
    a.set_xlim(0, 1)
    a.grid(True, lw=0.5, alpha=0.6)
    a.tick_params(labelsize=7)
ax[0].set_ylim(0, 3.4)
ax[1].set_yscale('log'); ax[1].set_ylim(1e-4, 8)
ax[0].legend(fontsize=6, loc='upper right', framealpha=0.9)
ax[1].legend(fontsize=6, loc='upper left', framealpha=0.9)
fig.tight_layout()
fig.savefig('fig8CoherenceSuccess.png', dpi=200)
