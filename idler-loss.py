import numpy as np
from qi_core import *

ch=default()
kap,NB=ch.kappa, ch.NB
etas=[1.0, 0.8, 0.4, 0.1]
cfg=[('Single side PC', 1, False), ('Biside PC', 2, False),
       ('nQSBP single side', 1, True), ('nQSBP biside', 2, True)]
c0=tmsv_amplitudes(lam)
coh=lambda N: kap*N*(np.sqrt(NB+ 1)-np.sqrt(NB))**2

print('Exponent ratios vs idler transm.(equal lambda, equal N_S, xi/xi_coh, rate vs TMSV)')
for eta in etas:
    x0=ch.xi(c0, eta)
    print('eta_I=%.1f TMSV xi/xi_coh = %.2f' % (eta, x0/coh(mean_photons(c0))))
    for name, m, amp in cfg:
        c=amplitudes(0.2, m, amp)
        x=ch.xi(c, eta)
        xe=ch.xi(equal_energy_tmsv(c), eta)
        P= success(0.2, m, amp)
        print(' %-20s %7.3f %9.3f %9.3f %11.3e' % (name, x/x0, x/xe, x/coh(mean_photons(c)), P*x/x0))

print('\nnQSBP on the lossy idler of a TMSV(heralded exp./unamplified+per attempt)')
for gain, order in [(3, 1), (1.5, 2)]:
    for eta in [1.0, 0.1, 0.01]:
        x0=ch.xi(c0, eta)
        xp, P=ch.xi_idler_nla(c0, eta, gain, order)
        print('g=%.1f n=%d eta_I=%.2f xi\'/xi0= %.2f P xi\'/xi0 = %.2f P = %.3f' % (gain, order, eta, xp/x0, P*xp/x0, P))
