import sympy as sp
import os
import numpy as np
import matplotlib.pyplot as plt
from New_Need_Functions import Bistability_with_K,Bistability_with_K_evo
f1=np.linspace(8.205,8.208,301)*1e9
P1=np.linspace(0,300e-3,301)

para = {'omega_a': 8.246e9,
        'omega_m': 8.184e9,
        'kaint': 3.39e6,
        'kaed': 2.974e6,
        'kmint': 1.011e6,
        'kmext': 0,
        'g_ma': 32.649e6,
        'K': 30e-9,
        'branch': 'upper',
        'omega_d': f1,
        'P_d': P1,
        }
a_s,m_s,delta, Time,power,wd=Bistability_with_K_evo(**para).m_a_evolution(100e-3,8.20669e9,100e-3,8.20670e9,1e-12,1e6,start_energy='higher')

# print(delta)
plt.figure(figsize=(12, 6))
axes1 = plt.subplot(111)
axes1.plot(Time,delta, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(wd,delta, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(Time,np.abs(m_s)**2, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(wd1, delta1, '^', linewidth=5, color='blue',label=r'backward')
axes1.set_xlabel(r'$f_d$ ', fontsize=20)
axes1.set_ylabel(r'$\Delta_m$ [MHz]', fontsize=20)
plt.tick_params(labelsize=20)
plt.legend(loc=0)
plt.show()