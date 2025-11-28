import sympy as sp
import os
import numpy as np
import matplotlib.pyplot as plt
from New_Need_Functions import Bistability_with_K,Bistability_with_K_try
import time
start_time = time.time()
P=np.linspace(0,1,101)
f=np.ones(len(P))*8.184e9
para = {'omega_a': 8.246e9,
        'omega_m': 8.184e9,
        'kaint': 3.39e6,
        'kaed': 2.974e6,
        'kmint': 1.011e6,
        'kmext': 0,
        'g_ma': 32.649e6,
        'K': 30e-9,
        'branch': 'upper',
        'omega_d': f,
        'P_d': P,
        }
para1 = {'omega_a': 8.246e9,
        'omega_m': 8.184e9,
        'kaint': 3.39e6,
        'kaed': 2.974e6,
        'kmint': 1.011e6,
        'kmext': 0,
        'g_ma': 32.649e6,
        'K': 30e-9,
        'branch': 'upper',
        'omega_d': f,
        'P_d': P[::-1],
        }
power,delta,wd,a_s,m_s=Bistability_with_K_try(**para).BS_array_with_as_and_ms(start_energy='lower')
power1,delta1,wd1,a_s1,m_s1=Bistability_with_K_try(**para1).BS_array_with_as_and_ms(start_energy='lower')

plt.figure(figsize=(12, 6))
axes1 = plt.subplot(121)
axes1.plot(power,delta, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes1.plot(power1, delta1, '^', linewidth=5, color='blue',label=r'backward')
# axes1.plot(wd,delta, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(wd1, delta1, '^', linewidth=5, color='blue',label=r'backward')
axes1.set_xlabel(r'$f_d$ ', fontsize=20)
axes1.set_ylabel(r'$\Delta_m$ [MHz]', fontsize=20)
plt.tick_params(labelsize=20)
plt.legend(loc=0)
plt.show()
