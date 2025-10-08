import sympy as sp
import os
import numpy as np
import matplotlib.pyplot as plt
from New_Need_Functions import Bistability_with_K
import time
start_time = time.time()
f = np.linspace(8.14, 8.22, 201)*1e9

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
        'P_d': 0.1,
        }

m_sf, a_sf, m_sb, a_sb, m_su, a_su, forward, forwardf, backward, backwardf, unstable, unstablef=Bistability_with_K(**para).Compute_ms_and_as_fre()

run_time = time.time() - start_time
print(f'Run time is {run_time}s.')


plt.figure(figsize=(7, 6))
axes1 = plt.subplot(111)
# axes1.plot(f,m_sf.real, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(f, m_sb.real, '^', linewidth=5, color='blue',label=r'backward')
# axes1.plot(unstablef/(2*np.pi),m_su.real, '--', linewidth=5, color='purple',label=r'unstable')

# axes1.plot(f,m_sf.imag, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(f, m_sb.imag, '^', linewidth=5, color='blue',label=r'backward')
# axes1.plot(unstablef/(2*np.pi),m_su.imag, '--', linewidth=5, color='purple',label=r'unstable')

axes1.plot(m_sf.real,m_sf.imag, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes1.plot(m_sb.real, m_sb.imag, '^', linewidth=5, color='blue',label=r'backward')
axes1.plot(m_su.real,m_su.imag, '--', linewidth=5, color='purple',label=r'unstable')
axes1.set_xlabel(r'$f_d$ ', fontsize=20)
axes1.set_ylabel(r'$\Delta_m$ [MHz]', fontsize=20)
plt.tick_params(labelsize=20)
plt.legend(loc=0)
plt.show()
