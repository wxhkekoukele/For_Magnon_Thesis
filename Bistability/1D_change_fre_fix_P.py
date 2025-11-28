import sympy as sp
import os
import numpy as np
import matplotlib.pyplot as plt
from New_Need_Functions import Bistability_with_K,Bistability_with_K_try
import time
start_time = time.time()
P_amp=50e-3
f = np.linspace(8.1e9, 8.3e9, 201)
P=np.ones(len(f))*P_amp

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
        'P_d': P_amp,
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
        'P_d': P,
        }

m_sf, a_sf, m_sb, a_sb, m_su, a_su, forward, forwardf, backward, backwardf, unstable, unstablef=Bistability_with_K(**para).Compute_ms_and_as_fre()
m_sf1, a_sf1, m_sb1, a_sb1, m_su1, a_su1, forward1, forwardp1,forwardf1, backward1, backwardp1,backwardf1, unstable1,unstablep1, unstablef1=Bistability_with_K_try(**para1).Bs_with_ms_and_as()
# forward1, forwardp1,forwardf1, backward1, backwardp1,backwardf1, unstable1,unstablep1, unstablef1=Bistability_with_K_try(**para1).BS()
# print(backwardf)
# print(backwardf1)

stop_time = time.time()
print(stop_time-start_time)

plt.figure(figsize=(12, 6))
axes1 = plt.subplot(121)
axes1.plot(forwardf/(2*np.pi),forward, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes1.plot(backwardf/(2*np.pi), backward, '^', linewidth=5, color='blue',label=r'backward')
axes1.plot(unstablef/(2*np.pi),unstable, '*', linewidth=5, color='purple',label=r'unstable')

axes2 = plt.subplot(122)
axes2.plot(forwardf1/(2*np.pi),forward1, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes2.plot(backwardf1/(2*np.pi), backward1, '^', linewidth=5, color='blue',label=r'backward')
axes2.plot(unstablef1/(2*np.pi),unstable1, '*', linewidth=5, color='purple',label=r'unstable')
axes1.set_xlabel(r'$f_d$ ', fontsize=20)
axes1.set_ylabel(r'$\Delta_m$ [MHz]', fontsize=20)
plt.tick_params(labelsize=20)
plt.legend(loc=0)
plt.show()


plt.figure(figsize=(12, 6))
axes1 = plt.subplot(121)
# axes1.plot(f,m_sf.real, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(f, m_sb.real, '^', linewidth=5, color='blue',label=r'backward')
# axes1.plot(unstablef/(2*np.pi),m_su.real, '--', linewidth=5, color='purple',label=r'unstable')

# axes1.plot(f,m_sf.imag, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(f, m_sb.imag, '^', linewidth=5, color='blue',label=r'backward')
# axes1.plot(unstablef/(2*np.pi),m_su.imag, '--', linewidth=5, color='purple',label=r'unstable')

axes1.plot(m_sf.real,m_sf.imag, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes1.plot(m_sb.real, m_sb.imag, '^', linewidth=5, color='blue',label=r'backward')
axes1.plot(m_su.real,m_su.imag, '*', linewidth=5, color='purple',label=r'unstable')
axes1.scatter(m_sf.real[0],m_sf.imag[0], marker='x', s=500, color='red',label=r'start',zorder=1)
axes1.scatter(m_sf.real[-1],m_sf.imag[-1], marker='+', s=250, color='red',label=r'stop',zorder=2)

axes2 = plt.subplot(122)
axes2.plot(m_sf1.real,m_sf1.imag, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes2.plot(m_sb1.real, m_sb1.imag, '^', linewidth=5, color='blue',label=r'backward')
axes2.plot(m_su1.real,m_su1.imag, '*', linewidth=5, color='purple',label=r'unstable')
axes1.set_xlabel(r'$f_d$ ', fontsize=20)
axes1.set_ylabel(r'$\Delta_m$ [MHz]', fontsize=20)
plt.tick_params(labelsize=20)
plt.legend(loc=0)
plt.show()
