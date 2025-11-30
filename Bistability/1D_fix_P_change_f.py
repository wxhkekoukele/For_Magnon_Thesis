import sympy as sp
import os
import numpy as np
import matplotlib.pyplot as plt
from New_Need_Functions import Bistability_with_K,Bistability_with_K_evo
import time
start_time = time.time()
f1=np.linspace(8.1,8.3,101)*1e9
P1=np.ones(len(f1))*100e-3

f2=np.linspace(8.1,8.3,101)*1e9
P2=np.ones(len(f2))*100e-3
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
para1 = {'omega_a': 8.246e9,
        'omega_m': 8.184e9,
        'kaint': 3.39e6,
        'kaed': 2.974e6,
        'kmint': 1.011e6,
        'kmext': 0,
        'g_ma': 32.649e6,
        'K': 30e-9,
        'branch': 'upper',
        'omega_d': f2[::-1],
        'P_d': P2,
        }

para2 = {'omega_a': 8.246e9,
        'omega_m': 8.184e9,
        'kaint': 3.39e6,
        'kaed': 2.974e6,
        'kmint': 1.011e6,
        'kmext': 0,
        'g_ma': 32.649e6,
        'K': 30e-9,
        'branch': 'upper',
        'omega_d':f2,
        'P_d': 100e-3,
        }

df,dfp,db,dbp,du,dup=Bistability_with_K(**para2).BS_fre(100e-3)
m_sf, a_sf, m_sb, a_sb, m_su, a_su, forward, forwardp, backward, backwardp, unstable, unstablep=Bistability_with_K(**para2).Compute_ms_and_as_fre()

power,delta,wd,a_s,m_s=Bistability_with_K_evo(**para).BS_array_with_as_and_ms(start_energy='higher')
power1,delta1,wd1,a_s1,m_s1=Bistability_with_K_evo(**para1).BS_array_with_as_and_ms(start_energy='lower')

plt.figure(figsize=(12, 6))
axes1 = plt.subplot(121)
# axes1.plot(power,delta, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(power1, delta1, '^', linewidth=5, color='blue',label=r'backward')
axes1.plot(wd,np.array(delta)/1e6, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes1.plot(wd1, np.array(delta1)/1e6, '^', linewidth=5, color='blue',label=r'backward')
axes1 = plt.subplot(122)
axes1.plot(dfp,df/1e6, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes1.plot(dbp, db/1e6, '^', linewidth=5, color='blue',label=r'backward')

plt.legend(loc=0)
plt.show()

# plt.figure(figsize=(12, 6))
# axes1 = plt.subplot(121)
# axes1.plot(np.array(m_s).real,np.array(m_s).imag, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(np.array(m_s1).real,np.array(m_s1).imag, '^', linewidth=5, color='blue',label=r'backward')
# # axes1.plot(wd,delta, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# # axes1.plot(wd1, delta1, '^', linewidth=5, color='blue',label=r'backward')
# axes1 = plt.subplot(122)
# axes1.plot(m_sf.real,m_sf.imag, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(m_sb.real, m_sb.imag, '^', linewidth=5, color='blue',label=r'backward')
# plt.legend(loc=0)
# plt.show()

plt.figure(figsize=(12, 6))
axes1 = plt.subplot(121)
axes1.plot(wd,np.array(delta)/1e6, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes1.plot(wd1, np.array(delta1)/1e6, '^', linewidth=5, color='blue',label=r'backward')
# axes1.plot(wd,delta, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(wd1, delta1, '^', linewidth=5, color='blue',label=r'backward')
axes1 = plt.subplot(122)
axes1.plot(m_sf.real,m_sf.imag, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
axes1.plot(m_sb.real, m_sb.imag, '^', linewidth=5, color='blue',label=r'backward')
plt.legend(loc=0)
plt.show()


