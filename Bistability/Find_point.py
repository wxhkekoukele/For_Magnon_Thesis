import numpy as np
import matplotlib.pyplot as plt
from New_Need_Functions import Bistability_with_K,Bistability_with_K_evo

drive_fre=np.loadtxt(r'/Users/xiaohanwang/Desktop/cp_try/drive fre.txt')
drive_power=np.loadtxt(r'/Users/xiaohanwang/Desktop/cp_try/drive power.txt')
# print(drive_fre[0])
for i in range(len(drive_fre)):
    if (drive_power[i]==0.05)&(drive_fre[i]==8.15):
        print('A')
        print(i)
    if (drive_power[i]==0)&(drive_fre[i]==8.15):
        print('B')
        print(i)
    if (drive_power[i]==0)&(drive_fre[i]==8.21):
        print('C')
        print(i)
    if (drive_power[i]==0.05)&(drive_fre[i]==8.21):
        print('D')
        print(i)
# plt.figure(figsize=(12, 6))
# axes1 = plt.subplot(111)
# # axes1.plot(Time[::1000],delta[::1000], '-', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(drive_fre,drive_power, '-', linewidth=5,color='orange',markersize=10,label=r'forward')
# axes1.plot(drive_fre[0],drive_power[0], 'o', linewidth=5,color='green',markersize=30)
#
# # axes1.plot(wd,delta, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# # axes1.plot(Time,np.abs(m_s)**2, 'o', linewidth=5,color='orange',markersize=10,label=r'forward')
# # axes1.plot(wd1, delta1, '^', linewidth=5, color='blue',label=r'backward')
# axes1.set_xlabel(r'$f$ ', fontsize=20)
# axes1.set_ylabel(r'$p$ ', fontsize=20)
# plt.tick_params(labelsize=20)
# plt.legend(loc=0)
# plt.show()

#S(50mW,8.191GHz) 0
#A(50mW,8.15GHz) 41
#B(0mW,8.15GHz) 91
#C(0mW,8.21GHz) 151
#D(50mW,8.21GHz) 201