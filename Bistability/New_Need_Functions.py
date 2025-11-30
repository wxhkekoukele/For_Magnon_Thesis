import sympy as sp
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import time
import numba

hbar = 6.626e-34 / (2 * np.pi)


def Solve_Cubic_Equation(A, B, C, D):
    x = sp.Symbol('x')
    f = ((A + x) ** 2 + B ** 2) * x - C * D
    t = sp.solve(f, x)
    return t


class Bistability_with_K():  ##统一单位全部为Hz，功率为W
    def __init__(self, **kwargs):
        self.omega_a = kwargs.get('omega_a', 0) *(2*np.pi)  # cavity frequency
        self.omega_m = kwargs.get('omega_m', 0) *(2*np.pi)  # initial magnon frequency
        self.kaint = kwargs.get('kaint', 0) *(2*np.pi)  # cavity internal dissipation
        self.kaed = kwargs.get('kaed', 0) *(2*np.pi)  # cavity drive dissipation
        self.kaep = kwargs.get('kaep', 0) *(2*np.pi)  # cavity probe dissipation
        self.kmint = kwargs.get('kmint', 0) *(2*np.pi)  # magnon internal dissipation
        self.kmed = kwargs.get('kmext', 0) *(2*np.pi)  # magnon drive dissipation
        self.P_d = np.asarray(kwargs.get('P_d'))  # drive power
        self.omega_d = np.asarray(kwargs.get('omega_d')) *(2*np.pi)  # drive frequency
        self.g_ma = kwargs.get('g_ma', 0) *(2*np.pi)  # coupling strength between cavity and magnon
        self.branch = kwargs.get('branch', 'upper')  # Magnon or upper branch or lower branch
        self.omega_start = self.branch_fre(self.omega_m)  # the start point of simulation or experiment
        self.K = kwargs.get('K', 0) *(2*np.pi)  # Kerr coefficient

        self.ka = self.kaint + self.kaed + self.kaep  # cavity dissipation
        self.km = self.kmint + self.kmed  # magnon dissipation
        self.Delta_am = self.omega_a - self.omega_m  # detuning between cavity and magnon

    def branch_fre(self, omega_m):

        omega_UP = (self.omega_a + omega_m) / 2 + np.sqrt((self.omega_a -omega_m) ** 2 / 4 + self.g_ma ** 2)
        omega_LP = (self.omega_a + omega_m) / 2 - np.sqrt((self.omega_a - omega_m) ** 2 / 4 + self.g_ma ** 2)
        if self.branch == 'lower':
            omega_out = omega_LP
        elif self.branch == 'upper':
            omega_out = omega_UP
        elif (self.branch != 'lower') & (self.branch != 'upper'):
            omega_out = omega_m
        return omega_out

    def wplus_to_wm(self, wplus):
        Wplus = np.array(wplus)
        fenzi = Wplus ** 2 - Wplus * self.omega_a - self.g_ma ** 2
        fenmu = Wplus - self.omega_a
        wm = fenzi / fenmu
        return wm
    
    def Compute_ms_and_as_Delta_plus(self,P,fre,Delta_plus):
        Fre=fre*(2*np.pi)
        delta_a = self.omega_a - Fre
        delta_m = self.omega_m - Fre

        wplus_init = self.branch_fre(self.omega_m)
        wplus_finalf = wplus_init + Delta_plus * (2*np.pi)
        Delta_mf = self.wplus_to_wm(wplus_finalf) - self.omega_m

        fenzif = -1j * self.g_ma * np.sqrt(self.kaed) * np.sqrt(P / (hbar * Fre))
        fenmuf = (1j * delta_a + self.ka / 2) * (1j * (delta_m + Delta_mf) + self.km / 2) + self.g_ma ** 2

        m_s = fenzif / fenmuf
        a_s = ((-1j * self.g_ma * m_s + np.sqrt(self.kaed) * np.sqrt(P / (hbar * Fre))) / (
            (1j * delta_a + self.ka / 2)))

        return m_s,a_s

    def Parameter_definition_power(self):
        delta_a = self.omega_a - self.omega_d
        delta_m = self.omega_m - self.omega_d

        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2
        S_am = S_m + self.g_ma ** 2 / S_a

        Imag = S_am.imag
        Real = S_am.real
        C = 2 * self.K * self.kaed / (hbar * self.omega_d) * np.abs(self.g_ma / S_a) ** 2 + 2 * self.K * self.kmed / (
                hbar * self.omega_d)

        return Real, Imag, C

    def Get_real_solution_power(self, A, B, C):
        x0 = []
        x1 = []
        x2 = []
        y0 = []
        y1 = []
        y2 = []
        for i, D in enumerate(self.P_d):
            t = Solve_Cubic_Equation(A, B, C, D)
            if sp.Abs(sp.im(t[0])) < 1:
                omega_m0 = self.omega_m + float(sp.re(t[0]))
                y0.append(float((self.branch_fre(omega_m0) - self.omega_start) / (2*np.pi)))
                x0.append(round(D, 9))

            if sp.Abs(sp.im(t[1])) < 1:
                omega_m1 = self.omega_m + float(sp.re(t[1]))
                y1.append(float((self.branch_fre(omega_m1) - self.omega_start) / (2*np.pi)))
                x1.append(round(D, 9))

            if sp.Abs(sp.im(t[2])) < 1:
                omega_m2 = self.omega_m + float(sp.re(t[2]))
                y2.append(float((self.branch_fre(omega_m2) - self.omega_start) / (2*np.pi)))
                x2.append(round(D, 9))
        return x0, y0, x1, y1, x2, y2

    def BS_power(self):
        Real, Imag, C = self.Parameter_definition_power()
        p0, s0, p1, s1, p2, s2 = self.Get_real_solution_power(-Imag, Real, C)
        forward = s0.copy()
        forwardp = p0.copy()
        for i in range(len(p0)):
            for j in range(len(p2)):
                if p0[i] == p2[j]:
                    s0[i] = s2[j]
        backward = s0
        backwardp = p0
        unstable = s1
        unstablep = p1
        return np.array(forward), np.array(forwardp), np.array(backward), np.array(backwardp), np.array(
            unstable), np.array(unstablep)

    def Compute_ms_and_as_power(self):
        forward, forwardp, backward, backwardp, unstable, unstablep = self.BS_power()
        m_sf, a_sf=self.Compute_ms_and_as_Delta_plus(forwardp, self.omega_d/(2*np.pi), forward)
        m_sb, a_sb = self.Compute_ms_and_as_Delta_plus(backwardp, self.omega_d / (2 * np.pi), backward)
        m_su, a_su= self.Compute_ms_and_as_Delta_plus(unstablep, self.omega_d / (2 * np.pi), unstable)
        return m_sf, a_sf, m_sb, a_sb, m_su, a_su, forward, forwardp, backward, backwardp, unstable, unstablep
    
    def Parameter_definition_fre(self):
        delta_a = self.omega_a - self.omega_d
        delta_m = self.omega_m - self.omega_d
        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2
        S_am = S_m + self.g_ma ** 2 / S_a
        Imag = S_am.imag
        Real = S_am.real
        C = 2 * self.K / (hbar * self.omega_d) * self.kaed * np.abs(
            self.g_ma / (S_a)) ** 2 + 2 * self.K / (hbar * self.omega_d) * self.kmed
        return Real, Imag, C

    def Get_real_solution_fre(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D):
        x0 = []
        x1 = []
        x2 = []
        y0 = []
        y1 = []
        y2 = []
        for i, f in enumerate(self.omega_d):
            t = Solve_Cubic_Equation(A[i], B[i], C[i], D)
            if sp.Abs(sp.im(t[0])) < 1:
                omega_m0 = self.omega_m + float(sp.re(t[0]))
                y0.append(float((self.branch_fre(omega_m0) - self.omega_start) / (2*np.pi)))
                x0.append(round(f/(2*np.pi), 9))

            if sp.Abs(sp.im(t[1])) < 1:
                omega_m1 = self.omega_m + float(sp.re(t[1]))
                y1.append(float((self.branch_fre(omega_m1) - self.omega_start) / (2*np.pi)))
                x1.append(round(f/(2*np.pi), 9))

            if sp.Abs(sp.im(t[2])) < 1:
                omega_m2 = self.omega_m + float(sp.re(t[2]))
                y2.append(float((self.branch_fre(omega_m2) - self.omega_start) / (2*np.pi)))
                x2.append(round(f/(2*np.pi), 9))
        return x0, y0, x1, y1, x2, y2

    def BS_fre(self,D):
        Real, Imag, C = self.Parameter_definition_fre()
        f0, s0, f1, s1, f2, s2 = self.Get_real_solution_fre(-Imag, -Real, C, D)
        backward = s0.copy()
        backwardf = f0.copy()
        for i in range(len(f0)):
            for j in range(len(f2)):
                if f0[i] == f2[j]:
                    s0[i] = s2[j]
        forward = s0
        forwardf = f0
        unstable = s1
        unstablef = f1
        return np.array(forward), np.array(forwardf), np.array(backward), np.array(backwardf), np.array(
            unstable), np.array(unstablef)

    def Compute_ms_and_as_fre(self):
        forward, forwardf, backward, backwardf, unstable, unstablef = self.BS_fre(self.P_d)
        m_sf, a_sf = self.Compute_ms_and_as_Delta_plus(self.P_d, forwardf, forward)
        m_sb, a_sb = self.Compute_ms_and_as_Delta_plus(self.P_d, backwardf, backward)
        m_su, a_su = self.Compute_ms_and_as_Delta_plus(self.P_d, unstablef, unstable)
        return m_sf, a_sf, m_sb, a_sb, m_su, a_su, forward, forwardf, backward, backwardf, unstable, unstablef

    def BS_2D(self):
        F = []
        B = []
        for i, D in enumerate(self.P_d):
            forward, forwardf, backward, backwardf, unstable, unstablef = self.BS_fre(D)
            F.append(forward)
            B.append(backward)
        return F, B

    def BS_2D_with_ms_and_as(self):
        F = []
        B = []
        F_ms=[]
        F_as=[]
        B_ms=[]
        B_as=[]
        for i, D in enumerate(self.P_d):
            forward, forwardf, backward, backwardf, unstable, unstablef = self.BS_fre(D)
            fms,fas=self.Compute_ms_and_as_Delta_plus(D, forwardf, forward)
            bms,bas=self.Compute_ms_and_as_Delta_plus(D, backwardf, backward)
            F.append(forward)
            B.append(backward)
            F_ms.append(fms)
            F_as.append(fas)
            B_ms.append(bms)
            B_as.append(bas)
        return F, B,F_ms,F_as,B_ms,B_as

    def m_a_evolution(self,m_s0,a_s0,P,fre,interval,steps):
        omega_d=np.array(fre)*(2*np.pi)
        delta_a=self.omega_a-omega_d
        delta_m=self.omega_m-omega_d
        S_a=-1j*delta_a-self.ka/2
        S_m=-1j*delta_m-self.km/2

        Time=[]
        M_s=[]
        A_s=[]

        M_s.append(m_s0)
        A_s.append(a_s0)
        Time.append(0)

        for i in range(round(steps)):
            da=S_a*A_s[i]-1j*self.g_ma*M_s[i]+np.sqrt(self.kaed)*np.sqrt(P/(hbar*omega_d))
            dm=S_m*M_s[i]-1j*self.g_ma*A_s[i]-1j*self.K*(2*np.abs(M_s[i])**2+1)*M_s[i]

            anext=da*interval+A_s[i]
            mnext=dm*interval+M_s[i]

            A_s.append(anext)
            M_s.append(mnext)
            Time.append(interval*(i+1))
        return np.array(M_s), np.array(A_s),np.array(Time)

    ##接下来的函数中Ps和fs的长度必须是相同的
    def P_F_len_determination(self):
        if len(self.omega_d)==len(self.P_d):
            pass
        else:
            raise Exception("Ps and fs are not the same length!")

    def Parameter_definition_Ps_and_fs(self):
        delta_a = self.omega_a - self.omega_d
        delta_m = self.omega_m - self.omega_d

        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2
        S_am = S_m + self.g_ma ** 2 / S_a

        Imag = S_am.imag
        Real = S_am.real
        C = 2 * self.K * self.kaed / (hbar * self.omega_d) * np.abs(self.g_ma / S_a) ** 2 + 2 * self.K * self.kmed / (
                hbar * self.omega_d)

        return Real, Imag, C

    def Get_real_solution_Ps_and_fs(self, A:np.ndarray, B:np.ndarray, C:np.ndarray):
        x0 = []
        x1 = []
        x2 = []
        y0 = []
        y1 = []
        y2 = []
        for i, D in enumerate(self.P_d):
            t = Solve_Cubic_Equation(A[i], B[i], C[i], D)
            if sp.Abs(sp.im(t[0])) < 1:
                omega_m0 = self.omega_m + float(sp.re(t[0]))
                y0.append(float((self.branch_fre(omega_m0) - self.omega_start) / (2*np.pi)))
                x0.append(round(D, 9))

            if sp.Abs(sp.im(t[1])) < 1:
                omega_m1 = self.omega_m + float(sp.re(t[1]))
                y1.append(float((self.branch_fre(omega_m1) - self.omega_start) / (2*np.pi)))
                x1.append(round(D, 9))

            if sp.Abs(sp.im(t[2])) < 1:
                omega_m2 = self.omega_m + float(sp.re(t[2]))
                y2.append(float((self.branch_fre(omega_m2) - self.omega_start) / (2*np.pi)))
                x2.append(round(D, 9))
        return x0, y0, x1, y1, x2, y2


class Bistability_with_K_evo(): #all Hz with 2pi
    def __init__(self, **kwargs):  # P_d and omega_d have the same length
        self.omega_a = kwargs.get('omega_a', 0) *(2*np.pi)  # cavity frequency (Hz*2pi)
        self.omega_m = kwargs.get('omega_m', 0) *(2*np.pi)  # initial magnon frequency (Hz*2pi)
        self.kaint = kwargs.get('kaint', 0) *(2*np.pi)  # cavity internal dissipation (Hz*2pi)
        self.kaed = kwargs.get('kaed', 0) *(2*np.pi)  # cavity drive dissipation (Hz*2pi)
        self.kaep = kwargs.get('kaep', 0) *(2*np.pi)  # cavity probe dissipation (Hz*2pi)
        self.kmint = kwargs.get('kmint', 0) *(2*np.pi)  # magnon internal dissipation (Hz*2pi)
        self.kmed = kwargs.get('kmext', 0) *(2*np.pi)  # magnon drive dissipation (Hz*2pi)
        self.P_d = np.asarray(kwargs.get('P_d'))  # drive power (W)
        self.omega_d = np.asarray(kwargs.get('omega_d')) *(2*np.pi)  # drive frequency (Hz*2pi)
        self.g_ma = kwargs.get('g_ma', 0) *(2*np.pi)  # coupling strength between cavity and magnon (Hz*2pi)
        self.branch = kwargs.get('branch', 'upper')  # Magnon or upper branch or lower branch
        self.omega_start = self.branch_fre(self.omega_m)  # the start point of simulation or experiment (Hz*2pi)
        self.K = kwargs.get('K', 0) *(2*np.pi)  # Kerr coefficient (Hz*2pi)

        self.ka = self.kaint + self.kaed + self.kaep  # cavity dissipation (Hz*2pi)
        self.km = self.kmint + self.kmed  # magnon dissipation (Hz*2pi)
        self.Delta_am = self.omega_a - self.omega_m  # detuning between cavity and magnon (Hz*2pi)
        self.P_F_len_determination()

    def P_F_len_determination(self):
        if len(self.omega_d)==len(self.P_d):
            print("Ps and fs have the same length!")
        else:
            raise Exception("Ps and fs have the difference length!")

    def branch_fre(self, omega_m):
        omega_UP = (self.omega_a + omega_m) / 2 + np.sqrt((self.omega_a - omega_m) ** 2 / 4 + self.g_ma ** 2)
        omega_LP = (self.omega_a + omega_m) / 2 - np.sqrt((self.omega_a - omega_m) ** 2 / 4 + self.g_ma ** 2)
        if self.branch == 'lower':
            omega_out = omega_LP
        elif self.branch == 'upper':
            omega_out = omega_UP
        elif (self.branch != 'lower') & (self.branch != 'upper'):
            omega_out = omega_m
        return np.array(omega_out)#(Hz*2pi) np.narray

    def wplus_to_wm(self, wplus):
        Wplus = np.array(wplus)
        fenzi = Wplus ** 2 - Wplus * self.omega_a - self.g_ma ** 2
        fenmu = Wplus - self.omega_a
        wm = fenzi / fenmu

        # wm = fenzi / fenmu
        # return wm
        return wm  #(Hz*2pi) np.narray

    def Compute_ms_and_as_Delta_plus(self,P,fre,Delta_plus):
        Fre=fre*(2*np.pi)
        delta_a = self.omega_a - Fre
        delta_m = self.omega_m - Fre

        wplus_init = self.branch_fre(self.omega_m)
        wplus_finalf = wplus_init + Delta_plus * (2*np.pi)
        Delta_mf = self.wplus_to_wm(wplus_finalf) - self.omega_m

        fenzif = -1j * self.g_ma * np.sqrt(self.kaed) * np.sqrt(P / (hbar * Fre))
        fenmuf = (1j * delta_a + self.ka / 2) * (1j * (delta_m + Delta_mf) + self.km / 2) + self.g_ma ** 2

        m_s = fenzif / fenmuf
        a_s = ((-1j * self.g_ma * m_s + np.sqrt(self.kaed) * np.sqrt(P / (hbar * Fre))) / (
            (1j * delta_a + self.ka / 2)))

        return a_s,m_s # np.narray and complex

    def Compara_as_and_ms_difference(self,a0,m0,a1,m1,a2,m2):
        d1=np.abs(a1-a0)+np.abs(m1-m0)
        d2=np.abs(a2-a0)+np.abs(m2-m0)
        if d1>d2:
            a_next=a2
            m_next=m2
        else:
            a_next = a1
            m_next = m1

        return a_next,m_next

    def Parameter_definition(self):
        delta_a = self.omega_a - self.omega_d
        delta_m = self.omega_m - self.omega_d

        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2
        S_am = S_m + self.g_ma ** 2 / S_a

        Imag = S_am.imag
        Real = S_am.real
        Cc = 2 * self.K * self.kaed / (hbar * self.omega_d) * np.abs(self.g_ma / S_a) ** 2 + 2 * self.K * self.kmed / (
                hbar * self.omega_d)

        return Real, Imag, Cc ## np.narray

    def Get_BS(self, A, B, C,D):#D=power  and this is bistability single point
        Real, Imag, Cc=self.Parameter_definition()
        index1=np.where(self.P_d==D)
        index2 = np.where(-Imag == A)

        if len(index1[0])==1:
            index=index1
        if len(index2[0])==1:
            index=index2
        if (len(index1[0])!=1)&(len(index2[0])!=1):
            raise Exception("Something wrong!")
        t = Solve_Cubic_Equation(A, B, C, D)
        if sp.Abs(sp.im(t[0])) < 1:
            omega_m0 = self.omega_m + float(sp.re(t[0]))
            y0=(float((self.branch_fre(omega_m0) - self.omega_start) / (2*np.pi)))
            x0=(round(D, 9))
            z0=(self.omega_d[index]/ (2*np.pi))

        else:
            y0 = np.nan
            x0 = np.nan
            z0 = np.nan
        if sp.Abs(sp.im(t[1])) < 1:
            omega_m1 = self.omega_m + float(sp.re(t[1]))
            y1=(float((self.branch_fre(omega_m1) - self.omega_start) / (2*np.pi)))
            x1=(round(D, 9))
            z1=(self.omega_d[index] / (2 * np.pi))
        else:
            y1 = np.nan
            x1 = np.nan
            z1 = np.nan
        if sp.Abs(sp.im(t[2])) < 1:
            omega_m2 = self.omega_m + float(sp.re(t[2]))
            y2=(float((self.branch_fre(omega_m2) - self.omega_start) / (2*np.pi)))
            x2=(round(D, 9))
            z2=(self.omega_d[index] / (2 * np.pi))

        else:
            y2 = np.nan
            x2 = np.nan
            z2 = np.nan

        return x0, y0,z0, x1, y1,z1, x2, y2,z2 # x=power,z=fre,y=delta

    def Get_BS_with_as_and_ms(self,A, B, C,D):
        x0, y0, z0, x1, y1, z1, x2, y2, z2=self.Get_BS(A, B, C,D)

        if x0==np.nan:
            as0=np.nan
            ms0=np.nan
        else:
            as0,ms0=self.Compute_ms_and_as_Delta_plus(x0, z0, y0)
        if x1==np.nan:
            as1 = np.nan
            ms1 = np.nan
        else:
            as1,ms1=self.Compute_ms_and_as_Delta_plus( x1, z1, y1)

        if x2==np.nan:
            as2 = np.nan
            ms2 = np.nan
        else:
            as2,ms2=self.Compute_ms_and_as_Delta_plus(x2, z2, y2)

        return x0, y0, z0,as0,ms0,x1, y1, z1,as1,ms1,x2, y2, z2,as2,ms2

    def BS_array_with_as_and_ms(self,start_energy='lower'): #start_energy corresponds to the initial energy, the higher or the lower
        power=[] #power=x
        delta=[] #delta=y
        wd=[] #wd=z
        a_s=[]
        m_s=[]
        Real, Imag, Cc = self.Parameter_definition()
        for i in range(len(self.P_d)):
            x0, y0, z0,as0,ms0,x1, y1, z1,as1,ms1,x2, y2, z2,as2,ms2= self.Get_BS_with_as_and_ms(-Imag[i], Real[i], Cc[i],self.P_d[i])

            if i==0:
                if (np.isnan(x1))&(np.isnan(x2)):
                    power.append(x0)
                    delta.append(y0)
                    wd.append(z0)
                    a_s.append(as0)
                    m_s.append(ms0)
                elif start_energy=='lower':
                    if y0>y2:
                        power.append(x2)
                        delta.append(y2)
                        wd.append(z2)
                        a_s.append(as2)
                        m_s.append(ms2)
                    else:
                        power.append(x0)
                        delta.append(y0)
                        wd.append(z0)
                        a_s.append(as0)
                        m_s.append(ms0)
                elif start_energy=='higher':
                    if y0>y2:
                        power.append(x0)
                        delta.append(y0)
                        wd.append(z0)
                        a_s.append(as0)
                        m_s.append(ms0)
                    else:
                        power.append(x2)
                        delta.append(y2)
                        wd.append(z2)
                        a_s.append(as2)
                        m_s.append(ms2)
            else:
                a_next,m_next=self.Compara_as_and_ms_difference(a_s[i-1], m_s[i-1], as0, ms0,as2, ms2)
                if a_next==as0:
                    power.append(x0)
                    delta.append(y0)
                    wd.append(z0)
                    a_s.append(as0)
                    m_s.append(ms0)
                if a_next==as2:
                    power.append(x2)
                    delta.append(y2)
                    wd.append(z2)
                    a_s.append(as2)
                    m_s.append(ms2)
        return power,delta,wd,a_s,m_s

    def Unstable_array_with_as_and_ms(self):
        power = []  # power=x
        delta = []  # delta=y
        wd = []  # wd=z
        a_s = []
        m_s = []
        Real, Imag, Cc = self.Parameter_definition()
        for i in range(len(self.P_d)):
            x0, y0, z0, as0, ms0, x1, y1, z1, as1, ms1, x2, y2, z2, as2, ms2 = self.Get_BS_with_as_and_ms(-Imag[i],
                                                                                                          Real[i],
                                                                                                          Cc[i],
                                                                                                          self.P_d[i])
            if (np.isnan(x1))==False:
                power.append(x1)
                delta.append(y1)
                wd.append(z1)
                a_s.append(as1)
                m_s.append(ms1)

        return power, delta, wd, a_s, m_s

    def m_a_evolution(self,Pi,fi,Pf,ff,interval,steps,start_energy='lower'):
        Fi=fi*2*np.pi
        Ff=ff*2*np.pi

        power = []  # power=x
        delta = []  # delta=y
        wd = []  # wd=z
        a_s = []
        m_s = []
        Time=[]

        delta_ai = self.omega_a - Fi
        delta_mi = self.omega_m - Fi

        S_ai = -1j * delta_ai - self.ka / 2
        S_mi = -1j * delta_mi - self.km / 2
        S_ami = S_mi + self.g_ma ** 2 / S_ai

        Imagi = S_ami.imag
        Reali= S_ami.real
        Cci = 2 * self.K * self.kaed / (hbar * Fi) * np.abs(self.g_ma / S_ai) ** 2 + 2 * self.K * self.kmed / (
                hbar * Fi)


        delta_a = self.omega_a - Ff
        delta_m = self.omega_m - Ff
        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2
        for i in range(round(steps)):
            # print(i)
            Time.append(interval*i)
            if i==0:
                x0, y0, z0, as0, ms0, x1, y1, z1, as1, ms1, x2, y2, z2, as2, ms2 = self.Get_BS_with_as_and_ms(-Imagi,
                                                                                                              Reali,
                                                                                                              Cci,
                                                                                                              Pi)
                if (np.isnan(x1))&(np.isnan(x2)):
                    print('1if')
                    power.append(x0)
                    delta.append(y0)
                    wd.append(z0)
                    a_s.append(as0[0])
                    m_s.append(ms0[0])
                elif start_energy=='lower':
                    if y0>y2:
                        print('2if')
                        power.append(x2)
                        delta.append(y2)
                        wd.append(z2)
                        a_s.append(as2[0])
                        m_s.append(ms2[0])
                    else:
                        print('3if')
                        power.append(x0)
                        delta.append(y0)
                        wd.append(z0)
                        a_s.append(as0[0])
                        m_s.append(ms0[0])
                elif start_energy=='higher':
                    if y0>y2:
                        print('4if')
                        power.append(x0)
                        delta.append(y0)
                        wd.append(z0)
                        a_s.append(as0[0])
                        m_s.append(ms0[0])
                    else:
                        print('5if')
                        power.append(x2)
                        delta.append(y2)
                        wd.append(z2)
                        a_s.append(as2[0])
                        m_s.append(ms2[0])
        # initial state is chosen

            else:
                da=S_a*a_s[i-1]-1j*self.g_ma*m_s[i-1]+np.sqrt(self.kaed)*np.sqrt(Pf/(hbar*Ff))
                dm=S_m*m_s[i-1]-1j*self.g_ma*a_s[i-1]-1j*self.K*(2*np.abs(m_s[i-1])**2+1)*m_s[i-1]
                anext=da*interval+a_s[i-1]
                mnext=dm*interval+m_s[i-1]

                a_s.append(anext)
                m_s.append(mnext)

                delta_next=self.branch_fre(self.omega_m+2*self.K*np.abs(m_s[i])**2)-self.omega_start
                delta.append(delta_next/(2*np.pi))
        return np.array(a_s),np.array(m_s),np.array(delta), np.array(Time),np.array(power),np.array(wd)

        # y0 = (float((self.branch_fre(omega_m0) - self.omega_start) / (2 * np.pi)))


        # omega_d=np.array(fre)*(2*np.pi)
        # delta_a=self.omega_a-omega_d
        # delta_m=self.omega_m-omega_d
        # S_a=-1j*delta_a-self.ka/2
        # S_m=-1j*delta_m-self.km/2
        #
        # Time=[]
        # M_s=[]
        # A_s=[]
        #
        # M_s.append(m_s0)
        # A_s.append(a_s0)
        # Time.append(0)
        #
        # for i in range(round(steps)):
        #     da=S_a*A_s[i]-1j*self.g_ma*M_s[i]+np.sqrt(self.kaed)*np.sqrt(P/(hbar*omega_d))
        #     dm=S_m*M_s[i]-1j*self.g_ma*A_s[i]-1j*self.K*(2*np.abs(M_s[i])**2+1)*M_s[i]
        #
        #     anext=da*interval+A_s[i]
        #     mnext=dm*interval+M_s[i]
        #
        #     A_s.append(anext)
        #     M_s.append(mnext)
        #     Time.append(interval*(i+1))
        # return np.array(M_s), np.array(A_s),np.array(Time)

