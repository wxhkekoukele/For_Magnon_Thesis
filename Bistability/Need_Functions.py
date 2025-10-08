import sympy as sp
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

hbar = 6.626e-34 / (2 * np.pi)


def solve_cubic_equation(A, B, C, D):
    x = sp.Symbol('x')
    f = ((A + x) ** 2 + B ** 2) * x - C * D
    t = sp.solve(f, x)
    return t

class Bistability():
    def __init__(self, **kwargs):
        self.omega_a = kwargs.get('omega_a', 0) * 1e9 * 2 * np.pi
        self.omega_m = kwargs.get('omega_m', 0) * 1e9 * 2 * np.pi
        self.kaint = kwargs.get('kaint', 0) * 1e6 * 2 * np.pi
        self.kaed = kwargs.get('kaed', 0) * 1e6 * 2 * np.pi
        self.kaep = kwargs.get('kaep', 0) * 1e6 * 2 * np.pi
        self.kmint = kwargs.get('kmint', 0) * 1e6 * 2 * np.pi
        self.kmext = kwargs.get('kmext', 0) * 1e6 * 2 * np.pi
        self.P_d = kwargs.get('P_d')
        self.omega_d = kwargs.get('omega_d')
        self.g_ma = kwargs.get('g_ma', 0) * 1e6 * 2 * np.pi
        self.branch = kwargs.get('branch', 'upper')
        self.omega_start = self.branch_fre(self.omega_m)
        self.K = kwargs.get('K', 0) * 1e-9 * 2 * np.pi

        self.ka = self.kaint + self.kaed + self.kaep
        self.km = self.kmint + self.kmext
        self.Delta = self.omega_a - self.omega_m

    def branch_fre(self, omega_m):
        omega_UP = (self.omega_a + omega_m) / 2 + np.sqrt((self.omega_a - omega_m) ** 2 / 4 + self.g_ma ** 2)
        omega_LP = (self.omega_a + omega_m) / 2 - np.sqrt((self.omega_a - omega_m) ** 2 / 4 + self.g_ma ** 2)
        if self.branch == 'lower':
            omega_out = omega_LP
        elif self.branch == 'upper':
            omega_out = omega_UP
        elif (self.branch != 'lower') & (self.branch != 'upper'):
            omega_out = omega_m
        return omega_out

    def before_BS_power(self):
        omega_D = self.omega_d * 1e9 * 2 * np.pi
        delta_a = self.omega_a - omega_D
        delta_m = self.omega_m - omega_D
        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2
        S_am = S_m + self.g_ma ** 2 / S_a
        Imag = S_am.imag
        Real = S_am.real
        C = 2 * self.K / (hbar * omega_D) * self.kaed * np.abs(
            self.g_ma / (S_a)) ** 2 + 2 * self.K / (hbar * omega_D) * self.kmext
        return Real, Imag, C

    def get_real_solutions_power(self, A, B, C):
        x0 = []
        x1 = []
        x2 = []
        y0 = []
        y1 = []
        y2 = []
        for i, D in enumerate(self.P_d):
            t = solve_cubic_equation(A, B, C, D)
            if sp.Abs(sp.im(t[0])) < 1:
                omega_m0 = self.omega_m + float(sp.re(t[0]))
                y0.append(float((self.branch_fre(omega_m0) - self.omega_start) / (1e6 * 2 * np.pi)))
                x0.append(round(D, 9))

            if sp.Abs(sp.im(t[1])) < 1:
                omega_m1 = self.omega_m + float(sp.re(t[1]))
                y1.append(float((self.branch_fre(omega_m1) - self.omega_start) / (1e6 * 2 * np.pi)))
                x1.append(round(D, 9))

            if sp.Abs(sp.im(t[2])) < 1:
                omega_m2 = self.omega_m + float(sp.re(t[2]))
                y2.append(float((self.branch_fre(omega_m2) - self.omega_start) / (1e6 * 2 * np.pi)))
                x2.append(round(D, 9))
        return x0, y0, x1, y1, x2, y2

    def BS_power(self):
        Real, Imag, C = self.before_BS_power()
        p0, s0, p1, s1, p2, s2 = self.get_real_solutions_power(-Imag, Real, C)
        forward = s0.copy()
        forwardp = p0.copy()
        for i in range(len(p0)):
            for j in range(len(p2)):
                if p0[i] == p2[j]:
                    s0[i] = s2[j]
        backward = s0
        backwardp = p0
        return forward, forwardp, backward, backwardp

    def BS_power_with_unstable(self):
        Real, Imag, C = self.before_BS_power()
        p0, s0, p1, s1, p2, s2 = self.get_real_solutions_power(-Imag, Real, C)
        forward = s0.copy()
        forwardp = p0.copy()
        for i in range(len(p0)):
            for j in range(len(p2)):
                if p0[i] == p2[j]:
                    s0[i] = s2[j]
        backward = s0
        backwardp = p0
        return forward, forwardp, backward, backwardp, p1, s1

    def BS_power_inside_BS(self):
        forward1, forwardp1, backward1, backwardp1, unstablep, unstable=self.BS_power_with_unstable()

        start_index_for_forward = list(forwardp1).index(unstablep[0])
        stop_index_for_forward = list(forwardp1).index(unstablep[-1])
        start_index_for_backward = list(backwardp1).index(unstablep[0])
        stop_index_for_backward = list(backwardp1).index(unstablep[-1])
        # print(unstablep)
        forwardp=forwardp1[start_index_for_forward-1:stop_index_for_forward +2]
        forward=forward1[start_index_for_forward-1:stop_index_for_forward +2]
        backwardp = backwardp1[start_index_for_backward - 1:stop_index_for_backward + 2]
        backward = backward1[start_index_for_backward - 1:stop_index_for_backward + 2]
        print('start')
        print(unstablep[0])
        print(forwardp[0])
        print(backwardp[0])
        print('stop')
        print(unstablep[-1])
        print(forwardp[-1])
        print(backwardp[-1])

        return forward,forwardp,backward,backwardp, unstable,unstablep

    def before_BS_fre(self):
        omega_D = np.array(self.omega_d) * 1e9 * 2 * np.pi
        delta_a = self.omega_a - omega_D
        delta_m = self.omega_m - omega_D
        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2
        S_am = S_m + self.g_ma ** 2 / S_a
        Imag = S_am.imag
        Real = S_am.real
        C = 2 * self.K / (hbar * omega_D) * self.kaed * np.abs(
            self.g_ma / (S_a)) ** 2 + 2 * self.K / (hbar * omega_D) * self.kmext
        return Real, Imag, C

    def get_real_solutions_fre(self, A: list, B: list, C):
        x0 = []
        x1 = []
        x2 = []
        y0 = []
        y1 = []
        y2 = []
        for i, f in enumerate(self.omega_d):
            t = solve_cubic_equation(A[i], B[i], C[i], self.P_d)
            if sp.Abs(sp.im(t[0])) < 1:
                omega_m0 = self.omega_m + float(sp.re(t[0]))
                y0.append(float((self.branch_fre(omega_m0) - self.omega_start) / (1e6 * 2 * np.pi)))
                x0.append(round(f, 9))

            if sp.Abs(sp.im(t[1])) < 1:
                omega_m1 = self.omega_m + float(sp.re(t[1]))
                y1.append(float((self.branch_fre(omega_m1) - self.omega_start) / (1e6 * 2 * np.pi)))
                x1.append(round(f, 9))

            if sp.Abs(sp.im(t[2])) < 1:
                omega_m2 = self.omega_m + float(sp.re(t[2]))
                y2.append(float((self.branch_fre(omega_m2) - self.omega_start) / (1e6 * 2 * np.pi)))
                x2.append(round(f, 9))
        return x0, y0, x1, y1, x2, y2

    def BS_fre_with_unstable(self):
        Real, Imag, C = self.before_BS_fre()
        f0, s0, f1, s1, f2, s2 = self.get_real_solutions_fre(-Imag, Real, C)
        backward = s0.copy()
        backwardf = f0.copy()
        for i in range(len(f0)):
            for j in range(len(f2)):
                if f0[i] == f2[j]:
                    s0[i] = s2[j]
        forward = s0
        forwardf = f0
        return forward, forwardf, backward, backwardf,s1,f1

    def BS_fre_inside_BS(self):
        forward1, forwardf1, backward1, backwardf1, unstable, unstablef = self.BS_fre_with_unstable()

        start_index_for_forward = list(forwardf1).index(unstablef[0])
        stop_index_for_forward = list(forwardf1).index(unstablef[-1])
        start_index_for_backward = list(backwardf1).index(unstablef[0])
        stop_index_for_backward = list(backwardf1).index(unstablef[-1])
        # print(unstablep)
        forwardf = forwardf1[start_index_for_forward - 1:stop_index_for_forward + 2]
        forward = forward1[start_index_for_forward - 1:stop_index_for_forward + 2]
        backwardf = backwardf1[start_index_for_backward - 1:stop_index_for_backward + 2]
        backward = backward1[start_index_for_backward - 1:stop_index_for_backward + 2]
        print('start')
        print(unstablef[0])
        print(forwardf[0])
        print(backwardf[0])
        print('stop')
        print(unstablef[-1])
        print(forwardf[-1])
        print(backwardf[-1])

        return forward, forwardf, backward, backwardf, unstable, unstablef

    def Power_side(self):
        fl = []
        fs = []
        upper = []
        lower = []
        Real, Imag, C = self.before_BS_power()
        for j, F in enumerate(tqdm(self.omega_d, position=0)):
            x1 = []
            y1 = []
            for i, D in enumerate(self.P_d):
                t = solve_cubic_equation(-Imag[j], Real[j], C[j], D)
                if len(t) < 2:
                    continue
                if sp.Abs(sp.im(t[1])) < 1:
                    omega_m1 = self.omega_m + float(sp.re(t[1]))
                    y1.append(float((self.branch_fre(omega_m1) - self.omega_start) / (1e6 * 2 * np.pi)))
                    x1.append(round(D, 5))
                else:
                    continue
            if len(x1) != 0:
                if max(x1) in upper:
                    fs.append(F)
                    lower.append(min(x1))
                else:
                    fl.append(F)
                    fs.append(F)
                    upper.append(max(x1))
                    lower.append(min(x1))
            if ((len(x1) == 1)&(F<8.18)):
                critical_power = x1[0]
                critical_fre = F
            else:
                critical_power='None'
                critical_fre='None'
        return fl, fs, upper, lower, critical_power, critical_fre

    def Fre_side(self):
        P = []
        upper = []
        lower = []
        Real, Imag, C = self.before_BS_fre()
        # print(Real)
        # print(Imag)
        # print(C)
        for j, D in enumerate(tqdm(self.P_d, position=0)):
            x1 = []
            y1 = []
            for i, F in enumerate(self.omega_d):
                t = solve_cubic_equation(-Imag[i], Real[i], C[i], D)
                # print(t)
                if len(t) < 2:
                    continue
                if sp.Abs(sp.im(t[1])) < 1:
                    omega_m1 = self.omega_m + float(sp.re(t[1]))
                    y1.append(float((self.branch_fre(omega_m1) - self.omega_start) / (1e6 * 2 * np.pi)))
                    x1.append(round(F, 5))
                else:
                    continue
            if len(x1) != 0:
                P.append(D)
                upper.append(max(x1))
                lower.append(min(x1))
        return P, upper, lower

    def before_BS(self):
        omega_D = self.omega_d * 1e9 * 2 * np.pi
        delta_a = self.omega_a - omega_D
        delta_m = self.omega_m - omega_D
        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2
        S_am = S_m + self.g_ma ** 2 / S_a
        Imag = S_am.imag
        Real = S_am.real
        C = 2 * self.K / (hbar * omega_D) * self.kaed * np.abs(
            self.g_ma / (S_a)) ** 2 + 2 * self.K / (hbar * omega_D) * self.kmext
        return Real, Imag, C

    def get_real_solutions(self, A, B, C):
        x0 = []
        x1 = []
        x2 = []
        y0 = []
        y1 = []
        y2 = []
        t = solve_cubic_equation(A, B, C, self.P_d)
        if sp.Abs(sp.im(t[0])) < 1:
            omega_m0 = self.omega_m + float(sp.re(t[0]))
            y0.append(float((self.branch_fre(omega_m0) - self.omega_start) / (1e6 * 2 * np.pi)))
            x0.append(round(self.P_d, 5))

        if sp.Abs(sp.im(t[1])) < 1:
            omega_m1 = self.omega_m + float(sp.re(t[1]))
            y1.append(float((self.branch_fre(omega_m1) - self.omega_start) / (1e6 * 2 * np.pi)))
            x1.append(round(self.P_d, 5))

        if sp.Abs(sp.im(t[2])) < 1:
            omega_m2 = self.omega_m + float(sp.re(t[2]))
            y2.append(float((self.branch_fre(omega_m2) - self.omega_start) / (1e6 * 2 * np.pi)))
            x2.append(round(self.P_d, 5))
        return x0, y0, x1, y1, x2, y2

    def BS(self):
        Real, Imag, C = self.before_BS()
        x0, y0, x1, y1, x2, y2 = self.get_real_solutions(-Imag, Real, C)
        return x0, y0, x1, y1, x2, y2

    def wplus_to_wm(self,wplus):
        Wplus=np.array(wplus)
        fenzi=Wplus**2-Wplus*self.omega_a-self.g_ma**2
        fenmu=Wplus-self.omega_a
        wm=fenzi/fenmu
        return wm

    def cal_ms_as_P(self,Delta_m,corres_P):
        omega_D= np.array(self.omega_d)*1e9*2*np.pi
        delta_a = np.array(self.omega_a - omega_D)
        delta_m = np.array(self.omega_m - omega_D)

        fenzi=-1j*self.g_ma*np.sqrt(self.kaed)*np.sqrt(corres_P/(hbar*omega_D))
        fenmu=(1j*delta_a+self.ka/2)*(1j*(delta_m+Delta_m)+self.km/2)+self.g_ma**2

        m_s=fenzi/fenmu
        a_s=((-1j*self.g_ma*m_s+np.sqrt(self.kaed)*np.sqrt(corres_P/(hbar*omega_D)))/((1j*delta_a+self.ka/2)))
        return m_s,a_s

    def cal_ms_as_f(self, Delta_m, corres_f):
        omega_D = np.array(corres_f) * 1e9 * 2 * np.pi
        delta_a = np.array(self.omega_a - omega_D)
        delta_m = np.array(self.omega_m - omega_D)

        fenzi = -1j * self.g_ma * np.sqrt(self.kaed) * np.sqrt(self.P_d / (hbar * omega_D))
        fenmu = (1j * delta_a + self.ka / 2) * (1j * (delta_m + Delta_m) + self.km / 2) + self.g_ma ** 2

        m_s = fenzi / fenmu
        a_s = ((-1j * self.g_ma * m_s + np.sqrt(self.kaed) * np.sqrt(self.P_d / (hbar * omega_D))) / (
        (1j * delta_a + self.ka / 2)))

        return m_s, a_s

    def m_a_evo(self,m_s0,a_s0,interval,steps,P_d,f_d):
        omega_D = f_d * 1e9 * 2 * np.pi
        delta_a = self.omega_a - omega_D
        delta_m = self.omega_m - omega_D
        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2

        M_s=[]
        A_s=[]
        Time=[]
        M_s.append(m_s0)
        A_s.append(a_s0)
        M_sr = []
        A_sr = []
        M_si = []
        A_si = []
        M_sr.append(m_s0.real)
        A_sr.append(a_s0.real)
        M_si.append(m_s0.imag)
        A_si.append(a_s0.imag)
        Time.append(0)
        for i in range(round(steps)):
            da=S_a*A_s[i]-1j*self.g_ma*M_s[i]+np.sqrt(self.kaed) * np.sqrt(P_d / (hbar * omega_D))
            dm=S_m*M_s[i]-1j*self.g_ma*A_s[i]-1j*self.K*(2*np.abs(M_s[i])**2+1)*M_s[i]
            # -3361378.0073090377

            anext = da * interval + A_s[i]
            mnext = dm * interval + M_s[i]

            # if np.abs((da * interval)/A_s[i])<1e-11:
            #     anext = A_s[i]
            # else:
            #     anext = da * interval + A_s[i]
            #
            # if np.abs((dm * interval)/M_s[i])<1e-11:
            #     mnext = M_s[i]
            # else:
            #     mnext = dm * interval + M_s[i]
            A_s.append(anext)
            M_s.append(mnext)
            M_sr.append(mnext.real)
            A_sr.append(anext.real)
            M_si.append(mnext.imag)
            A_si.append(anext.imag)

            Time.append(interval*(i+1))

        return M_sr,M_si,A_sr,A_si,Time

    def m_a_evo_and_back_P(self,m_s0,a_s0,interval,steps1,P_d1,steps2,P_d2,f_d):
        ## steps1 is jump time, steps2 is back time
        ## P_d1 is jump power, P_d2 is back power
        M_sr1,M_si1,A_sr1,A_si1,Time1=self.m_a_evo(m_s0,a_s0,interval,steps1,P_d1,f_d)
        M_sr2,M_si2,A_sr2,A_si2,Time2=self.m_a_evo(M_sr1[-1]+1j*M_si1[-1],A_sr1[-1]+1j*A_si1[-1],interval,steps2,P_d2,f_d)
        M_sr=np.hstack((M_sr1,np.delete(M_sr2, 0)))
        M_si=np.hstack((M_si1,np.delete(M_si2, 0)))
        A_sr=np.hstack((A_sr1,np.delete(A_sr2, 0)))
        A_si=np.hstack((A_si1,np.delete(A_si2, 0)))
        for i in range(len(Time2)):
            Time2[i]=Time2[i]+steps1*interval

        Time=np.hstack((Time1,np.delete(Time2, 0)))

        return M_sr,M_si,A_sr,A_si,Time

    def m_a_evo_and_back_f(self,m_s0,a_s0,interval,steps1,f_d1,steps2,f_d2,P_d):
        ## steps1 is jump time, steps2 is back time
        ## f_d1 is jump frequency, P_d2 is back frequency
        M_sr1,M_si1,A_sr1,A_si1,Time1=self.m_a_evo(m_s0,a_s0,interval,steps1,P_d,f_d1)
        M_sr2,M_si2,A_sr2,A_si2,Time2=self.m_a_evo(M_sr1[-1]+1j*M_si1[-1],A_sr1[-1]+1j*A_si1[-1],interval,steps2,P_d,f_d2)
        M_sr=np.hstack((M_sr1,np.delete(M_sr2, 0)))
        M_si=np.hstack((M_si1,np.delete(M_si2, 0)))
        A_sr=np.hstack((A_sr1,np.delete(A_sr2, 0)))
        A_si=np.hstack((A_si1,np.delete(A_si2, 0)))
        for i in range(len(Time2)):
            Time2[i]=Time2[i]+steps1*interval

        Time=np.hstack((Time1,np.delete(Time2, 0)))

        return M_sr,M_si,A_sr,A_si,Time

    def m_a_evo_Periodic_P(self,m_s0,a_s0,interval,steps,P_start,P_stop,f_d,f_P_d):
        omega_D = f_d * 1e9 * 2 * np.pi
        delta_a = self.omega_a - omega_D
        delta_m = self.omega_m - omega_D
        S_a = -1j * delta_a - self.ka / 2
        S_m = -1j * delta_m - self.km / 2
        M_s=[]
        A_s=[]
        M_s.append(m_s0)
        A_s.append(a_s0)
        M_sr = []
        A_sr = []
        M_si = []
        A_si = []
        M_sr.append(m_s0.real)
        A_sr.append(a_s0.real)
        M_si.append(m_s0.imag)
        A_si.append(a_s0.imag)
        Time=[]
        Time.append(0)
        P_ds=[]
        P_ds.append(P_start)
        for i in range(round(steps)):
            P_d=(P_start+P_stop)/2+(P_stop-P_start)/2*np.sin(f_P_d*2*np.pi*Time[-1])
            da=S_a*A_s[i]-1j*self.g_ma*M_s[i]+np.sqrt(self.kaed) * np.sqrt(P_d / (hbar * omega_D))
            dm=S_m*M_s[i]-1j*self.g_ma*A_s[i]-1j*self.K*(2*np.abs(M_s[i])**2+1)*M_s[i]

            anext = da * interval + A_s[i]
            mnext = dm * interval + M_s[i]

            A_s.append(anext)
            M_s.append(mnext)
            M_sr.append(mnext.real)
            A_sr.append(anext.real)
            M_si.append(mnext.imag)
            A_si.append(anext.imag)
            P_ds.append(P_d)
            Time.append(interval*(i+1))

        return M_sr,M_si,A_sr,A_si,Time,P_ds

    def m_a_evo_Periodic_f(self,m_s0,a_s0,interval,steps,f_start,f_stop,P_d,f_P_d):
        f_ds = []
        f_ds.append((f_start + f_stop) / 2)
        M_s=[]
        A_s=[]
        M_s.append(m_s0)
        A_s.append(a_s0)
        M_sr = []
        A_sr = []
        M_si = []
        A_si = []
        M_sr.append(m_s0.real)
        A_sr.append(a_s0.real)
        M_si.append(m_s0.imag)
        A_si.append(a_s0.imag)
        Time=[]
        Time.append(0)

        for i in range(round(steps)):
            f_d=(f_start+f_stop)/2+(f_stop-f_start)/2*np.sin(f_P_d*2*np.pi*Time[-1])
            omega_D = f_d * 1e9 * 2 * np.pi
            delta_a = self.omega_a - omega_D
            delta_m = self.omega_m - omega_D
            S_a = -1j * delta_a - self.ka / 2
            S_m = -1j * delta_m - self.km / 2
            da=S_a*A_s[i]-1j*self.g_ma*M_s[i]+np.sqrt(self.kaed) * np.sqrt(P_d / (hbar * omega_D))
            dm=S_m*M_s[i]-1j*self.g_ma*A_s[i]-1j*self.K*(2*np.abs(M_s[i])**2+1)*M_s[i]

            anext = da * interval + A_s[i]
            mnext = dm * interval + M_s[i]

            A_s.append(anext)
            M_s.append(mnext)
            M_sr.append(mnext.real)
            A_sr.append(anext.real)
            M_si.append(mnext.imag)
            A_si.append(anext.imag)
            f_ds.append(f_d)
            Time.append(interval*(i+1))

        return M_sr,M_si,A_sr,A_si,Time,f_ds
class Combination():
    def __init__(self):
        pass

    def Array_two_1D(self,a,b):
        t=np.concatenate((a,b))
        return t

    def Array_three_1D(self,a,b,c):
        t=np.concatenate((a,b,c))
        return t