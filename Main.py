import numpy as np
import GraphicMain
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import PyQt5.uic.pyuic

def Simuate(L, T, h, tau, mu_in, mu, H0, H1, H2, lam, p, Cp, Ct, T0, T1, T2, D, y, n, C0, C1, C2, m, i1):
    def Tr_plus(u):
        return (-p * Cp * u + abs(p * Cp * u)) / 2.0

    def Tr_minus(u):
        return (-p * Cp * u - abs(p * Cp * u)) / 2.0

    def u(k, h_i, h_pi, t_i, t_pi, mu):
        return k * ((h_i - h_pi) / h) + mu * ((t_i - t_pi) / h)

    def Hu(k, h_i, h_pi, t_i, t_pi, mu):
        if abs(h_i - h_pi) / h < i1:
            return k / (m * i1 ** (m - 1.0)) * np.sign(h_i - h_pi) * abs(h_i - h_pi) ** m

        if abs(h_i - h_pi) / h >= i1:
            return k * ((h_i - h_pi) / h - i0) + mu * ((t_i - t_pi) / h)

    def Teta(h, u, lam):
        return lam / (1.0 + (h * abs(p * Cp * u) / (2 * lam)))

    def Ta(tau, n, h, u, lam):
        return tau / n * ((Teta(h, u, lam) / h ** 2) - (Tr_minus(u) / h))

    def Tb(tau, n, h, u, lam):
        return tau / n * ((Teta(h, u, lam) / h ** 2) + (Tr_plus(u) / h))

    def Tc(tau, n, h, u, lam):
        return 1.0 + tau / n * ((2.0 * Teta(h, u, lam) / h ** 2) + ((Tr_plus(u) - Tr_minus(u)) / h))

    def Ha():
        return (tau * k) / (mu_in * h ** 2)

    def Hb():
        return (tau * k) / (mu_in * h ** 2)

    def Hc():
        return 1.0 + 2.0 * ((tau * k) / (mu_in * h ** 2))

    def Hf(Hk_1):
        return - Hk_1

    def kc(C, T):
        #return 0.00000129
        temp = 0
        aCmT = np.array([0.042557151,
                         -0.12857957,
                         0.140291508,
                         -0.0643274,
                         0.00934566864,
                         0.0010339])
        for i in range(0, aCmT.size - 1, 1):
            temp += aCmT[i] * (C / CmT(T)) ** (5 - i)
        temp += aCmT[aCmT.size - 1]
        return temp

    def CmT(T):
        if 0 <= T < 20:
            return 357
        elif 20 <= T < 25:
            return 359
        elif 25 <= T < 30:
            return 360
        elif 30 <= T < 40:
            return 361
        elif 40 <= T < 50:
            return 364
        else:
            return 366

    def Cr_plus(u):
        return (-u + abs(u)) / 2.0

    def Cr_minus(u):
        return (-u - abs(u)) / 2.0

    def Ceta(h, u, D):
        return D / (1 + (h * abs(u) / (2.0 * D)))

    def Ca(tau, n, h, u, D):
        return tau / n * ((Ceta(h, u, D) / h ** 2) - (Cr_minus(u) / h))

    def Cb(tau, n, h, u, D):
        return tau / n * ((Ceta(h, u, D) / h ** 2) + (Cr_plus(u) / h))

    def Cc(tau, n, h, u, D, y):
        return 1 + (tau / n) * ((2 * Ceta(h, u, D) / h ** 2) + ((Cr_plus(u) - Cr_minus(u)) / h) + y)

    def Cf(tau, n, y, Cm):
        return (tau / n) * (Cm * y)
        #return 0

    L = float(L)
    iterations = float(T)
    h = float(h)
    tau = float(tau)
    mu_in = float(mu_in)
    mu = float(mu)
    H0 = float(H0)
    H1 = float(H1)
    H2 = float(H2)
    lam = float(lam)
    p = float(p)
    Cp = float(Cp)
    Ct = float(Ct)
    T0 = float(T0)
    T1 = float(T1)
    T2 = float(T2)
    D = float(D)
    y = float(y)
    n = float(n)
    C0 = float(C0)
    C1 = float(C1)
    C2 = float(C2)
    m = float(m)
    i1 = float(i1)

    i0 = i1 * (m - 1) / m
    N = int(L / h)

    H = np.empty(0)
    T = np.empty(0)
    C = np.empty(0)

    H = np.append(H, H1)
    T = np.append(T, T1)
    C = np.append(C, C1)
    for i in range(0, N-2, 1):
        H = np.append(H, H0)
        T = np.append(T, T0)
        C = np.append(C, C0)
    H = np.append(H, H2)
    T = np.append(T, T2)
    C = np.append(C, C2)
    data_H = np.array(H)
    data_T = np.array(T)
    data_C = np.array(C)

    data_u = np.empty(data_H.size)

    t = 1
    while t <= iterations:
        alpha = np.empty(0)
        alpha = np.append(alpha, 0)
        for i in range(1, N, 1):
            k = kc(C[i], T[i])
            alpha = np.append(alpha, Hb() / (Hc() - Ha() * alpha[i - 1]))
        beta = np.empty(0)
        beta = np.append(beta, H[0])
        for i in range(1, N, 1):
            k = kc(C[i], T[i])
            beta = np.append(beta, (beta[i - 1] * Ha() - Hf(H[i - 1])) / (Hc() - Ha() * alpha[i - 1]))
        Hp = H.copy()
        for i in range(np.size(H) - 2, 0, -1):
            H[i] = alpha[i] * Hp[i + 1] + beta[i]
        data_H = np.vstack([data_H, H])

        alpha_T = np.empty(0)
        alpha_T = np.append(alpha_T, 0)
        alpha_C = np.empty(0)
        alpha_C = np.append(alpha_C, 0)

        Au = np.zeros(1)
        for i in range(1, N, 1):
            k = kc(C[i], T[i])
            Nu = u(k, H[i], H[i - 1], T[i], T[i - 1], mu)
            Au = np.append(Au, Nu)
            alpha_T = np.append(alpha_T, (Tb(tau, Ct, h, Nu, lam) / (Tc(tau, Ct, h, Nu, lam) - Ta(tau, Ct, h, Nu, lam) * alpha_T[i - 1])))
            alpha_C = np.append(alpha_C, (Cb(tau, n, h, Nu, D) / (Cc(tau, n, h, Nu, D, y) - Ca(tau, n, h, Nu, D) * alpha_C[i - 1])))

        data_u = np.vstack([data_u, Au])
        beta_T = np.empty(0)
        beta_T = np.append(beta_T, T[0])
        beta_C = np.empty(0)
        beta_C = np.append(beta_C, C[0])
        for i in range(1, N, 1):
            k = kc(C[i], T[i])
            Nu = u(k, H[i], H[i - 1], T[i], T[i - 1], mu)
            beta_T = np.append(beta_T, ((beta_T[i - 1] * Ta(tau, Ct, h, Nu, lam) + T[i - 1]) / (Tc(tau, Ct, h, Nu, lam) - Ta(tau, Ct, h, Nu, lam) * alpha_T[i - 1])))
            beta_C = np.append(beta_C, ((beta_C[i - 1] * Ca(tau, n, h, Nu, D) + C[i - 1] - Cf(tau, n, y, CmT(T[i]))) / (Cc(tau, n, h, Nu, D, y) - Ca(tau, n, h, Nu, D) * alpha_C[i - 1])))

        PT = T.copy()
        PC = C.copy()
        for i in range(np.size(T) - 2, -1, -1):
            T[i] = alpha_T[i] * PT[i + 1] + beta_T[i]
            C[i] = alpha_C[i] * PC[i + 1] + beta_C[i]
        data_T = np.vstack([data_T, T])
        data_C = np.vstack([data_C, C])
        t += 1

    # ================
    # ================
    # ================

    H = np.empty(0)
    T = np.empty(0)
    C = np.empty(0)

    H = np.append(H, H1)
    T = np.append(T, T1)
    C = np.append(C, C1)
    for i in range(0, N-2, 1):
        H = np.append(H, H0)
        T = np.append(T, T0)
        C = np.append(C, C0)
    H = np.append(H, H2)
    T = np.append(T, T2)
    C = np.append(C, C2)
    Hdata_H = np.array(H)
    Hdata_T = np.array(T)
    Hdata_C = np.array(C)

    Hdata_u = np.empty(Hdata_H.size)

    t = 1
    while t <= iterations:
        alpha = np.empty(0)
        alpha = np.append(alpha, 0)
        for i in range(1, N, 1):
            k = kc(C[i], T[i])
            alpha = np.append(alpha, Hb() / (Hc() - Ha() * alpha[i - 1]))
        beta = np.empty(0)
        beta = np.append(beta, H[0])
        for i in range(1, N, 1):
            k = kc(C[i], T[i])
            beta = np.append(beta, (beta[i - 1] * Ha() - Hf(H[i - 1])) / (Hc() - Ha() * alpha[i - 1]))
        Hp = H.copy()
        for i in range(np.size(H) - 2, 0, -1):
            H[i] = alpha[i] * Hp[i + 1] + beta[i]
        Hdata_H = np.vstack([Hdata_H, H])

        alpha_T = np.empty(0)
        alpha_T = np.append(alpha_T, 0)
        alpha_C = np.empty(0)
        alpha_C = np.append(alpha_C, 0)
        Au = np.zeros(1)
        for i in range(1, N, 1):
            k = kc(C[i], T[i])
            Nu = Hu(k, H[i], H[i - 1], T[i], T[i - 1], mu)
            Au = np.append(Au, Nu)
            alpha_T = np.append(alpha_T, (Tb(tau, Ct, h, Nu, lam) / (Tc(tau, Ct, h, Nu, lam) - Ta(tau, Ct, h, Nu, lam) * alpha_T[i - 1])))
            alpha_C = np.append(alpha_C, (Cb(tau, n, h, Nu, D) / (Cc(tau, n, h, Nu, D, y) - Ca(tau, n, h, Nu, D) * alpha_C[i - 1])))

        Hdata_u = np.vstack([Hdata_u, Au])
        beta_T = np.empty(0)
        beta_T = np.append(beta_T, T[0])
        beta_C = np.empty(0)
        beta_C = np.append(beta_C, C[0])
        for i in range(1, N, 1):
            k = kc(C[i], T[i])
            Nu = Hu(k, H[i], H[i - 1], T[i], T[i - 1], mu)
            beta_T = np.append(beta_T, ((beta_T[i - 1] * Ta(tau, Ct, h, Nu, lam) + T[i - 1]) / (Tc(tau, Ct, h, Nu, lam) - Ta(tau, Ct, h, Nu, lam) * alpha_T[i - 1])))
            beta_C = np.append(beta_C, ((beta_C[i - 1] * Ca(tau, n, h, Nu, D) + C[i - 1] - Cf(tau, n, y, CmT(T[i]))) / (Cc(tau, n, h, Nu, D, y) - Ca(tau, n, h, Nu, D) * alpha_C[i - 1])))
        PT = T.copy()
        PC = C.copy()
        for i in range(np.size(T) - 2, -1, -1):
            T[i] = alpha_T[i] * PT[i + 1] + beta_T[i]
            C[i] = alpha_C[i] * PC[i + 1] + beta_C[i]
        Hdata_T = np.vstack([Hdata_T, T])
        Hdata_C = np.vstack([Hdata_C, C])
        t += 1

    # ================
    # ================
    # ================

    diff_data_u = data_u - Hdata_u
    diff_data_T = data_T - Hdata_T
    diff_data_C = data_C - Hdata_C

    GraphicMain.graph(tau, iterations, data_H=data_H, data_T=data_T, data_C=data_C,
                      Hdata_H=Hdata_H, Hdata_T=Hdata_T, Hdata_C=Hdata_C,
                      diff_data_H=diff_data_u, diff_data_T=diff_data_T, diff_data_C=diff_data_C)

if __name__ == '__main__':
    Simuate(L=100, T=50, h=1.0, tau=1.0,
            mu_in=0.0005, mu=0.000028, H0=5, H1=0, H2=0,
            lam=108, p=1100,
            Cp=4200, Ct=796, T0=1, T1=20, T2=1, D=0.02, y=0.000065, n=0.3, C0=50, C1=350, C2=50,
            m=1.5, i1=30)

    """y = 0.000065
        Cp = 4200
        p = 1100
        n = 0.4
        mu_in = 0.0005
        Ct = 796
        Dt = 0.02
        D0 = 0.02
        mu = 0.000028
        lam = 108
        D = 0.02

        H1 = 20
        H2 = 5
        H0 = 3
        T1 = 10
        T2 = 1
        T0 = 1
        C1 = 100
        C2 = 10
        C0 = 50

        L = 100
        h = 1
        tau = 1


        m = 1.80
        i1 = (H1 - H2) / 10.0

        """