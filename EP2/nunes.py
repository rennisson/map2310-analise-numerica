import numpy as np
import sympy as smp
import scipy as sp
import matplotlib.pyplot as plt

def Pade_Maclaurin(N, m):
    N += 1
    n = N - m
    x = smp.Symbol('x')
    f = smp.exp(-x)
    coef_maclaurin = np.zeros(N)
    for i in range(0, N):
        df = smp.diff(f, x, i)
        print(f'{i=}, {df=}')
        val = df.subs([(x, 0)]).evalf()
        coef_maclaurin[i] = val / smp.factorial(i)

    p = np.zeros(N)
    q = np.zeros(N)
    b = np.zeros((N, N + 1))

    for i in range(1, N + 1):
        for j in range(1, i):
            if (j <= n): b[i - 1][j - 1] = 0

        if (i <= n): b[i - 1][i - 1] = 1

        for j in range(i + 1, N + 1):
            b[i - 1][j - 1] = 0

        for j in range(1, i + 1):
            if (j <= m): b[i - 1][n + j - 1] = -1 * coef_maclaurin[i - j - 1]

        for j in range(n + i + 1, N + 1): b[i - 1][j - 1] = 0

        b[i - 1][N] = coef_maclaurin[i - 1]

    print(b)
    mat_1 = np.array_split(b, [N], axis=1)[0]
    mat_2 = np.array_split(b, [N], axis=1)[1]
    res = np.linalg.solve(mat_1, mat_2)
    print(f'{res=}')
    print("\n")

    p, q = sp.interpolate.pade(coef_maclaurin, 2)
    print(f'{coef_maclaurin=}')
    print(f'{p=}')
    print(f'{q=}')
    # q = np.flip(res[m + 1:].reshape(m))
    # p = np.flip(res[:m + 1].reshape(n))
    p = smp.Poly.from_list(p, x)
    q = smp.Poly.from_list(q, x)
    print(f'{p=}')
    print(f'{q=}')

    return 1


Pade_Maclaurin(5, 2)