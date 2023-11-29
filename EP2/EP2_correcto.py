import numpy as np
import sympy as smp
import scipy as sp
import matplotlib.pyplot as plt


def pade_approximation(N, m):
    N += 1
    n = N - m
    x = smp.symbols('x', real=True)
    f = smp.exp(-x)

    a = np.zeros(N)
    for i in range(0, N):
        df = smp.diff(f, x, i)
        val = df.subs([(x, 0)]).evalf()
        a[i] = val / smp.factorial(i)

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
            if (j <= m): b[i - 1][n + j - 1] = -1 * a[i - j - 1]

        for j in range(n + i + 1, N + 1): b[i - 1][j - 1] = 0

        b[i - 1][N] = a[i - 1]

    mat_1 = np.array_split(b, [N], axis=1)[0]
    mat_2 = np.array_split(b, [N], axis=1)[1]
    res = np.linalg.solve(mat_1, mat_2)

    p, q = sp.interpolate.pade(a, 2)
    print(f'{a=}')
    # q = np.flip(res[m + 1:].reshape(m))
    # p = np.flip(res[:m + 1].reshape(n))
    p = smp.Poly.from_list(p, x)
    q = smp.Poly.from_list(q, x)
    print(f'{p=}')
    print(f'{q=}')

    return 1


pade_approximation(5, 2)