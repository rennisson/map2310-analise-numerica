import numpy as np
import sympy as smp
import scipy as sp
import matplotlib.pyplot as plt


def main():
    x = smp.symbols('x', real=True)
    f = smp.exp(-x)
    text = input("Digite dois inteiros positivos para m e n: ")
    m, n = text.split(", ")
    m = int(m)
    n = int(n)
    N = m + n + 1
    n += 1
    q = np.zeros(N)  # Initializing q_i coefficients with zero
    p = np.zeros(N)  # Initializing p_i coefficients with zero

    # STEP 1
    a = np.zeros(N)  # Initializing the maclaurin coefficients with zero
    b = np.zeros((N, N + 1))  # Initializing the matrix B (shape=(N+1, N+2)) with zeros

    # STEP 2
    for i in range(N):
        dfdx = smp.diff(f, x, i)
        a[i] = dfdx.subs([(x, 0)]).evalf() / smp.factorial(i)
    print(f'{a=}')

    # # # STEP 3
    # # q[0] = 1
    # # p[0] = a[0]
    #
    # # STEP 4
    # for i in range(1, N+1):
    #     # STEP 5
    #     for j in range(1, i):
    #         if j <= n:
    #             b[i-1][j-1] = 0.
    #
    #     # STEP 6
    #     if i <= n:
    #         b[i-1][i-1] = 1
    #
    #     # STEP 7
    #     for j in range(i+1, N+1):
    #         b[i-1][j-1] = 0
    #
    #     # STEP 8
    #     for j in range(1, i+1):
    #         if j <= m:
    #             b[i-1][n + j - 1] = -1 * a[i - j - 1]
    #
    #     # STEP 9
    #     for j in range(n + i + 1, N+1):
    #         b[i - 1][j - 1] = 0
    #
    #     # STEP 10
    #     b[i-1][N] = a[i-1]

    for i in range(1, N + 1):
        for j in range(1, i):
            if j <= n:
                b[i - 1][j - 1] = 0

        if i <= n:
            b[i - 1][i - 1] = 1

        for j in range(i + 1, N + 1):
            b[i - 1][j - 1] = 0

        for j in range(1, i + 1):
            if j <= m:
                b[i - 1][n + j - 1] = -1 * a[i - j - 1]

        for j in range(n + i + 1, N + 1):
            b[i - 1][j - 1] = 0

        b[i - 1][N] = a[i - 1]

    mat_1 = np.array_split(b, [N], axis=1)[0]
    mat_2 = np.array_split(b, [N], axis=1)[1]
    res = np.linalg.solve(mat_1, mat_2)
    print(res)
    print("\n")

    p, q = sp.interpolate.pade(a, 2)
    # q = np.flip(res[m + 1:].reshape(m))
    # p = np.flip(res[:m + 1].reshape(n))
    p = smp.Poly.from_list(p, x)
    q = smp.Poly.from_list(q, x)


    # # STEP 11
    # for i in range(n + 1, N):
    #     # STEP 12
    #     max_element = 0
    #     k = 0
    #     for j in range(i, N):
    #         if np.abs(b[j][i]) > max_element:
    #             max_element = np.abs(b[j][i])
    #         for pivot in range(i, N+1):
    #             if np.abs(b[pivot][i]) == np.abs(max_element):
    #                 k = pivot
    #                 break
    #
    #     # STEP 13
    #     if b[k][i] == 0:
    #         print("The system is singular")
    #         break
    #
    #     # STEP 14
    #     if k != i:  # (Interchange row i and row k.)
    #         for j in range(i, N + 2):
    #             b[i][j], b[k][j] = b[k][j], b[i][j]
    #
    #     # STEP 15
    #     for j in range(i + 1, N):  # Perfom elimination
    #         # STEP 16
    #         try:
    #             xm = b[j][i] / b[i][i]
    #         except ZeroDivisionError as e:
    #             print(e)
    #             return
    #
    #         # STEP 17
    #         for k in range(i + 1, N + 2):
    #             b[j][k] = b[j][k] - xm * b[i][k]
    #
    #         # STEP 18
    #         b[j][i] = 0

    # # STEP 19
    # if b[N-1][N] == 0:
    #     print("The system is singular")
    #     return
    #
    # # STEP 20
    # if m > 0:  # Start backward substitution
    #     try:
    #         q[m] = b[N-1][N] / b[N-1][N]
    #     except ZeroDivisionError as e:
    #         print(e)
    #         return
    #
    # # STEP 21
    # for i in range(N - 1, n + 2, -1):
    #     sum = 0
    #     for j in range(i + 1, N+1):
    #         sum += b[i][j] * q[j - n]
    #
    #     try:
    #         q[i - n] = (b[i][N+1] - sum) / b[i][i]
    #     except ZeroDivisionError as e:
    #         print(e)
    #         return
    #
    # # STEP 22
    # for i in range(n, 0, -1):
    #     sum = 0
    #     for j in range(n+1, N+1):
    #         sum += b[i][j] * q[j - n]
    #
    #     p[i] = b[i][N] - sum

    # STEP 23
    print(f'{q=}')
    print(f'{p=}\n')


main()
