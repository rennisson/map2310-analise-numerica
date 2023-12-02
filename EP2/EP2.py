import numpy as np
import sympy as smp
import scipy as sp
import matplotlib.pyplot as plt


def main():
    x = smp.symbols('x', real=True)
    #f = smp.exp(-x)
    f = smp.exp(x) * smp.sin(x) * smp.cos(x)
    text = input("Digite dois inteiros positivos para m e n: ")
    m, n = text.split(", ")
    m, n = int(m), int(n)
    N = m + n + 1
    n += 1
    p = np.zeros(n)  # Initializing p_i coefficients with zero
    q = np.zeros(m)  # Initializing q_i coefficients with zero

    # STEP 1
    a = np.zeros(N)  # Initializing the maclaurin coefficients with zero
    b = np.zeros((N, N + 1))  # Initializing the matrix B (shape=(N+1, N+2)) with zeros

    # STEP 2
    for i in range(N):
        dfdx = smp.diff(f, x, i)
        a[i] = dfdx.subs([(x, 0)]).evalf() / smp.factorial(i)
    print(f'{a=}')

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

    print(f'{res=}')
    #### p, q = sp.interpolate.pade(a, 2)
    p = np.flip(res[:n].reshape(1, n))
    q = np.flip(res[n:].reshape(1, m))
    p = np.append(np.array(p), [])
    q = np.append(np.array(q), [1.])
    p = smp.Poly.from_list(p, x)
    q = smp.Poly.from_list(q, x)
    r = p / q

    # print(f'{p=}')
    # print(f'{q=}')

    plt.figure()
    plt.subplot(211)
    x1 = np.linspace(0., 1.0, 300)
    print(f'{x1.size} = tamanho de x1')

    # ERRO Q MEDIO, E CALCULO DE F(X)
    f_values = []
    r_values = []
    max, ponto_max, erro_quadratico = 0, 0, 0
    for value in x1:
        y1, y2 = f.subs(x, value), r.subs(x, value)
        f_values.append(y1)
        r_values.append(y2)
        erro_quadratico = np.sqrt(float((value - y1)**2) + float((value - y2)**2))
        if erro_quadratico >= max:
            ponto_max = value

    f_values = np.array(f_values)
    r_values = np.array(r_values)

    # STEP 23
    print(f'{p=}')
    print(f'{q=}')
    print(f'{r=}\n')

    print(f'{erro_quadratico=}\n{ponto_max=}')
    plt.plot(x1, f_values, 'k', x1, r_values, 'g--')
    plt.show()


main()
