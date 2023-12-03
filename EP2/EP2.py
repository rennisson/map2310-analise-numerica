import numpy as np
import sympy as smp
import scipy as sp
import matplotlib.pyplot as plt


def main():
    x = smp.symbols('x', real=True)
    # f = smp.exp(x) * smp.sin(x) * smp.cos(x)
    f = smp.exp(-x) * smp.sin(x) * smp.cos(x)**2
    text = input("Digite dois inteiros positivos para m e n: ")
    m, n = text.split(", ")
    m, n = int(m), int(n)
    N = m + n + 1
    n += 1

    # STEP 1
    a = np.zeros(N)  # Initializing the maclaurin coefficients with zero
    b = np.zeros((N, N + 1))  # Initializing the matrix B (shape=(N+1, N+2)) with zeros

    # STEP 2
    # Coeficientes de MacLaurin
    for i in range(N):
        dfdx = smp.diff(f, x, i)
        a[i] = dfdx.subs([(x, 0)]).evalf() / smp.factorial(i)

    # Cria a matriz B
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

    # Resolução do sistema linear
    mat_1 = np.array_split(b, [N], axis=1)[0]
    mat_2 = np.array_split(b, [N], axis=1)[1]
    res = np.linalg.solve(mat_1, mat_2)

    # Arrumando os arrays para os coeficientes de P e Q
    p, q = np.flip(res[:n].reshape(1, n)), np.flip(res[n:].reshape(1, m))
    p, q = np.append(np.array(p), []), np.append(np.array(q), [1.])
    p, q = smp.Poly.from_list(p, x), smp.Poly.from_list(q, x)
    r = p / q

    maclaurin = smp.Poly.from_list(np.flip(a), x)
    f_values = []
    r_values = []
    maclaurin_values = []
    x1 = np.linspace(0., 1.0, 300)
    # Variaveis:
    # norma do erro l^2, ponto do erro l^2, norma do erro maximo, ponto do erro maximo
    max_err_q, ponto_max_q, err_max, ponto_max, mac_err_max, mac_err_q, mac_ponto_max_q,\
        mac_ponto_max = 0, 0, 0, 0, 0, 0, 0, 0

    for value in x1:
        y1, y2, y3 = f.subs(x, value), r.subs(x, value), maclaurin.subs(x, value) # valor das funcoes no ponto 'value'
        # Adiciona o valores das funcoes em seus respectivos arrays
        f_values.append(y1)
        r_values.append(y2)
        maclaurin_values.append(y3)
        # Calcula a norma de l^2
        aux = np.sqrt(float((value - y1)**2) + float((value - y2)**2))
        if aux >= max_err_q:
            max_err_q = aux
            ponto_max_q = value

        aux = np.sqrt(float((value - y3) ** 2) + float((value - y3) ** 2))
        if aux >= mac_err_q:
            mac_err_q = aux
            mac_ponto_max_q = value

        # Calcula a norme do erro maximo
        aux = np.abs(y1 - y2)
        if aux >= err_max:
            err_max = aux
            ponto_max = value

        aux = np.abs(y1 - y3)
        if aux >= mac_err_max:
            mac_err_max = aux
            mac_ponto_max = value

    f_values = np.array(f_values)
    r_values = np.array(r_values)
    maclaurin_values = np.array(maclaurin_values)

    # STEP 23
    print(f'{a=}')
    print(f'{r=}\n')

    print(f'{f=}')
    print(f'{max_err_q=}\n{ponto_max_q=}')
    print(f'{err_max=}\n{ponto_max=}\n')
    print(f'{maclaurin=}')
    print(f'{mac_err_q=}\n{mac_ponto_max_q=}')
    print(f'{mac_err_max=}\n{mac_ponto_max=}')

    plt.figure()
    plt.plot(x1, f_values, 'k', x1, r_values, 'g--', x1, maclaurin_values, 'r--')
    plt.show()


main()
