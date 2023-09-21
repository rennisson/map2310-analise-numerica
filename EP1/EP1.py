import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def main():

    np.random.seed(13687175)
    # rg8.dat \ Rolf Girsberger RG 8 airfoil
    extradorso1, intradorso1 = read_data("rg8.dat", "r")

    # arad13.dat \ Aeronautical Research Association / Bocci - Dowty Rotol ARA - D13 % thick propeller airfoil
    extradorso2, intradorso2 = read_data("arad13.dat", "r")

    # drgnfly.dat \ Dragonfly Canard airfoil
    extradorso3, intradorso3 = read_data("drgnfly.dat", "r")

    x_theta = dist_theta(size=500)
    x_theta.sort()

    # Catching the values of 'x'
    x1 = catch_values(extradorso1)
    x2 = catch_values(intradorso1)

    x3 = catch_values(extradorso2)
    x4 = catch_values(intradorso2)

    x5 = catch_values(extradorso3)
    x6 = catch_values(intradorso3)

    # Catching the coefficients 'a' (a = f(x))
    a1 = catch_values(extradorso1, 1)
    a2 = catch_values(intradorso1, 1)

    a3 = catch_values(extradorso2, 1)
    a4 = catch_values(intradorso2, 1)

    a5 = catch_values(extradorso3, 1)
    a6 = catch_values(intradorso3, 1)

    # Finding the Natural Splines
    # Step 1 - Calculating the differences between x_j+1 and x_j
    h1 = diff_x(x1)
    h2 = diff_x(x2)

    h3 = diff_x(x3)
    h4 = diff_x(x4)

    h5 = diff_x(x5)
    h6 = diff_x(x6)

    # Step 2 - Calculating the values of alpha
    alpha1 = find_alpha(x1, h1, a1)
    alpha2 = find_alpha(x2, h2, a2)

    alpha3 = find_alpha(x3, h3, a3)
    alpha4 = find_alpha(x4, h4, a4)

    alpha5 = find_alpha(x5, h5, a5)
    alpha6 = find_alpha(x6, h6, a6)

    # Step 3, 4 and 5 contained here
    l1, u1, z1 = create_l_u_z(x1, h1, alpha1)
    l2, u2, z2 = create_l_u_z(x2, h2, alpha2)

    l3, u3, z3 = create_l_u_z(x3, h3, alpha3)
    l4, u4, z4 = create_l_u_z(x4, h4, alpha4)

    l5, u5, z5 = create_l_u_z(x5, h5, alpha5)
    l6, u6, z6 = create_l_u_z(x6, h6, alpha6)

    # Step 6
    b1, c1, d1 = coeff_b_c_d(x1, h1, a1, u1, z1)
    b2, c2, d2 = coeff_b_c_d(x2, h2, a2, u2, z2)
    print_polynomials(x1, a1, b1, c1, d1)
    print_polynomials(x2, a2, b2, c2, d2)

    b3, c3, d3 = coeff_b_c_d(x3, h3, a3, u3, z3)
    b4, c4, d4 = coeff_b_c_d(x4, h4, a4, u4, z4)
    print_polynomials(x3, a3, b3, c3, d3)
    print_polynomials(x4, a4, b4, c4, d4)

    b5, c5, d5 = coeff_b_c_d(x5, h5, a5, u5, z5)
    b6, c6, d6 = coeff_b_c_d(x6, h6, a6, u6, z6)
    print_polynomials(x5, a5, b5, c5, d5)
    print_polynomials(x6, a6, b6, c6, d6)

    # Interpolation with the points from (0.5 * (1 - np.cos(theta)))
    values1 = interpolate(x1, x_theta, a1, b1, c1, d1)
    values2 = interpolate(x2, x_theta, a2, b2, c2, d2)

    values3 = interpolate(x3, x_theta, a3, b3, c3, d3)
    values4 = interpolate(x4, x_theta, a4, b4, c4, d4)

    values5 = interpolate(x5, x_theta, a5, b5, c5, d5)
    values6 = interpolate(x6, x_theta, a6, b6, c6, d6)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,7.5))

    max_dist = 0
    pontos_max = [(0,0),(0,0)]
    for i in range(len(x_theta)):
        distance = np.sqrt( (x_theta[i] - x_theta[i])**2 + (values1[i] - values2[i])**2)
        if distance >= max_dist:
            max_dist = distance
            pontos_max = [(x_theta[i], values1[i]), (x_theta[i], values2[i])]

    print(f'{max_dist=}\n {pontos_max=}')

    ax1.set_title('Rolf Girsberger RG 8 airfoil')
    ax1.plot(x_theta, values1, label='cubic spline', color='blue', linestyle='--', marker='*', markersize=3)
    #ax1.plot(extradorso1[:, 0], extradorso1[:, 1], label='data', color='red', linestyle='--', marker='o', markersize=4)
    ax1.plot(x_theta, values2, color='blue', linestyle='--', marker='*', markersize=3)
    #ax1.plot(intradorso1[:, 0], intradorso1[:, 1], color='red', linestyle='--', marker='o', markersize=4)
    ax1.grid(visible=True, linestyle='--')

    ax2.set_title('ARA - D13% thick propeller airfoil')
    ax2.plot(x_theta, values3, color='blue', linestyle='--', marker='*', markersize=3)
    ax2.plot(extradorso2[:, 0], extradorso2[:, 1], color='red', linestyle='--', marker='o', markersize=4)
    ax2.plot(x_theta, values4, color='blue', linestyle='--', marker='*', markersize=3)
    ax2.plot(intradorso2[:, 0], intradorso2[:, 1], color='red', linestyle='--', marker='o', markersize=4)
    ax2.grid(visible=True, linestyle='--')

    ax3.set_title('Dragonfly Canard airfoil')
    ax3.plot(x_theta, values5, color='blue', linestyle='--', marker='*', markersize=3)
    ax3.plot(extradorso3[:, 0], extradorso3[:, 1], color='red', linestyle='--', marker='o', markersize=4)
    ax3.plot(x_theta, values6, color='blue', linestyle='--', marker='*', markersize=3)
    ax3.plot(intradorso3[:, 0], intradorso3[:, 1], color='red', linestyle='--', marker='o', markersize=4)
    ax3.grid(visible=True, linestyle='--')

    blue_lines = mlines.Line2D([], [], color='blue', marker='*', markersize=10, label='Cubic splines')
    red_lines = mlines.Line2D([], [], color='red', marker='o', markersize=8, label='Real data')
    ax1.legend(handles=[blue_lines, red_lines])
    plt.tight_layout()
    plt.show()
    print()


def read_data(name, mode):
    with open(name, mode) as file:
        dados = []
        meio, i = 0, 0

        for line in file.readlines():
            if i == 0: name = line
            if i > 2:
                data = line.split()
                if not data:
                    meio = i
                else:
                    dados.append((float(data[0]), float(data[1])))
            i += 1

        extradorso = np.array(dados[0:meio - 3])
        intradorso = np.array(dados[meio - 3:])

    return extradorso, intradorso


def dist_theta(size=500):
    """
    :return: an array with values according to the distribuition (0.5 * (1 - np.cos(theta)))
    """
    theta = np.random.random(size=size) * np.pi
    return np.array(0.5 * (1 - np.cos(theta)))


def interpolate(x, x_theta, a, b, c, d):
    # Interpolation with the points from 'dist_theta'
    values = []
    for x_i in x_theta:
        for i in range(len(x) - 1):
            if x[i] <= x_i <= x[i + 1]:
                h = (x_i - x[i])
                values.append(a[i] + b[i] * h + c[i] * h ** 2 + d[i] * h ** 3)

    return np.array(values)


def catch_values(x, index=0):
    """
    :param x: array containing tuples of values
    :return: an array containing only the values in column 'index'
    """
    return x[:, index]


def diff_x(x):
    """
    :param x: array of integers
    :return: an array containing the differences between x_j+1 and x_j,
                for all j = 0, ..., len(x)-1.
    """
    h = []
    for i in range(len(x)-1):
        h.append(x[i + 1] - x[i])
    return np.array(h)


def find_alpha(x, h, a):
    """
    :param x: array containing the values of 'x'
    :param h: array containing the values of 'h'
    :param a: array containing the coefficients 'a'
    :return: an array with the values of 'alpha', according to ((3 / h[i]) * (a[i+1] - a[i]) - (3 / h[i-1]) * (a[i] - a[i-1])),
                for all i = indexes of 'x'
    """
    alpha = []
    for i in range(1, len(x)-1):
        alpha.append((3 / h[i]) * (a[i+1] - a[i]) - (3 / h[i-1]) * (a[i] - a[i-1]))
    return np.array(alpha)


def create_l_u_z(x, h, alpha):
    # Step 3 - Set l[0]=1, u[0]=0 and z[0]=0
    l, u, z = [1], [0], [0]

    # Step 4 - Calculating all values for l[i], u[i] and z[i]
    for i in range(1, len(x) - 2):
        l.append(2 * (x[i + 1] - x[i - 1] - h[i - 1] * u[i - 1]))
        u.append(h[i] / l[i])
        z.append((alpha[i] - h[i - 1] * z[i - 1]) / l[i])

    # Step 5 - Set l[n-1], u[n-1] and z[n-1]
    l.append(1)
    z.append(0)
    return l, u, z


def coeff_b_c_d(x, h, a, u, z):
    # Creating arrays for coefficients 'b', 'c' and 'd'
    b = np.zeros(shape=(len(x),))
    c = np.zeros(shape=(len(x),))
    d = np.zeros(shape=(len(x),))

    # Setting up the last value for 'c' equals 0. It means, the nth coefficient 'c' is ZERO.
    c[len(x) - 1] = 0

    # Step 6 - Calculating all the values for coefficients 'b', 'c' and 'd'
    for j in range(len(x) - 3, -1, -1):
        c[j] = z[j] - u[j] * c[j + 1]
        b[j] = ((a[j + 1] - a[j]) / h[j]) - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / 3 * h[j]

    return b, c, d


def print_polynomials(x, a, b, c, d):
    print(f'Lista dos polinômios interpoladores S(x), para x = 0, ..., {len(x)-1}')
    for i in range(len(x)):
        print(f'S({i}) = {a[i]:.7f} + '
              f'{b[i]:.7f}*(x - {x[i]:.7f}) +'
              f'{c[i]:.7f}*(x - {x[i]:.7f})^2 +'
              f'{d[i]:.7f}*(x - {x[i]:.7f})^3')
    print('\n')


main()