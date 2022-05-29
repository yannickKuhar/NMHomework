import numpy as np
from numpy import sin

import scipy.integrate as integrate


def sin_x(x):
    return sin(x) / x


def int_sin_x():
    return integrate.quad(lambda x: sin_x(x), 0, 5)[0]


def sestavljeno_parvilo(a, b, n):
    h = (b - a) / n
    w = h * np.ones(n)
    x = np.linspace(a + h / 2, b - h / 2, n)
    return w, x


def integral(fun, a, b, n):
    w, x = sestavljeno_parvilo(a, b, n)
    return np.dot(w, np.array(list(map(fun, x))))


def calc_steps():
    min_err = 1e-10

    original = int_sin_x()

    a = 10000
    b = 40000

    err = abs(original - integral(sin_x, 0, 5, a))

    while err > min_err:
        mid = int((a + b) / 2)
        n_err = abs(original - integral(sin_x, 0, 5, mid))

        if n_err < err:
            a = mid

        if n_err < min_err:
            break

    return original, integral(sin_x, 0, 5, a), a, abs(original - integral(sin_x, 0, 5, a))


def main():
    print(calc_steps())


if __name__ == '__main__':
    main()