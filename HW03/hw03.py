import numpy as np
from numpy import sin

import scipy.integrate as integrate


def sin_x(x):
    """
    :param x: The value of the independent variable.
    :return: The value of the function sin(x) / x.
    """
    return sin(x) / x


def int_sin_x():
    """
    :return: The integral of function sin(x) / x calculated by the scipy library.
    """
    return integrate.quad(lambda x: sin_x(x), 0, 5)[0]


def sestavljeno_parvilo(a, b, n):
    """
    This function calculates the weight and roots by the method mentioned in the instructions.
    :param a: Lower bound.
    :param b: Upper bound.
    :param n: Number of intervals.
    :return: List of weights and list of roots.
    """
    h = (b - a) / n
    w = h * np.ones(n)
    x = np.linspace(a + h / 2, b - h / 2, n)
    return w, x


def integral(fun, a, b, n):
    """
    :param fun: The function to be integrated.
    :param a: Lower bound.
    :param b: Upper bound.
    :param n: Number of intervals.
    :return: The integral of function `fun` on interval `[a, b]` calculated by our method.
    """
    w, x = sestavljeno_parvilo(a, b, n)
    return np.dot(w, np.array(list(map(fun, x))))


def calc_steps(a, b, min_err):
    """
    Calculates the number of intervals needed for the integral to fall below the acceptable error.
    :param a: Lower bound of intervals.
    :param b: Upper bound of intervals.
    :param min_err: Acceptable error.
    :return:
    """
    original = int_sin_x()
    err = abs(original - integral(sin_x, 0, 5, a))

    while err > min_err:
        mid = int((a + b) / 2)
        n_err = abs(original - integral(sin_x, 0, 5, mid))

        if n_err < err:
            a = mid
        else:
            b = mid

        if n_err < min_err:
            break

    return original, integral(sin_x, 0, 5, a), a, abs(original - integral(sin_x, 0, 5, a))


def main():
    print(calc_steps(10000, 40000, 1e-10))


if __name__ == '__main__':
    main()