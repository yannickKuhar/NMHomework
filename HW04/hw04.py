import numpy as np
import matplotlib.pyplot as plt


def f(t, v, l=1):
    """
    :param t: Curent time.
    :param v: Curent value fot velocity calculation.
    :param l: Length of string. Default ste to 1 as per instructions.
    """
    g = 9.80665
    return np.array([v, -g/l * np.sin(t)])


def f2(t, l, v):
    g = 9.80665
    return np.array([v, -g/l * t])


def nihalo(l, t, theta0, dtheta0, n):
    h = t / n

    # Initial values of velocities.
    v_vals = np.zeros(n)
    v_vals[0] = dtheta0

    # Initial values of angles.
    a_vals = np.zeros(n)
    a_vals[0] = theta0

    # Runge–Kutta method of the 4th order.
    # Source: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    # Source: https://www.geeksforgeeks.org/runge-kutta-4th-order-method-solve-differential-equation/
    for i in range(0, n - 1):
        k1_v, k1_a = h * f(v_vals[i], a_vals[i])
        k2_v, k2_a = h * f(v_vals[i] + 0.5 * k1_v, a_vals[i] + 0.5 * k1_a)
        k3_v, k3_a = h * f(v_vals[i] + 0.5 * k2_v, a_vals[i] + 0.5 * k2_a)
        k4_v, k4_a = h * f(v_vals[i] + k3_v, a_vals[i] + k3_a)

        v_vals[i + 1] = v_vals[i] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
        a_vals[i + 1] = a_vals[i] + (k1_a + 2 * k2_a + 2 * k3_a + k4_a) / 6.0

    return v_vals, a_vals


def main():
    l = 1
    t = 10
    n = 1000
    theta0 = np.deg2rad(30)
    dtheta0 = np.radians(1)

    times = np.linspace(0, t, n)

    a, v = nihalo(l, t, theta0, dtheta0, n)

    plot_pendulum(times, a)
    # plot_pendulum(times, v)


def plot_pendulum(times, solutions):
    plt.plot(times, solutions)
    plt.xlabel("Times")
    plt.ylabel("Solutions")
    plt.show()


if __name__ == "__main__":
    main()