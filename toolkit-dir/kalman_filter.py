from utils.ex4_utils import kalman_step
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp

np.random.seed(8)


def compute_input_matrices(q, F, L):
    T = sp.symbols('T')
    F = sp.Matrix(F)
    Fi = (sp.exp(F * T)).subs(T, 1)

    L = sp.Matrix(L)
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    Q = Q.subs(T, 1)

    Fi = np.array(Fi, dtype="float")
    Q = np.array(Q, dtype="float")

    return Fi, Q


def get_model_matrices(model_, r):
    if model_ == "RW":
        F_matrix = [[0, 0],
                    [0, 0]]

        L_matrix = [[1, 0],
                    [0, 1]]

        H = np.array(
            [
                [1, 0],
                [0, 1]
            ],
            dtype="float")

        R = r * np.array(
            [
                [1, 0],
                [0, 1]
            ],
            dtype="float")

    elif model_ == "NCV":
        F_matrix = [[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]]

        L_matrix = [[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]]

        H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ],
            dtype="float")

        R = r * np.array(
            [
                [1, 0],
                [0, 1]
            ],
            dtype="float")

    elif model_ == "NCA":
        F_matrix = [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]

        L_matrix = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ]

        H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ],
            dtype="float")

        R = r * np.array(
            [
                [1, 0],
                [0, 1]
            ],
            dtype="float")
    else:
        raise NotImplementedError

    return F_matrix, L_matrix, H, R


def plot_curves(q, r, x_, y_, model_, ax, title):
    F, L, H, R = get_model_matrices(model_, r)
    Fi, Q = compute_input_matrices(q, F, L)

    ax.plot(x_, y_, c="red", linewidth=1)

    sx = np.zeros((x_.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y_.size, 1), dtype=np.float32).flatten()

    sx[0] = x_[0]
    sy[0] = y_[0]

    state = np.zeros((Fi.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x_[0]
    state[1] = y_[0]
    covariance = np.eye(Fi.shape[0], dtype=np.float32)

    for j in range(1, x_.size):
        state, covariance, _, _ = kalman_step(
            Fi, H, Q, R,
            np.reshape(np.array([x_[j], y_[j]]), (-1, 1)),
            np.reshape(state, (-1, 1)),
            covariance
        )
        sx[j] = state[0]
        sy[j] = state[1]

    ax.plot(sx, sy, c="blue", linewidth=1)
    ax.title.set_text(title)


if __name__ == '__main__':
    N = 40

    curves = [
        (
            np.cos(np.linspace(5 * math.pi, 0, N)) * np.linspace(5 * math.pi, 0, N),
            np.sin(np.linspace(5 * math.pi, 0, N)) * np.linspace(5 * math.pi, 0, N)
        ),
        (
            np.array([4, 6, 7, 7, 6, 4, 3, 3, 5, 7, 8, 10, 12, 13, 13]),
            np.array([-4, -3, -1, 1, 3, 4, 6, 8, 9, 8, 6, 5, 6, 8, 10])),
        (
            np.array([1, 3, 4, 4, 4, 3, 1, -1, -3, -5, -6, -6, -6, -5, -3, -1]),
            np.array([2, 2, 0, -2, -4, -6, -7, -7, -7, -6, -4, -2, 0, 2, 2, 2])
        )]

    for index, (x, y) in enumerate(curves):
        fig1, ((ax1_11, ax1_12, ax1_14, ax1_15),
               (ax1_21, ax1_22, ax1_24, ax1_25),
               (ax1_31, ax1_32, ax1_34, ax1_35)) = plt.subplots(3, 4, figsize=(12, 9))

        model = "RW"
        plot_curves(100, 1, x, y, model, ax1_11, model + ": q = 100, r = 1")
        plot_curves(5, 1, x, y, model, ax1_12, model + ": q = 5, r = 1")
        # plot_curves(1, 1, x, y, model, ax1_13, model + ": q = 1, r = 1")
        plot_curves(1, 5, x, y, model, ax1_14, model + ": q = 1, r = 5")
        plot_curves(1, 100, x, y, model, ax1_15, model + ": q = 1, r = 100")

        model = "NCV"
        plot_curves(100, 1, x, y, model, ax1_21, model + ": q = 100, r = 1")
        plot_curves(5, 1, x, y, model, ax1_22, model + ": q = 5, r = 1")
        # plot_curves(1, 1, x, y, model, ax1_23, model + ": q = 1, r = 1")
        plot_curves(1, 5, x, y, model, ax1_24, model + ": q = 1, r = 5")
        plot_curves(1, 100, x, y, model, ax1_25, model + ": q = 1, r = 100")

        model = "NCA"
        plot_curves(100, 1, x, y, model, ax1_31, model + ": q = 100, r = 1")
        plot_curves(5, 1, x, y, model, ax1_32, model + ": q = 5, r = 1")
        # plot_curves(1, 1, x, y, model, ax1_33, model + ": q = 1, r = 1")
        plot_curves(1, 5, x, y, model, ax1_34, model + ": q = 1, r = 5")
        plot_curves(1, 100, x, y, model, ax1_35, model + ": q = 1, r = 100")

        plt.savefig(f'figures/curve{str(index)}.png')
        plt.show()
