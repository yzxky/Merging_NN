from scipy.optimize import fsolve
import numpy as np

class UnconstrainedOptimal:
    para = [0, 0]
    init_guess = [0, 0, 0, 0, 20]

    @staticmethod
    def func(x, t0, v0, l, beta):
        return [1/2 * x[0] * (t0 ** 2) + x[1] * t0 + x[2] - v0,
                1/6 * x[0] * (t0 ** 3) + 1/2 * x[1] * (t0 ** 2) + x[2] * t0 + x[3],
                1/6 * x[0] * (x[4] ** 3) + 1/2 * x[1] * (x[4] ** 2) + x[2] * x[4] + x[3] - l,
                x[0] * x[4] + x[1],
                beta + 1/2 * (x[0] ** 2) * (x[4] ** 2) + x[0] * x[1] * x[4] + x[0] * x[2]]

    @staticmethod
    def calculate_unconstrained_opt(t0, v0, l, beta):
        UnconstrainedOptimal.init_guess[4] += 20
        root = fsolve(UnconstrainedOptimal.func, UnconstrainedOptimal.init_guess, args=(t0, v0, l, beta))
        UnconstrainedOptimal.init_guess = root
        if root[4] < 0:
            print('error: opt_t < 0')
        return root
