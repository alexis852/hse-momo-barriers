import numpy as np


class LassoLogBarrierOracle:
    """
    Oracle for barrier function for LASSO-regression
    """

    def __init__(self, A, b, reg_coef, t):
        self.A = A
        self.b = b
        self.reg_coef = reg_coef
        self.t = t
        self.tATA = None

    def func(self, x, u):
        return self.t * (0.5 * np.linalg.norm(self.A @ x - self.b) ** 2 + self.reg_coef * sum(u)) - sum(np.log(u ** 2 - x ** 2))

    def original_func(self, x, u):
        return 0.5 * np.linalg.norm(self.A @ x - self.b) ** 2 + self.reg_coef * sum(u)

    def grad_x(self, x, u):
        return self.t * self.A.T @ (self.A @ x - self.b) + 2 * x / (u ** 2 - x ** 2)

    def grad_u(self, x, u):
        return np.full(u.shape[0], self.t * self.reg_coef) - 2 * u / (u ** 2 - x ** 2)

    def reduced_hess(self, x, u):
        """
        Computes matrix t * A^T A + C - D^2 C^{-1}
        """
        if self.tATA is None:
            self.tATA = self.t * self.A.T @ self.A
        diag_term = np.diag(2 / (u ** 2 + x ** 2))
        return self.tATA + diag_term

    def func_directional(self, x, u, d_x, d_u, alpha):
        """
        Computes phi(alpha) = f(x + alpha * d).
        """
        return np.squeeze(self.func(x + alpha * d_x, u + alpha * d_u))


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    mu = min(1, regcoef / np.linalg.norm(ATAx_b, np.inf)) * Ax_b
    return 0.5 * Ax_b.T @ Ax_b + regcoef * np.linalg.norm(x, 1) + \
           0.5 * mu.T @ mu + b @ mu
