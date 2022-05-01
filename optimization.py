from collections import defaultdict
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from datetime import datetime
import oracles


def is_finite(a):
    return a is not None and np.isfinite(a).all()


def backtracking(oracle, x_k, u_k, d_k_x, d_k_u, grad_dir, alpha_0, c1):
    alpha = alpha_0
    func_dir = oracle.func_directional(x_k, u_k, d_k_x, d_k_u, 0)
    while oracle.func_directional(x_k, u_k, d_k_x, d_k_u, alpha) > func_dir + c1 * alpha * grad_dir:
        alpha /= 2
    return alpha


def newton_lasso(oracle, x_0, u_0, tolerance=1e-8, max_iter=20, c1=1e-4, display=False):
    """
    Newton's optimization method for barrier function for LASSO-regression
    """
    x_k = np.copy(x_0)
    u_k = np.copy(u_0)
    grad_x = oracle.grad_x(x_0, u_0)
    grad_u = oracle.grad_u(x_0, u_0)
    grad_norm_0 = grad_x.T @ grad_x + grad_u.T @ grad_u

    if display:
        print('--- Newton\'s method ---')

    for k in range(max_iter):
        if (not is_finite(grad_x)) or (not is_finite(grad_u)):
            return (x_k, u_k), 'computational_error'

        grad_norm = grad_x.T @ grad_x + grad_u.T @ grad_u
        if display:
            print(f'k = {k}, x_k = {x_k}, u_k = {u_k}, f(x_k, u_k) = {oracle.func(x_k, u_k)}, '
                  f'||\\nabla f(x_k, u_k)||_2^2 = {grad_norm}')

        if grad_norm <= tolerance * grad_norm_0:
            break

        C_inv = 0.5 * (u_k ** 2 - x_k ** 2) ** 2 / (u_k ** 2 + x_k ** 2)
        DC_inv = -2 * x_k * u_k / (u_k ** 2 + x_k ** 2)
        b_k = -grad_x + DC_inv * grad_u
        A_k = oracle.reduced_hess(x_k, u_k)
        if (not is_finite(b_k)) or (not is_finite(A_k)):
            return (x_k, u_k), 'computational_error'

        try:
            d_k_x = cho_solve(cho_factor(A_k), b_k)
        except np.linalg.LinAlgError:
            return (x_k, u_k), 'newton_direction_error'
        d_k_u = -C_inv * grad_u - DC_inv * d_k_x
        if (not is_finite(d_k_x)) or (not is_finite(d_k_u)):
            return (x_k, u_k), 'computational_error'

        d_diff = np.where(d_k_x > d_k_u, d_k_x - d_k_u, -1)
        d_sum = np.where(d_k_x < -d_k_u, d_k_x + d_k_u, 1)
        alpha_bound_1 = 0.99 * min(np.where(d_diff > 0, (u_k - x_k) / d_diff, np.inf))
        alpha_bound_2 = 0.99 * min(np.where(d_sum < 0, -(u_k + x_k) / d_sum, np.inf))
        alpha_start = min(1, alpha_bound_1, alpha_bound_2)

        alpha_k = backtracking(oracle, x_k, u_k, d_k_x, d_k_u, \
                               grad_x @ d_k_x + grad_u @ d_k_u, alpha_start, c1)

        x_k = x_k + alpha_k * d_k_x
        u_k = u_k + alpha_k * d_k_u
        grad_x = oracle.grad_x(x_k, u_k)
        grad_u = oracle.grad_u(x_k, u_k)
    else:
        if grad_x.T @ grad_x + grad_u.T @ grad_u > tolerance * grad_norm_0:
            return (x_k, u_k), 'iterations_exceeded'

    if display:
        print('--- End Newton\'s method ---')
    return (x_k, u_k), 'success'


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    def append_history():
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['func'].append(func)
        history['duality_gap'].append(dual_gap)
        if x_k.size <= 2:
            history['x'].append(x_k)

    history = defaultdict(list) if trace else None
    t_k = t_0
    x_k = np.copy(x_0)
    u_k = np.copy(u_0)
    compute_duality_gap = lasso_duality_gap if lasso_duality_gap is not None else oracles.lasso_duality_gap
    start_time = datetime.now()

    for k in range(max_iter):
        oracle = oracles.LassoLogBarrierOracle(A, b, reg_coef, t_k)
        dual_gap = compute_duality_gap(x_k, A @ x_k - b, A.T @ (A @ x_k - b), b, reg_coef)
        func = oracle.original_func(x_k, u_k)
        if display:
            print(f'k = {k}, x_k = {x_k}, u_k = {u_k}, t_k = {t_k}, f(x_k, u_k) = {func}, duality_gap = {dual_gap}')
        if trace:
            append_history()

        if dual_gap <= tolerance:
            break

        (x_k, u_k), message_newton = newton_lasso(oracle, x_k, u_k, tolerance_inner, max_iter_inner, c1, display)
        if message_newton != 'success':
            return (x_k, u_k), message_newton, history

        t_k = t_k * gamma
    else:
        if trace:
            append_history()
        if dual_gap > tolerance:
            return (x_k, u_k), 'iterations_exceeded', history

    return (x_k, u_k), 'success', history
