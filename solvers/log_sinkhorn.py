"""Algorithm A3: Log-domain stabilized Sinkhorn.

Arithmetically equivalent to A1 but operates in log-space to prevent
numerical overflow/underflow at small epsilon. The key trick is
log-sum-exp with max subtraction:
    log(sum(exp(z))) = z_max + log(sum(exp(z - z_max)))
"""

import numpy as np
import time


def _log_sum_exp(z, axis=None):
    """Numerically stable log-sum-exp along given axis."""
    z_max = np.max(z, axis=axis, keepdims=True)
    return z_max.squeeze(axis=axis) + np.log(
        np.sum(np.exp(z - z_max), axis=axis)
    )


def log_sinkhorn(C, a, b, eps, tol=1e-6, max_iter=10000,
                 f_init=None, g_init=None, verbose=False):
    """Solve entropy-regularized OT via log-domain Sinkhorn.

    Args:
        C: (m, n) cost matrix
        a: (m,) source distribution
        b: (n,) target distribution
        eps: regularization parameter (> 0)
        tol: marginal residual tolerance
        max_iter: maximum iterations
        f_init: (m,) initial dual potential f (for warm start)
        g_init: (n,) initial dual potential g (for warm start)
        verbose: print progress

    Returns:
        Pi: (m, n) transport plan
        history: dict with convergence diagnostics
        f: (m,) dual potential
        g: (n,) dual potential
    """
    m, n = C.shape

    # Initialize dual potentials (or warm start)
    if f_init is not None:
        f = f_init.copy()
    else:
        f = np.zeros(m)
    if g_init is not None:
        g = g_init.copy()
    else:
        g = np.zeros(n)

    log_a = np.log(a + 1e-300)  # guard against log(0)
    log_b = np.log(b + 1e-300)

    hist_residual = []
    hist_cost = []
    hist_time = []

    t_start = time.perf_counter()
    k = 0
    while k < max_iter:
        # f-update: f_i = eps * log(a_i) - eps * LSE_j((g_j - C_ij) / eps)
        # M has shape (m, n): M_ij = (g_j - C_ij) / eps
        M = (g[None, :] - C) / eps
        f = eps * log_a - eps * _log_sum_exp(M, axis=1)

        # g-update: g_j = eps * log(b_j) - eps * LSE_i((f_i - C_ij) / eps)
        # M has shape (m, n): M_ij = (f_i - C_ij) / eps
        M = (f[:, None] - C) / eps
        g = eps * log_b - eps * _log_sum_exp(M, axis=0)

        # Compute marginal residual stably
        # After g-update, column marginals are automatically satisfied (by construction).
        # Convergence means row marginals also match — check sum_j Pi_ij vs a_i.
        log_Pi = (f[:, None] + g[None, :] - C) / eps
        log_row_sum = _log_sum_exp(log_Pi, axis=1)  # log(sum_j Pi_ij) for each i
        row_marginal = np.exp(log_row_sum)
        r = np.abs(row_marginal - a).sum()

        elapsed = time.perf_counter() - t_start
        hist_residual.append(r)

        # Transport cost (stably computed)
        cost = np.sum(np.exp(log_Pi) * C)
        hist_cost.append(cost)
        hist_time.append(elapsed)

        if verbose and k % 100 == 0:
            print(f"  [LogSinkhorn] iter {k:5d}, residual = {r:.2e}")

        if r < tol:
            break
        k += 1

    # Reconstruct transport plan
    log_Pi = (f[:, None] + g[None, :] - C) / eps
    Pi = np.exp(log_Pi)

    history = {
        'marginal_residual': hist_residual,
        'cost': hist_cost,
        'time': hist_time,
        'iterations': k + 1,
    }
    return Pi, history, f, g
