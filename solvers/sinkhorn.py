"""Algorithm A1: Sinkhorn iteration (dual block coordinate ascent).

The core update is two lines of vector division:
    u = a / (K @ v)
    v = b / (K.T @ u)

Convergence: linear (Franklin-Lorenz 1989), rate degrades as eps -> 0.
"""

import numpy as np
import time


def sinkhorn(C, a, b, eps, tol=1e-6, max_iter=10000, verbose=False):
    """Solve entropy-regularized OT via Sinkhorn iteration.

    Args:
        C: (m, n) cost matrix
        a: (m,) source distribution
        b: (n,) target distribution
        eps: regularization parameter (> 0)
        tol: marginal residual tolerance
        max_iter: maximum iterations
        verbose: print progress

    Returns:
        Pi: (m, n) transport plan
        history: dict with convergence diagnostics
    """
    m, n = C.shape
    K = np.exp(-C / eps)  # Gibbs kernel, computed once

    v = np.ones(n)

    hist_residual = []
    hist_cost = []
    hist_time = []

    t_start = time.perf_counter()
    k = 0
    while k < max_iter:
        u = a / (K @ v)           # row scaling: closed-form coordinate ascent
        v = b / (K.T @ u)         # column scaling: closed-form coordinate ascent

        # Marginal residual = dual gradient norm (KKT condition)
        row_marginal = u * (K @ v)
        r = np.abs(row_marginal - a).sum()

        elapsed = time.perf_counter() - t_start
        hist_residual.append(r)
        hist_cost.append(u @ (C * K) @ v)  # <C, Pi> = sum(C * u_i K_ij v_j)
        hist_time.append(elapsed)

        if verbose and k % 100 == 0:
            print(f"  [Sinkhorn] iter {k:5d}, residual = {r:.2e}")

        if r < tol:
            break
        k += 1

    # Reconstruct transport plan: Pi = diag(u) @ K @ diag(v)
    Pi = u[:, None] * K * v[None, :]

    history = {
        'marginal_residual': hist_residual,
        'cost': hist_cost,
        'time': hist_time,
        'iterations': k + 1,
    }
    return Pi, history
