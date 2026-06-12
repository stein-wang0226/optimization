"""Algorithm A2: Dual gradient ascent for entropy-regularized OT.

Baseline method: gradient ascent on the smooth concave dual function.
Convergence: O(1/k) sublinear (vs Sinkhorn's linear rate).
"""

import numpy as np
import time


def dual_gradient(C, a, b, eps, tol=1e-6, max_iter=10000,
                  step_size=None, verbose=False):
    """Solve entropy-regularized OT via dual gradient ascent.

    Args:
        C: (m, n) cost matrix
        a: (m,) source distribution
        b: (n,) target distribution
        eps: regularization parameter (> 0)
        tol: gradient norm tolerance
        max_iter: maximum iterations
        step_size: gradient step size (default: eps/2, theoretical safe step)
        verbose: print progress

    Returns:
        Pi: (m, n) transport plan
        history: dict with convergence diagnostics
    """
    m, n = C.shape
    if step_size is None:
        step_size = eps / 2.0  # Lipschitz constant L_Phi <= 2/eps, safe step = 1/L

    f = np.zeros(m)
    g = np.zeros(n)

    hist_residual = []
    hist_cost = []
    hist_time = []

    t_start = time.perf_counter()
    k = 0
    while k < max_iter:
        # Reconstruct Pi from dual potentials: Pi_ij = exp((f_i + g_j - C_ij)/eps)
        # Use outer sum: f[:,None] + g[None,:] gives (m,n) matrix of f_i + g_j
        log_Pi = (f[:, None] + g[None, :] - C) / eps

        # Numerical guard: clamp to prevent overflow
        log_Pi = np.clip(log_Pi, -500, 500)
        Pi = np.exp(log_Pi)

        # Dual gradient = marginal residual
        gf = a - Pi @ np.ones(n)     # gradient w.r.t. f
        gg = b - Pi.T @ np.ones(m)   # gradient w.r.t. g

        grad_norm = np.abs(gf).sum() + np.abs(gg).sum()

        elapsed = time.perf_counter() - t_start
        hist_residual.append(grad_norm)
        hist_cost.append(np.sum(C * Pi))
        hist_time.append(elapsed)

        if verbose and k % 100 == 0:
            print(f"  [DualGrad] iter {k:5d}, residual = {grad_norm:.2e}")

        if grad_norm < tol:
            break

        # Gradient ascent step
        f = f + step_size * gf
        g = g + step_size * gg

        k += 1

    # Final transport plan
    log_Pi = (f[:, None] + g[None, :] - C) / eps
    log_Pi = np.clip(log_Pi, -500, 500)
    Pi = np.exp(log_Pi)

    history = {
        'marginal_residual': hist_residual,
        'cost': hist_cost,
        'time': hist_time,
        'iterations': k + 1,
    }
    return Pi, history
