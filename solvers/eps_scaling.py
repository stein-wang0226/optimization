"""Algorithm A5: Epsilon-scaling homotopy framework.

Implements the continuation strategy (cf. textbook Ch.7 penalty/homotopy):
solve a sequence of problems with decreasing epsilon, using each solution
as warm start for the next. Dramatically accelerates convergence at
small target epsilon.
"""

import numpy as np
import time
from .log_sinkhorn import log_sinkhorn


def eps_scaling(C, a, b, eps_target, gamma=0.5,
               tol_final=1e-6, max_iter_per_layer=5000,
               verbose=False):
    """Solve OT with epsilon-scaling homotopy.

    Starts at eps=1.0, decreases by factor gamma each layer,
    warm-starting dual potentials from the previous layer.

    Args:
        C: (m, n) cost matrix
        a: (m,) source distribution
        b: (n,) target distribution
        eps_target: target regularization parameter
        gamma: shrinkage factor per layer (default 0.5)
        tol_final: final tolerance at target epsilon
        max_iter_per_layer: max iterations per inner solve
        verbose: print progress

    Returns:
        Pi: (m, n) transport plan
        history: dict with aggregate convergence diagnostics
    """
    m, n = C.shape
    f = np.zeros(m)
    g = np.zeros(n)

    total_iters = 0
    layer_costs = []
    layer_residuals = []
    layer_times = []
    layer_eps = []
    layer_iters = []

    t_start = time.perf_counter()
    eps = 1.0
    layer_idx = 0

    while True:
        # Loose tolerance for intermediate layers; tight only for the final layer
        is_final = (eps <= eps_target + 1e-15)
        layer_tol = tol_final if is_final else 1e-2

        if verbose:
            print(f"  [EpsScaling] layer {layer_idx}: eps={eps:.4e}, tol={layer_tol:.2e}")

        # Inner solve with warm start
        Pi, hist, f, g = log_sinkhorn(
            C, a, b, eps,
            tol=layer_tol,
            max_iter=max_iter_per_layer,
            f_init=f,
            g_init=g,
            verbose=False
        )

        iters = hist['iterations']
        total_iters += iters
        elapsed = time.perf_counter() - t_start

        layer_costs.append(hist['cost'][-1] if hist['cost'] else 0.0)
        layer_residuals.append(hist['marginal_residual'][-1] if hist['marginal_residual'] else 0.0)
        layer_times.append(elapsed)
        layer_eps.append(eps)
        layer_iters.append(iters)

        if verbose:
            print(f"    -> {iters} iters, residual = {layer_residuals[-1]:.2e}")

        # Check if we've reached the target
        if eps <= eps_target + 1e-15:
            break

        # Decrease epsilon
        eps = max(gamma * eps, eps_target)
        layer_idx += 1

    history = {
        'marginal_residual': layer_residuals,
        'cost': layer_costs,
        'time': layer_times,
        'iterations': total_iters,
        'layer_eps': layer_eps,
        'layer_iters': layer_iters,
    }
    return Pi, history
