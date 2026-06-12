"""Algorithm A4: ADMM for solving the unregularized OT linear program.

Provides an exact (epsilon=0) baseline for comparison.
Variable splitting: min <C,Pi> + delta_A(Pi) + delta_{>=0}(Z) s.t. Pi = Z
where A = {Pi: Pi@1 = a, Pi.T@1 = b}.
"""

import numpy as np
import time


def _project_affine(X, a, b):
    """Project matrix X onto the affine set A = {Pi: Pi@1=a, Pi.T@1=b}.

    Uses explicit rank-1 correction formula (no solver needed):
        P_A(X) = X + (a - X@1)@1.T/n + 1@(b - X.T@1).T/m - sigma/(m*n) * 11.T
    where sigma = 1.T @ X @ 1 - 1.

    Args:
        X: (m, n) matrix to project
        a: (m,) target row sums
        b: (n,) target column sums

    Returns:
        P: (m, n) projected matrix with P@1 = a and P.T@1 = b
    """
    m, n = X.shape
    ones_m = np.ones(m)
    ones_n = np.ones(n)

    row_sum = X @ ones_n          # current row sums
    col_sum = X.T @ ones_m        # current column sums
    sigma = ones_m @ X @ ones_n - 1.0

    row_deficit = a - row_sum     # (m,)
    col_deficit = b - col_sum     # (n,)

    # P_ij = X_ij + row_deficit_i/n + col_deficit_j/m + sigma/(m*n)
    P = X + np.outer(row_deficit, ones_n) / n \
        + np.outer(ones_m, col_deficit) / m \
        + sigma / (m * n) * np.outer(ones_m, ones_n)
    return P


def admm_ot(C, a, b, rho=1.0, tol=1e-6, max_iter=10000, verbose=False):
    """Solve unregularized OT LP via ADMM.

    Args:
        C: (m, n) cost matrix
        a: (m,) source distribution
        b: (n,) target distribution
        rho: ADMM penalty parameter
        tol: convergence tolerance (primal + dual residual)
        max_iter: maximum iterations
        verbose: print progress

    Returns:
        Z: (m, n) transport plan (non-negative, satisfies marginals)
        history: dict with convergence diagnostics
    """
    m, n = C.shape

    # Initialize
    Pi = np.outer(a, b)
    Z = Pi.copy()
    Y = np.zeros((m, n))  # dual variable (scaled Lagrange multiplier)

    hist_primal = []
    hist_dual = []
    hist_cost = []
    hist_time = []

    t_start = time.perf_counter()
    k = 0
    while k < max_iter:
        # Pi-update: project onto affine constraint set
        Pi = _project_affine(Z - (C + Y) / rho, a, b)

        # Z-update: project onto non-negative orthant
        Z_old = Z.copy()
        Z = np.maximum(Pi + Y / rho, 0.0)

        # Dual variable update
        Y = Y + rho * (Pi - Z)

        # Residuals
        r_primal = np.linalg.norm(Pi - Z, 'fro')       # primal residual
        r_dual = rho * np.linalg.norm(Z - Z_old, 'fro') # dual residual

        elapsed = time.perf_counter() - t_start
        hist_primal.append(r_primal)
        hist_dual.append(r_dual)
        hist_cost.append(np.sum(C * Z))
        hist_time.append(elapsed)

        if verbose and k % 100 == 0:
            print(f"  [ADMM] iter {k:5d}, primal = {r_primal:.2e}, dual = {r_dual:.2e}")

        if r_primal < tol and r_dual < tol:
            break
        k += 1

    # Return Z (non-negative, approximately satisfies marginals)
    history = {
        'primal_residual': hist_primal,
        'dual_residual': hist_dual,
        'marginal_residual': [p + d for p, d in zip(hist_primal, hist_dual)],
        'cost': hist_cost,
        'time': hist_time,
        'iterations': k + 1,
    }
    return Z, history
