"""Analytical 1D optimal transport via CDF quantile matching.

For 1D distributions, the optimal transport map is given by
the monotone rearrangement T = F_b^{-1} ∘ F_a, and the
Wasserstein-2 cost is:
    W_2^2 = integral_0^1 |F_a^{-1}(t) - F_b^{-1}(t)|^2 dt
"""

import numpy as np


def ot_1d_ground_truth(a, b, x, n_quad=10000):
    """Compute exact 1D OT cost via inverse CDF matching.

    Args:
        a: (n,) source distribution
        b: (n,) target distribution
        x: (n,) grid points
        n_quad: number of quadrature points for integration

    Returns:
        W: exact Wasserstein-2 squared cost
    """
    F_a = np.cumsum(a)
    F_b = np.cumsum(b)

    # Uniform samples in [0, 1] for inverse CDF evaluation
    t = np.linspace(0, 1, n_quad)

    # Inverse CDF via interpolation: F^{-1}(t) maps quantile -> position
    xa = np.interp(t, F_a, x)
    xb = np.interp(t, F_b, x)

    # W_2^2 = integral |F_a^{-1}(t) - F_b^{-1}(t)|^2 dt
    W = np.trapezoid((xa - xb) ** 2, t)
    return W
