"""1D Gaussian mixture histograms for OT experiments."""

import numpy as np


def make_1d_ot_problem(n=200, seed=2026):
    """Build a 1D optimal transport problem with Gaussian mixture distributions.

    Source a: bimodal Gaussian mixture (peaks at 0.3 and 0.7)
    Target b: unimodal Gaussian (peak at 0.5)
    Cost C: squared Euclidean distance

    Returns:
        C: (n, n) cost matrix
        a: (n,) source distribution (sums to 1)
        b: (n,) target distribution (sums to 1)
        x: (n,) grid points in [0, 1]
    """
    np.random.seed(seed)
    x = np.linspace(0, 1, n)

    # Source: bimodal mixture
    a = 0.5 * np.exp(-((x - 0.3) ** 2) / (2 * 0.05 ** 2)) \
        + 0.5 * np.exp(-((x - 0.7) ** 2) / (2 * 0.05 ** 2))
    a /= a.sum()

    # Target: unimodal Gaussian
    b = np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2))
    b /= b.sum()

    # Squared Euclidean cost matrix
    C = (x[:, None] - x[None, :]) ** 2

    return C, a, b, x
