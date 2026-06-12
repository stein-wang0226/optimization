"""Metrics for evaluating transport plans."""

import numpy as np


def marginal_residual(Pi, a, b):
    """Compute L1 marginal residual: ||Pi@1 - a||_1 + ||Pi.T@1 - b||_1.

    This equals the dual gradient norm (KKT condition).
    """
    row_err = np.abs(Pi @ np.ones(Pi.shape[1]) - a).sum()
    col_err = np.abs(Pi.T @ np.ones(Pi.shape[0]) - b).sum()
    return row_err + col_err


def transport_cost(C, Pi):
    """Compute transport cost: <C, Pi> = sum(C * Pi)."""
    return np.sum(C * Pi)
