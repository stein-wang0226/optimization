"""E5: Algorithm grand comparison.

All 4 core solvers (A1-A4) on the same problem, comparing convergence curves.
Theory predicts: A1/A3 linear (straight lines on semi-log), A2/A4 sublinear (curves).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.distributions import make_1d_ot_problem
from data.ground_truth import ot_1d_ground_truth
from solvers.sinkhorn import sinkhorn
from solvers.dual_gradient import dual_gradient
from solvers.log_sinkhorn import log_sinkhorn
from solvers.admm_lp import admm_ot
from utils.plotting import setup_plot_style, save_fig

setup_plot_style()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run():
    print("=" * 60)
    print("E5: Algorithm grand comparison")
    print("=" * 60)

    C, a, b, x = make_1d_ot_problem(n=200)
    W_true = ot_1d_ground_truth(a, b, x)
    eps = 0.05  # moderate epsilon where all methods work
    print(f"  W_true = {W_true:.6f}, eps = {eps}")

    # A1: Sinkhorn
    print("\n  Running A1 (Sinkhorn)...")
    Pi_a1, hist_a1 = sinkhorn(C, a, b, eps, tol=1e-8, max_iter=20000)
    cost_a1 = np.sum(C * Pi_a1)
    print(f"  A1: {hist_a1['iterations']} iters, cost={cost_a1:.6f}")

    # A2: Dual gradient ascent
    print("  Running A2 (Dual Gradient)...")
    Pi_a2, hist_a2 = dual_gradient(C, a, b, eps, tol=1e-8, max_iter=20000)
    cost_a2 = np.sum(C * Pi_a2)
    print(f"  A2: {hist_a2['iterations']} iters, cost={cost_a2:.6f}")

    # A3: Log-domain Sinkhorn
    print("  Running A3 (Log-Sinkhorn)...")
    Pi_a3, hist_a3, _, _ = log_sinkhorn(C, a, b, eps, tol=1e-8, max_iter=20000)
    cost_a3 = np.sum(C * Pi_a3)
    print(f"  A3: {hist_a3['iterations']} iters, cost={cost_a3:.6f}")

    # A4: ADMM (LP solver, no epsilon)
    print("  Running A4 (ADMM-LP)...")
    Pi_a4, hist_a4 = admm_ot(C, a, b, rho=10.0, tol=1e-8, max_iter=20000)
    cost_a4 = np.sum(C * Pi_a4)
    print(f"  A4: {hist_a4['iterations']} iters, cost={cost_a4:.6f}")

    # Plot: convergence curves (semi-log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: marginal residual vs iteration
    ax = axes[0]
    ax.semilogy(hist_a1['marginal_residual'], label=f'A1 Sinkhorn ({hist_a1["iterations"]} iters)')
    ax.semilogy(hist_a2['marginal_residual'], label=f'A2 DualGrad ({hist_a2["iterations"]} iters)')
    ax.semilogy(hist_a3['marginal_residual'], label=f'A3 LogSinkhorn ({hist_a3["iterations"]} iters)')
    ax.semilogy(hist_a4['marginal_residual'], label=f'A4 ADMM ({hist_a4["iterations"]} iters)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Marginal Residual (L1)')
    ax.set_title('Convergence: Marginal Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: cost error vs iteration
    ax = axes[1]
    errors_a1 = [abs(c - W_true) for c in hist_a1['cost']]
    errors_a2 = [abs(c - W_true) for c in hist_a2['cost']]
    errors_a3 = [abs(c - W_true) for c in hist_a3['cost']]
    errors_a4 = [abs(c - W_true) for c in hist_a4['cost']]

    ax.semilogy(errors_a1, label='A1 Sinkhorn')
    ax.semilogy(errors_a2, label='A2 DualGrad')
    ax.semilogy(errors_a3, label='A3 LogSinkhorn')
    ax.semilogy(errors_a4, label='A4 ADMM')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|cost - W_true|')
    ax.set_title('Cost Error vs Analytical Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'E5: Algorithm Comparison (ε={eps})', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig_e5_algorithm_comparison.png', BASE_DIR)
    print()


if __name__ == '__main__':
    run()
