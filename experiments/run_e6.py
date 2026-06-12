"""E6: ADMM penalty parameter (rho) sensitivity.

Two views:
1. Convergence curves at different rho values
2. Cost accuracy after fixed iteration budget vs rho (U-shaped curve)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.distributions import make_1d_ot_problem
from data.ground_truth import ot_1d_ground_truth
from solvers.admm_lp import admm_ot
from utils.plotting import setup_plot_style, save_fig

setup_plot_style()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run():
    print("=" * 60)
    print("E6: ADMM ρ sensitivity")
    print("=" * 60)

    C, a, b, x = make_1d_ot_problem(n=200)
    W_true = ot_1d_ground_truth(a, b, x)

    rho_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    iter_budget = 3000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: convergence curves for selected rho values
    selected_rhos = [0.1, 1.0, 10.0, 50.0]
    print(f"\n  Convergence curves (selected ρ values):")
    for rho in selected_rhos:
        Pi, hist = admm_ot(C, a, b, rho=rho, tol=1e-10, max_iter=iter_budget)
        combined = [p + d for p, d in zip(hist['primal_residual'], hist['dual_residual'])]
        axes[0].semilogy(combined, label=f'ρ={rho}')
        cost = np.sum(C * Pi)
        print(f"    rho={rho:6.2f}: {hist['iterations']:5d} iters, "
              f"cost={cost:.6f}, err={abs(cost-W_true):.2e}")

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Primal + Dual Residual')
    axes[0].set_title('Convergence at Different ρ')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: cost error vs rho after fixed iteration budget
    print(f"\n  Cost error after {iter_budget} iterations:")
    errors = []
    for rho in rho_values:
        Pi, hist = admm_ot(C, a, b, rho=rho, tol=1e-10, max_iter=iter_budget)
        cost = np.sum(C * Pi)
        error = abs(cost - W_true)
        errors.append(error)
        print(f"    rho={rho:7.3f}: cost={cost:.6f}, error={error:.2e}")

    axes[1].loglog(rho_values, errors, 'bo-', markersize=8, linewidth=2)
    axes[1].set_xlabel('ρ')
    axes[1].set_ylabel('|cost - W_true|')
    axes[1].set_title(f'Cost Error After {iter_budget} Iterations')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('E6: ADMM ρ Sensitivity', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig_e6_admm_rho.png', BASE_DIR)
    print()


if __name__ == '__main__':
    run()
