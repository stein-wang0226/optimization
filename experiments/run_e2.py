"""E2: Effect of epsilon on convergence speed.

Semi-log residual curves at different epsilon values.
Theory predicts: convergence rate degrades as eps -> 0 (kappa -> 1).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.distributions import make_1d_ot_problem
from solvers.log_sinkhorn import log_sinkhorn
from utils.plotting import setup_plot_style, save_fig

setup_plot_style()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run():
    print("=" * 60)
    print("E2: Effect of epsilon on convergence speed")
    print("=" * 60)

    C, a, b, x = make_1d_ot_problem(n=200)
    eps_values = [1.0, 0.1, 0.01, 0.001]

    fig, ax = plt.subplots(figsize=(10, 6))

    for eps in eps_values:
        Pi, hist, _, _ = log_sinkhorn(C, a, b, eps, tol=1e-10, max_iter=50000)

        residuals = hist['marginal_residual']
        ax.semilogy(residuals, label=f'ε = {eps} ({hist["iterations"]} iters)')

        print(f"  eps={eps:.3f}: {hist['iterations']} iterations, "
              f"final residual = {residuals[-1]:.2e}")

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Marginal Residual (L1)')
    ax.set_title('E2: Convergence Speed vs ε (Log-Sinkhorn)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'fig_e2_convergence_vs_eps.png', BASE_DIR)
    print()


if __name__ == '__main__':
    run()
