"""E1: Effect of epsilon on the transport plan.

Vary epsilon and visualize transport plan heatmaps.
Compare transport cost against analytical ground truth.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.distributions import make_1d_ot_problem
from data.ground_truth import ot_1d_ground_truth
from solvers.sinkhorn import sinkhorn
from solvers.log_sinkhorn import log_sinkhorn
from utils.plotting import setup_plot_style, save_fig

setup_plot_style()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run():
    print("=" * 60)
    print("E1: Effect of epsilon on transport plan")
    print("=" * 60)

    C, a, b, x = make_1d_ot_problem(n=200)
    W_true = ot_1d_ground_truth(a, b, x)
    print(f"  Analytical W_true = {W_true:.6f}")

    eps_values = [1.0, 0.1, 0.01, 0.001]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for idx, eps in enumerate(eps_values):
        ax = axes[idx // 2, idx % 2]

        # Use log_sinkhorn for small eps (numerical stability)
        if eps >= 0.01:
            Pi, hist = sinkhorn(C, a, b, eps, tol=1e-8, max_iter=20000)
        else:
            Pi, hist, _, _ = log_sinkhorn(C, a, b, eps, tol=1e-8, max_iter=20000)

        cost = np.sum(C * Pi)
        error = abs(cost - W_true)

        ax.imshow(Pi, cmap='hot', aspect='auto', origin='lower')
        ax.set_title(f'ε = {eps}, cost = {cost:.4f}\n|error| = {error:.2e}')
        ax.set_xlabel('Target index j')
        ax.set_ylabel('Source index i')

        print(f"  eps={eps:.3f}: cost={cost:.6f}, error={error:.2e}, iters={hist['iterations']}")

    fig.suptitle('E1: Transport Plan Heatmaps at Different ε', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig_e1_heatmaps.png', BASE_DIR)
    print()


if __name__ == '__main__':
    run()
