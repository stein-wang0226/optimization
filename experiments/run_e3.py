"""E3: Epsilon-scaling on/off comparison.

Compare total iterations to reach target epsilon with and without
the homotopy continuation strategy.
Theory predicts: eps-scaling reduces total iterations for small target epsilon.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.distributions import make_1d_ot_problem
from solvers.log_sinkhorn import log_sinkhorn
from solvers.eps_scaling import eps_scaling
from utils.plotting import setup_plot_style, save_fig

setup_plot_style()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run():
    print("=" * 60)
    print("E3: ε-scaling on/off comparison")
    print("=" * 60)

    C, a, b, x = make_1d_ot_problem(n=200)

    # Use a range of target epsilons — scaling helps most at small eps
    eps_targets = [0.005, 0.001, 5e-4, 1e-4]

    results = {'target': [], 'bare_iters': [], 'scaling_iters': [], 'layers': []}

    for eps_target in eps_targets:
        # Bare log-Sinkhorn: solve directly at target epsilon
        Pi_bare, hist_bare, _, _ = log_sinkhorn(
            C, a, b, eps_target, tol=1e-6, max_iter=200000
        )
        bare_iters = hist_bare['iterations']

        # Eps-scaling: homotopy from large to small epsilon
        Pi_scale, hist_scale = eps_scaling(
            C, a, b, eps_target, gamma=0.5, tol_final=1e-6,
            max_iter_per_layer=50000, verbose=False
        )
        scaling_iters = hist_scale['iterations']
        n_layers = len(hist_scale['layer_eps'])

        speedup = bare_iters / max(scaling_iters, 1)

        results['target'].append(eps_target)
        results['bare_iters'].append(bare_iters)
        results['scaling_iters'].append(scaling_iters)
        results['layers'].append(n_layers)

        print(f"  eps={eps_target:.4f}: bare={bare_iters:6d}, "
              f"scaling={scaling_iters:5d} ({n_layers} layers), "
              f"speedup={speedup:.1f}x")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(eps_targets))
    width = 0.35

    ax.bar(x_pos - width / 2, results['bare_iters'], width,
           label='Bare Log-Sinkhorn', color='steelblue')
    ax.bar(x_pos + width / 2, results['scaling_iters'], width,
           label='ε-Scaling (warm start)', color='coral')

    ax.set_xlabel('Target ε')
    ax.set_ylabel('Total Iterations')
    ax.set_title('E3: ε-Scaling Speedup (total iterations to reach tol=1e-6)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{e}' for e in eps_targets])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_fig(fig, 'fig_e3_eps_scaling.png', BASE_DIR)
    print()


if __name__ == '__main__':
    run()
