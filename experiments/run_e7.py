"""E7: Scalability — time and iterations vs problem size n.

Theory predicts: per-iteration cost is O(n^2) (matrix-vector product).
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
    print("E7: Scalability vs problem size n")
    print("=" * 60)

    n_values = [50, 100, 200, 500, 1000]
    eps = 0.01
    results = {'n': [], 'iters': [], 'time_per_iter': [], 'total_time': []}

    for n in n_values:
        C, a, b, x = make_1d_ot_problem(n=n)
        Pi, hist, _, _ = log_sinkhorn(C, a, b, eps, tol=1e-6, max_iter=20000)

        total_time = hist['time'][-1]
        time_per_iter = total_time / max(hist['iterations'], 1)

        results['n'].append(n)
        results['iters'].append(hist['iterations'])
        results['time_per_iter'].append(time_per_iter)
        results['total_time'].append(total_time)

        print(f"  n={n:4d}: {hist['iterations']:5d} iters, "
              f"total={total_time:.3f}s, per_iter={time_per_iter*1000:.2f}ms")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Iterations vs n
    axes[0].plot(results['n'], results['iters'], 'bo-', markersize=8)
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('Iterations')
    axes[0].set_title('Iterations vs n')
    axes[0].grid(True, alpha=0.3)

    # Time per iteration vs n (should be ~O(n^2))
    axes[1].loglog(results['n'], [t * 1000 for t in results['time_per_iter']],
                   'rs-', markersize=8, label='measured')
    # Reference O(n^2) line
    n_ref = np.array(results['n'], dtype=float)
    t_ref = np.array(results['time_per_iter']) * 1000
    scale = t_ref[1] / (n_ref[1] ** 2) if len(n_ref) > 1 else 1e-6
    axes[1].loglog(n_ref, scale * n_ref ** 2, 'k--', alpha=0.5, label='O(n²) reference')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('Time per iter (ms)')
    axes[1].set_title('Per-Iteration Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Total time vs n
    axes[2].loglog(results['n'], results['total_time'], 'g^-', markersize=8)
    axes[2].set_xlabel('n')
    axes[2].set_ylabel('Total time (s)')
    axes[2].set_title('Total Solve Time')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'E7: Scalability (ε={eps})', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig_e7_scalability.png', BASE_DIR)
    print()


if __name__ == '__main__':
    run()
