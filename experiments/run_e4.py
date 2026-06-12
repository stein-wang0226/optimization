"""E4: Numerical stability comparison — A1 (naive Sinkhorn) vs A3 (log-domain).

At very small epsilon, A1's kernel K=exp(-C/eps) underflows to 0,
causing division by zero and NaN. A3 remains stable via log-sum-exp.

We use a problem where source and target grids are OFFSET (no shared points),
so C has no zero diagonal entry, making underflow realistic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.ground_truth import ot_1d_ground_truth
from solvers.sinkhorn import sinkhorn
from solvers.log_sinkhorn import log_sinkhorn
from utils.plotting import setup_plot_style, save_fig

setup_plot_style()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_offset_problem(n=100, seed=2026):
    """Build 1D OT problem with offset grids (no C_ii=0 entries).

    Source on [0, 0.5], target on [0.5, 1].
    Minimum cost is (0.5-0.5)^2/n^2 ≈ 0 for boundary points,
    but most entries have C >> eps for small eps.
    """
    np.random.seed(seed)
    x_src = np.linspace(0, 0.5, n)
    x_tgt = np.linspace(0.5, 1.0, n)

    a = np.exp(-((x_src - 0.25) ** 2) / (2 * 0.05 ** 2))
    a /= a.sum()
    b = np.exp(-((x_tgt - 0.75) ** 2) / (2 * 0.05 ** 2))
    b /= b.sum()

    C = (x_src[:, None] - x_tgt[None, :]) ** 2
    return C, a, b, x_src, x_tgt


def run():
    print("=" * 60)
    print("E4: Numerical stability — A1 vs A3 at small ε")
    print("=" * 60)

    C, a, b, x_src, x_tgt = make_offset_problem(n=100)
    n = len(a)
    print(f"  Problem: {n}x{n}, C_min={C.min():.4f}, C_max={C.max():.4f}")

    # Test at two epsilon values
    eps_values = [0.01, 0.001, 0.0001]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: convergence at moderate eps (both work)
    eps_ok = 0.01
    print(f"\n  ε={eps_ok} (both stable):")
    Pi_a1_ok, h1_ok = sinkhorn(C, a, b, eps_ok, tol=1e-8, max_iter=5000)
    Pi_a3_ok, h3_ok, _, _ = log_sinkhorn(C, a, b, eps_ok, tol=1e-8, max_iter=5000)
    print(f"    A1: {h1_ok['iterations']} iters, cost={np.sum(C*Pi_a1_ok):.6f}")
    print(f"    A3: {h3_ok['iterations']} iters, cost={np.sum(C*Pi_a3_ok):.6f}")

    ax = axes[0, 0]
    ax.semilogy(h1_ok['marginal_residual'], label='A1 naive', color='red')
    ax.semilogy(h3_ok['marginal_residual'], label='A3 log-domain', color='blue')
    ax.set_title(f'ε={eps_ok}: Both Stable')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Marginal Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: kernel sparsity vs epsilon
    ax = axes[0, 1]
    for eps in eps_values:
        K = np.exp(-C / eps)
        sparsity = np.mean(K < 1e-300)  # fraction of entries that underflow
        nonzero = np.mean(K > 1e-15)
        print(f"  ε={eps}: K nonzero={nonzero*100:.1f}%, underflow={sparsity*100:.1f}%")

    # Run A1 at very small eps to show breakdown
    eps_bad = 0.0001
    K_bad = np.exp(-C / eps_bad)
    print(f"\n  ε={eps_bad} (A1 kernel underflows):")
    print(f"    K: {np.mean(K_bad < 1e-300)*100:.1f}% underflow to 0")

    Pi_a1_bad, h1_bad = sinkhorn(C, a, b, eps_bad, tol=1e-6, max_iter=2000)
    has_nan = np.any(np.isnan(Pi_a1_bad)) or np.any(np.isinf(Pi_a1_bad))
    cost_a1 = np.sum(C * Pi_a1_bad) if not has_nan else float('nan')
    print(f"    A1: NaN/Inf={has_nan}, cost={cost_a1:.6f}" if not has_nan else f"    A1: CRASHED (NaN/Inf)")

    Pi_a3_bad, h3_bad, _, _ = log_sinkhorn(C, a, b, eps_bad, tol=1e-6, max_iter=5000)
    cost_a3 = np.sum(C * Pi_a3_bad)
    print(f"    A3: {h3_bad['iterations']} iters, cost={cost_a3:.6f}")

    # Bottom-left: A1 vs A3 at eps_bad — residual curves
    ax = axes[1, 0]
    r_a1 = [r for r in h1_bad['marginal_residual'] if not (np.isnan(r) or np.isinf(r))]
    r_a3 = h3_bad['marginal_residual']
    if r_a1:
        ax.semilogy(r_a1, label=f'A1 naive ({len(r_a1)} iters)', color='red', alpha=0.7)
    else:
        ax.axhline(y=1, color='red', linestyle='--', label='A1 naive (immediate NaN)')
    ax.semilogy(r_a3, label=f'A3 log-domain ({h3_bad["iterations"]} iters)', color='blue')
    ax.set_title(f'ε={eps_bad}: A1 Breaks Down')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Marginal Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: transport plans comparison
    ax = axes[1, 1]
    if has_nan:
        # Show A3 result only
        ax.imshow(Pi_a3_bad, cmap='hot', aspect='auto', origin='lower')
        ax.set_title(f'A3 Transport Plan (ε={eps_bad})\nA1 produced NaN')
    else:
        # Side by side
        ax.imshow(Pi_a3_bad, cmap='hot', aspect='auto', origin='lower')
        ax.set_title(f'A3 Transport Plan (ε={eps_bad})\ncost={cost_a3:.6f}')
    ax.set_xlabel('Target index j')
    ax.set_ylabel('Source index i')

    plt.suptitle('E4: Numerical Stability — A1 vs A3', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig_e4_stability.png', BASE_DIR)
    print()


if __name__ == '__main__':
    run()
