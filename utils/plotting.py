"""Shared matplotlib visualization helpers."""

import matplotlib.pyplot as plt
import os


def setup_plot_style():
    """Set clean academic plot style."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.figsize': (8, 5),
    })


def ensure_figures_dir(base_dir='.'):
    """Create figures/ directory if it doesn't exist."""
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def save_fig(fig, name, base_dir='.'):
    """Save figure to figures/ directory."""
    fig_dir = ensure_figures_dir(base_dir)
    path = os.path.join(fig_dir, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
