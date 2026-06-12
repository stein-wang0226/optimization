"""E8: Color transfer between images using optimal transport.

Pipeline:
1. Load two images, extract RGB pixels
2. Hand-written K-means to cluster each into k color centroids
3. Solve OT between cluster weight distributions
4. Remap source colors via barycentric transport
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from solvers.log_sinkhorn import log_sinkhorn
from utils.plotting import setup_plot_style, save_fig

setup_plot_style()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kmeans_simple(pixels, k=256, max_iter=100, seed=2026):
    """Hand-written K-means clustering.

    Args:
        pixels: (N, d) data points
        k: number of clusters
        max_iter: maximum iterations
        seed: random seed

    Returns:
        centers: (k, d) cluster centers
        labels: (N,) cluster assignments
        weights: (k,) cluster frequencies (sum to 1)
    """
    np.random.seed(seed)
    N, d = pixels.shape

    # Initialize: random selection of k data points
    indices = np.random.choice(N, k, replace=False)
    centers = pixels[indices].copy()

    labels = np.zeros(N, dtype=int)

    iter_count = 0
    while iter_count < max_iter:
        # Assignment step: assign each point to nearest center
        # Compute distances: ||x_i - c_j||^2 = ||x_i||^2 - 2*x_i.c_j + ||c_j||^2
        dists = -2 * pixels @ centers.T + np.sum(centers ** 2, axis=1)[None, :]
        new_labels = np.argmin(dists, axis=1)

        # Check convergence
        if np.all(new_labels == labels) and iter_count > 0:
            break
        labels = new_labels

        # Update step: recompute centers
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = pixels[mask].mean(axis=0)

        iter_count += 1

    # Compute weights (cluster frequencies)
    weights = np.zeros(k)
    for j in range(k):
        weights[j] = np.sum(labels == j)
    weights = weights / weights.sum()

    return centers, labels, weights


def remap_colors(source_pixels, source_labels, source_centers,
                 target_centers, Pi):
    """Remap source pixel colors via barycentric transport.

    For each source cluster j, the transport plan Pi[:,j] gives
    a distribution over target clusters. The new color is the
    weighted average of target cluster centers.
    """
    # Barycentric mapping: for source cluster j,
    # new_color = sum_i (Pi_ij / a_j) * target_center_i
    k_source = source_centers.shape[0]
    k_target = target_centers.shape[0]

    # Compute mapping matrix: M[j, :] = new color for source cluster j
    source_weights = Pi.sum(axis=1)  # a_j = sum_i Pi_ij (row sums)
    source_weights = np.maximum(source_weights, 1e-10)  # avoid division by zero

    M = np.zeros_like(source_centers)
    for j in range(k_source):
        # Weight for each target cluster i: Pi_ij / a_j
        w = Pi[j, :] / source_weights[j]
        # Weighted average of target centers
        M[j] = w @ target_centers

    # Apply mapping to all pixels
    new_pixels = M[source_labels]
    return new_pixels


def make_demo_images(size=100, seed=2026):
    """Generate two synthetic demo images for color transfer.

    Image A: warm tones (reds, oranges, yellows)
    Image B: cool tones (blues, greens, purples)
    """
    np.random.seed(seed)

    # Image A: warm landscape-like
    img_a = np.zeros((size, size, 3), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            t = i / size
            img_a[i, j, 0] = 0.7 + 0.3 * np.sin(t * np.pi) + np.random.normal(0, 0.05)
            img_a[i, j, 1] = 0.3 + 0.4 * t + np.random.normal(0, 0.05)
            img_a[i, j, 2] = 0.1 + 0.2 * (1 - t) + np.random.normal(0, 0.05)
    img_a = np.clip(img_a, 0, 1)

    # Image B: cool ocean-like
    img_b = np.zeros((size, size, 3), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            t = i / size
            img_b[i, j, 0] = 0.1 + 0.2 * t + np.random.normal(0, 0.05)
            img_b[i, j, 1] = 0.3 + 0.3 * np.sin(t * np.pi) + np.random.normal(0, 0.05)
            img_b[i, j, 2] = 0.5 + 0.4 * (1 - t) + np.random.normal(0, 0.05)
    img_b = np.clip(img_b, 0, 1)

    return img_a, img_b


def run():
    print("=" * 60)
    print("E8: Color Transfer via Optimal Transport")
    print("=" * 60)

    # Generate demo images (or load real images if available)
    img_source, img_target = make_demo_images(size=100)
    print("  Generated synthetic demo images (100x100)")

    k = 64  # number of color clusters (reduced for speed)
    eps = 0.01

    # Extract pixels
    h, w, _ = img_source.shape
    source_pixels = img_source.reshape(-1, 3)
    target_pixels = img_target.reshape(-1, 3)

    # K-means clustering
    print(f"  Running K-means (k={k}) on source...")
    src_centers, src_labels, src_weights = kmeans_simple(source_pixels, k=k, seed=2026)

    print(f"  Running K-means (k={k}) on target...")
    tgt_centers, tgt_labels, tgt_weights = kmeans_simple(target_pixels, k=k, seed=2027)

    # Cost matrix: squared Euclidean distance between color centroids
    C = np.sum((src_centers[:, None, :] - tgt_centers[None, :, :]) ** 2, axis=2)

    # Solve OT
    print(f"  Solving OT (ε={eps})...")
    Pi, hist, _, _ = log_sinkhorn(C, src_weights, tgt_weights, eps, tol=1e-6, max_iter=5000)
    print(f"  OT solved in {hist['iterations']} iterations")

    # Remap colors
    new_pixels = remap_colors(source_pixels, src_labels, src_centers, tgt_centers, Pi)
    img_result = new_pixels.reshape(h, w, 3).clip(0, 1)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_source)
    axes[0].set_title('Source Image')
    axes[0].axis('off')

    axes[1].imshow(img_target)
    axes[1].set_title('Target Image')
    axes[1].axis('off')

    axes[2].imshow(img_result)
    axes[2].set_title('After Color Transfer')
    axes[2].axis('off')

    plt.suptitle('E8: Color Transfer via Optimal Transport', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig_e8_color_transfer.png', BASE_DIR)

    # Also show transport plan
    fig2, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(Pi, cmap='hot', aspect='auto', origin='lower')
    ax.set_title(f'Transport Plan (ε={eps})')
    ax.set_xlabel('Target cluster')
    ax.set_ylabel('Source cluster')
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
    plt.tight_layout()
    save_fig(fig2, 'fig_e8_transport_plan.png', BASE_DIR)
    print()


if __name__ == '__main__':
    run()
