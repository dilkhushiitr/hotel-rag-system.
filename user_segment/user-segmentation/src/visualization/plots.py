"""
Visualization Module
─────────────────────
Generates all analysis plots:
  - PCA 2D cluster scatter
  - t-SNE cluster scatter
  - Feature importance heatmap (cluster means)
  - Segment distribution bar chart
  - Silhouette plot
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

PALETTE = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
    "#00BCD4", "#8BC34A", "#FF5722", "#607D8B", "#E91E63",
]


def _get_color_map(labels: np.ndarray) -> Dict[int, str]:
    unique = sorted(set(labels))
    cmap = {}
    color_idx = 0
    for lbl in unique:
        if lbl == -1:
            cmap[lbl] = "#B0B0B0"   # grey for noise / anomaly
        else:
            cmap[lbl] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1
    return cmap


def plot_pca_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    title: str = "K-Means Clusters (PCA 2D Projection)",
    save_path: Optional[str] = None,
) -> None:
    """Reduce to 2D via PCA and scatter-plot cluster assignments."""
    logger.info("Computing PCA projection …")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    cmap = _get_color_map(labels)
    colors = [cmap[l] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.5, s=8, linewidths=0)

    # Legend
    unique_labels = sorted(set(labels))
    patches = []
    for lbl in unique_labels:
        name = (label_names or {}).get(lbl, f"Cluster {lbl}")
        patches.append(mpatches.Patch(color=cmap[lbl], label=name))
    ax.legend(handles=patches, loc="best", fontsize=9, framealpha=0.8)

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("PCA plot saved → %s", save_path)
    plt.close()


def plot_tsne_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    sample_size: int = 3000,
    save_path: Optional[str] = None,
) -> None:
    """t-SNE projection (sampled for speed on large datasets)."""
    n = min(sample_size, len(X_scaled))
    idx = np.random.choice(len(X_scaled), n, replace=False)
    X_s, labels_s = X_scaled[idx], labels[idx]

    logger.info("Computing t-SNE on %d samples …", n)
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X_s)

    cmap = _get_color_map(labels_s)
    colors = [cmap[l] for l in labels_s]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.5, s=8, linewidths=0)

    unique_labels = sorted(set(labels_s))
    patches = [
        mpatches.Patch(
            color=cmap[lbl],
            label=(label_names or {}).get(lbl, f"Cluster {lbl}"),
        )
        for lbl in unique_labels
    ]
    ax.legend(handles=patches, loc="best", fontsize=9, framealpha=0.8)
    ax.set_title("t-SNE Cluster Visualization", fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("t-SNE plot saved → %s", save_path)
    plt.close()


def plot_cluster_heatmap(
    profile_df: pd.DataFrame,
    label_col: str = "kmeans_cluster",
    label_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Heatmap of normalised cluster means across features."""
    feat_cols = [c for c in profile_df.columns if c not in [label_col, "cluster_size", "cluster_pct"]]
    data = profile_df.set_index(label_col)[feat_cols].copy()

    # Rename index rows with business labels
    if label_names:
        data.index = [label_names.get(i, f"Cluster {i}") for i in data.index]

    # Normalise each column 0–1 for visual comparison
    data_norm = (data - data.min()) / (data.max() - data.min()).replace(0, 1)

    fig, ax = plt.subplots(figsize=(max(14, len(feat_cols) * 0.7), max(5, len(data) * 0.9)))
    sns.heatmap(
        data_norm,
        annot=data.round(2),
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalised Value"},
    )
    ax.set_title("Cluster Feature Heatmap (normalised)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Segment")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Heatmap saved → %s", save_path)
    plt.close()


def plot_segment_distribution(
    segments_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Horizontal bar chart of user count per segment."""
    counts = segments_df["segment_label"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.7)))
    bars = ax.barh(counts.index, counts.values, color=PALETTE[: len(counts)], edgecolor="white")

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_width() + counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Number of Users")
    ax.set_title("User Distribution by Segment", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Distribution chart saved → %s", save_path)
    plt.close()


def generate_all_plots(
    X_scaled: np.ndarray,
    feature_df: pd.DataFrame,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    segments_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    label_names: Dict[int, str],
    output_dir: str,
) -> None:
    """Convenience function — generates the full set of analysis plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_pca_clusters(X_scaled, kmeans_labels, label_names,
                      save_path=str(out / "pca_clusters.png"))
    plot_tsne_clusters(X_scaled, kmeans_labels, label_names,
                       save_path=str(out / "tsne_clusters.png"))
    plot_cluster_heatmap(profile_df, label_names=label_names,
                         save_path=str(out / "cluster_heatmap.png"))
    plot_segment_distribution(segments_df,
                              save_path=str(out / "segment_distribution.png"))
    logger.info("All plots written to %s", out)
