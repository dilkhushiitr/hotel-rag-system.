"""
DBSCAN Clustering Module
─────────────────────────
Micro-segmentation and anomaly/fraud detection layer.
Label -1 = noise = fraud suspects, bots, reward exploiters.
"""

import logging
import pickle
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from src.config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, MODELS_DIR

logger = logging.getLogger(__name__)
MODEL_PATH = MODELS_DIR / "dbscan_model.pkl"


def estimate_eps(X_scaled: np.ndarray, min_samples: int = None,
                 plot_path: str = None) -> float:
    k = min_samples or DBSCAN_MIN_SAMPLES
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    kth_distances = np.sort(distances[:, k - 1])
    diff = np.diff(kth_distances)
    knee_idx = int(np.argmax(diff))
    suggested_eps = float(kth_distances[knee_idx])
    logger.info("Suggested eps: %.4f", suggested_eps)

    if plot_path:
        plt.figure(figsize=(10, 5))
        plt.plot(kth_distances, linewidth=1.5, color="steelblue")
        plt.axhline(suggested_eps, color="red", linestyle="--",
                    label=f"Suggested eps={suggested_eps:.3f}")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{k}-NN Distance")
        plt.title("DBSCAN: k-Distance Graph")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    return suggested_eps


def train_dbscan(X_scaled: np.ndarray, eps: float = None,
                 min_samples: int = None, save: bool = True) -> Tuple[DBSCAN, np.ndarray]:
    eps_val = eps or DBSCAN_EPS
    min_s   = min_samples or DBSCAN_MIN_SAMPLES
    logger.info("Training DBSCAN: eps=%.4f, min_samples=%d on %d samples",
                eps_val, min_s, len(X_scaled))

    model  = DBSCAN(eps=eps_val, min_samples=min_s, algorithm="auto", n_jobs=-1)
    labels = model.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    logger.info("DBSCAN done | clusters=%d | noise=%d (%.1f%%)",
                n_clusters, n_noise, 100 * n_noise / len(labels))

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info("DBSCAN model saved -> %s", MODEL_PATH)

    return model, labels


def load_dbscan() -> DBSCAN:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"DBSCAN model not found at {MODEL_PATH}.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def evaluate_dbscan(labels: np.ndarray, X_scaled: np.ndarray) -> Dict:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    noise_pct  = round(100 * n_noise / len(labels), 2)

    metrics = {
        "n_clusters":     n_clusters,
        "n_noise_points": n_noise,
        "noise_pct":      noise_pct,
        "silhouette_score": None,
    }

    mask = labels != -1
    if n_clusters >= 2 and mask.sum() > 100:
        sil = silhouette_score(X_scaled[mask], labels[mask],
                               sample_size=min(5000, mask.sum()))
        metrics["silhouette_score"] = round(float(sil), 4)

    logger.info("DBSCAN evaluation: %s", metrics)
    return metrics


def get_anomaly_profile(feature_df, labels: np.ndarray, top_n: int = 20):
    import pandas as pd
    mask     = labels == -1
    anomalies = feature_df[mask].copy()
    anomalies["dbscan_label"] = -1
    logger.info("Anomaly profile: %d users flagged", len(anomalies))
    return anomalies.head(top_n)
