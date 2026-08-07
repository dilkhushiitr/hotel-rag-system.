"""
K-Means Clustering Module
──────────────────────────
Provides:
  - find_optimal_k: Elbow + Silhouette analysis
  - train_kmeans: fit and persist model
  - predict_clusters: assign labels
  - evaluate_kmeans: metrics
"""

import logging
import pickle
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import (
    KMEANS_FINAL_K, KMEANS_INIT, KMEANS_K_MAX, KMEANS_K_MIN,
    KMEANS_MAX_ITER, KMEANS_N_INIT, MODELS_DIR, RANDOM_STATE,
)

logger = logging.getLogger(__name__)
MODEL_PATH = MODELS_DIR / "kmeans_model.pkl"


def find_optimal_k(X_scaled: np.ndarray, k_range=None,
                   plot_path: str = None) -> Dict:
    if k_range is None:
        k_range = range(KMEANS_K_MIN, KMEANS_K_MAX + 1)

    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init=KMEANS_INIT, n_init=KMEANS_N_INIT,
                    max_iter=KMEANS_MAX_ITER, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        silhouettes.append(sil)
        logger.debug("K=%d | inertia=%.2f | silhouette=%.4f", k, km.inertia_, sil)

    k_values    = list(k_range)
    suggested_k = k_values[int(np.argmax(silhouettes))]
    logger.info("Suggested K=%d (silhouette=%.4f)", suggested_k, max(silhouettes))

    if plot_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("K-Means: Elbow & Silhouette", fontsize=14, fontweight="bold")

        ax1.plot(k_values, inertias, "b-o", linewidth=2, markersize=6)
        ax1.axvline(suggested_k, color="red", linestyle="--", label=f"K={suggested_k}")
        ax1.set_xlabel("K"); ax1.set_ylabel("Inertia (WCSS)")
        ax1.set_title("Elbow Method"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(k_values, silhouettes, "g-s", linewidth=2, markersize=6)
        ax2.axvline(suggested_k, color="red", linestyle="--", label=f"K={suggested_k}")
        ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Scores"); ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Elbow plot saved -> %s", plot_path)

    return {"k_values": k_values, "inertias": inertias,
            "silhouettes": silhouettes, "suggested_k": suggested_k}


def train_kmeans(X_scaled: np.ndarray, k: int = None, save: bool = True) -> KMeans:
    k = k or KMEANS_FINAL_K
    logger.info("Training K-Means K=%d on %d samples …", k, len(X_scaled))

    model = KMeans(n_clusters=k, init=KMEANS_INIT, n_init=KMEANS_N_INIT,
                   max_iter=KMEANS_MAX_ITER, random_state=RANDOM_STATE)
    model.fit(X_scaled)
    logger.info("K-Means trained | inertia=%.2f | n_iter=%d", model.inertia_, model.n_iter_)

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved -> %s", MODEL_PATH)

    return model


def load_kmeans() -> KMeans:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"K-Means model not found at {MODEL_PATH}.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_clusters(model: KMeans, X_scaled: np.ndarray) -> np.ndarray:
    labels = model.predict(X_scaled)
    unique, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        logger.debug("Cluster %d -> %d users (%.1f%%)", lbl, cnt, 100*cnt/len(labels))
    return labels


def evaluate_kmeans(model: KMeans, X_scaled: np.ndarray) -> Dict:
    labels = model.predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
    metrics = {"inertia": model.inertia_, "silhouette_score": round(sil, 4)}
    logger.info("K-Means evaluation: %s", metrics)
    return metrics
