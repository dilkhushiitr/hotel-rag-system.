"""Model selection, training, and artifact persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from user_segmentation.config import ProjectConfig
from user_segmentation.features import UserSegmentationPreprocessor
from user_segmentation.segment_labels import assign_business_labels, profile_segments


def choose_kmeans_model(matrix: np.ndarray, config: ProjectConfig) -> tuple[KMeans, pd.DataFrame]:
    """Train K-Means across a K range and pick the best silhouette score."""

    k_cfg = config.raw["model"]["kmeans"]
    min_k = int(k_cfg["min_k"])
    max_k = int(k_cfg["max_k"])
    random_state = int(config.raw["model"]["random_state"])
    n_init = int(k_cfg["n_init"])

    rows: list[dict[str, float]] = []
    best_model: KMeans | None = None
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        if len(matrix) <= k:
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(score)})

        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise ValueError("Not enough users to train K-Means. Provide at least min_k + 1 users.")

    return best_model, pd.DataFrame(rows)


def choose_dbscan_model(matrix: np.ndarray, config: ProjectConfig) -> tuple[DBSCAN, pd.DataFrame]:
    """Select DBSCAN eps using silhouette on non-noise clusters when possible."""

    db_cfg = config.raw["model"]["dbscan"]
    min_samples = int(db_cfg["min_samples"])

    rows: list[dict[str, float]] = []
    best_model: DBSCAN | None = None
    best_score = -1.0

    for eps in db_cfg["eps_values"]:
        model = DBSCAN(eps=float(eps), min_samples=min_samples)
        labels = model.fit_predict(matrix)
        cluster_labels = set(labels) - {-1}
        noise_rate = float(np.mean(labels == -1))

        # Silhouette is only valid when at least two non-noise clusters exist.
        if len(cluster_labels) >= 2:
            non_noise_mask = labels != -1
            score = silhouette_score(matrix[non_noise_mask], labels[non_noise_mask])
        else:
            score = -1.0

        rows.append(
            {
                "eps": float(eps),
                "min_samples": min_samples,
                "clusters": len(cluster_labels),
                "noise_rate": noise_rate,
                "silhouette": float(score),
            }
        )

        # Prefer useful micro-clusters with a reasonable silhouette. If all eps
        # values fail to form two clusters, we still keep the last trained model.
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        best_model = DBSCAN(eps=float(db_cfg["eps_values"][0]), min_samples=min_samples).fit(matrix)

    return best_model, pd.DataFrame(rows)


def train_models(features: pd.DataFrame, config: ProjectConfig) -> dict[str, Any]:
    """Train preprocessing, K-Means macro segments, and DBSCAN micro segments."""

    preprocessor = UserSegmentationPreprocessor(
        feature_columns=config.clustering_columns,
        outlier_iqr_multiplier=float(config.raw["features"]["outlier_iqr_multiplier"]),
    )
    matrix = preprocessor.fit_transform(features)

    kmeans, kmeans_selection = choose_kmeans_model(matrix, config)
    dbscan, dbscan_selection = choose_dbscan_model(matrix, config)

    trained_features = features.copy()
    trained_features["kmeans_cluster"] = kmeans.predict(matrix)
    trained_features["dbscan_cluster"] = dbscan.fit_predict(matrix)

    labels = assign_business_labels(trained_features)
    trained_features["segment_label"] = trained_features["kmeans_cluster"].map(labels)
    trained_features["is_dbscan_outlier"] = trained_features["dbscan_cluster"].eq(-1)

    segment_profile = profile_segments(trained_features)
    segment_profile["segment_label"] = segment_profile["kmeans_cluster"].map(labels)

    return {
        "preprocessor": preprocessor,
        "kmeans": kmeans,
        "dbscan": dbscan,
        "business_labels": labels,
        "feature_columns": config.clustering_columns,
        "trained_features": trained_features,
        "segment_profile": segment_profile,
        "kmeans_selection": kmeans_selection,
        "dbscan_selection": dbscan_selection,
    }


def save_artifacts(artifacts: dict[str, Any], artifact_path: Path) -> None:
    """Persist model artifacts with joblib."""

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, artifact_path)


def load_artifacts(artifact_path: str | Path) -> dict[str, Any]:
    """Load model artifacts for batch inference or API serving."""

    return joblib.load(Path(artifact_path))

