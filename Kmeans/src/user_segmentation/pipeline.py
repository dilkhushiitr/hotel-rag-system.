"""End-to-end training and inference pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

from user_segmentation.config import ProjectConfig
from user_segmentation.data_loader import clean_tables, load_raw_tables
from user_segmentation.features import build_user_features
from user_segmentation.modeling import load_artifacts, save_artifacts, train_models
from user_segmentation.utils import ensure_directories


def train_pipeline(config: ProjectConfig, data_dir: str | Path | None = None) -> dict[str, Path]:
    """Run the full batch training pipeline and write outputs to disk."""

    ensure_directories(config.processed_data_dir, config.model_dir, config.report_dir)

    raw_tables = load_raw_tables(config, data_dir=data_dir)
    cleaned_tables = clean_tables(raw_tables)
    features = build_user_features(cleaned_tables, config)
    artifacts = train_models(features, config)

    artifact_path = config.model_dir / "segmentation_artifacts.joblib"
    save_artifacts(
        {
            key: value
            for key, value in artifacts.items()
            # Large tabular outputs are saved separately as CSV reports.
            if key not in {"trained_features", "segment_profile", "kmeans_selection", "dbscan_selection"}
        },
        artifact_path,
    )

    feature_path = config.processed_data_dir / "user_features_with_segments.csv"
    profile_path = config.report_dir / "segment_profile.csv"
    kmeans_report_path = config.report_dir / "kmeans_model_selection.csv"
    dbscan_report_path = config.report_dir / "dbscan_model_selection.csv"

    artifacts["trained_features"].to_csv(feature_path, index=False)
    artifacts["segment_profile"].to_csv(profile_path, index=False)
    artifacts["kmeans_selection"].to_csv(kmeans_report_path, index=False)
    artifacts["dbscan_selection"].to_csv(dbscan_report_path, index=False)

    return {
        "artifact": artifact_path,
        "features": feature_path,
        "segment_profile": profile_path,
        "kmeans_report": kmeans_report_path,
        "dbscan_report": dbscan_report_path,
    }


def predict_from_feature_frame(features: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Score an already engineered feature table."""

    preprocessor = artifacts["preprocessor"]
    matrix = preprocessor.transform(features)

    result = features.copy()
    result["kmeans_cluster"] = artifacts["kmeans"].predict(matrix)

    # DBSCAN from sklearn does not expose predict() because density-based
    # clustering depends on the training neighborhood graph. For serving, we use
    # a standard production approximation: assign each new user to the label of
    # the nearest trained core sample if it is within eps; otherwise mark it as
    # noise/outlier (-1). This keeps API predictions stable and avoids refitting
    # the anomaly detector for every request.
    result["dbscan_cluster"] = approximate_dbscan_predict(artifacts["dbscan"], matrix)
    result["segment_label"] = result["kmeans_cluster"].map(artifacts["business_labels"])
    result["is_dbscan_outlier"] = result["dbscan_cluster"].eq(-1)
    return result


def approximate_dbscan_predict(dbscan: Any, matrix: np.ndarray) -> np.ndarray:
    """Assign DBSCAN labels to new points using nearest trained core samples."""

    if not hasattr(dbscan, "components_") or len(dbscan.components_) == 0:
        return np.full(shape=len(matrix), fill_value=-1, dtype=int)

    nearest_core_index, nearest_distance = pairwise_distances_argmin_min(matrix, dbscan.components_)
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]
    labels = core_labels[nearest_core_index]
    labels = np.where(nearest_distance <= dbscan.eps, labels, -1)
    return labels.astype(int)


def batch_predict_pipeline(
    config: ProjectConfig,
    artifact_path: str | Path,
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Build features from CSVs and produce user-level segment predictions."""

    raw_tables = load_raw_tables(config, data_dir=data_dir)
    cleaned_tables = clean_tables(raw_tables)
    features = build_user_features(cleaned_tables, config)
    artifacts = load_artifacts(artifact_path)
    predictions = predict_from_feature_frame(features, artifacts)

    output = Path(output_path) if output_path else config.processed_data_dir / "segment_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    return output
