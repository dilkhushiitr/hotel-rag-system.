"""Isolation Forest training, scoring, and persistence."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.feature_engineering import FEATURE_COLUMNS, feature_matrix


def build_model(config: dict) -> Pipeline:
    """Build the sklearn pipeline used in training and inference.

    Isolation Forest is tree-based and does not strictly require scaling, but
    scaling keeps feature ranges stable and makes the artifact easier to extend
    with other models later.
    """
    params = config["model"]
    model = IsolationForest(
        n_estimators=params["n_estimators"],
        contamination=params["contamination"],
        max_samples=params["max_samples"],
        max_features=params["max_features"],
        random_state=params["random_state"],
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def train_isolation_forest(features: pd.DataFrame, config: dict) -> Pipeline:
    """Train Isolation Forest on numeric fraud features."""
    pipeline = build_model(config)
    pipeline.fit(feature_matrix(features))
    return pipeline


def anomaly_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return normalized anomaly scores from 0 to 100.

    sklearn's `score_samples` returns higher values for more normal points.
    We invert that value so higher means more suspicious, then min-max scale it.
    """
    raw_scores = -model.score_samples(feature_matrix(features))
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    if np.isclose(max_score, min_score):
        return np.zeros_like(raw_scores)
    return 100.0 * (raw_scores - min_score) / (max_score - min_score)


def predict_anomalies(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return 1 for anomalous/fraud-like rows and 0 for normal rows."""
    predictions = model.predict(feature_matrix(features))
    return (predictions == -1).astype(int)


def evaluate_model(model: Pipeline, features: pd.DataFrame) -> dict[str, float]:
    """Evaluate model when the optional `is_fraud` label is available."""
    if "is_fraud" not in features.columns:
        return {}

    y_true = features["is_fraud"].astype(int)
    y_pred = predict_anomalies(model, features)
    y_score = anomaly_scores(model, features)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))

    return metrics


def save_model(model: Pipeline, path: str | Path) -> None:
    """Persist trained sklearn pipeline and feature metadata."""
    artifact = {
        "pipeline": model,
        "feature_columns": FEATURE_COLUMNS,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_model(path: str | Path) -> Pipeline:
    """Load trained sklearn pipeline from disk."""
    artifact = joblib.load(path)
    return artifact["pipeline"]

