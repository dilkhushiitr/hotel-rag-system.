"""FastAPI deployment layer for real-time segment scoring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from user_segmentation.config import load_config
from user_segmentation.modeling import load_artifacts
from user_segmentation.pipeline import predict_from_feature_frame


class FeaturePayload(BaseModel):
    """Single user feature payload expected by the API.

    The API accepts engineered features rather than raw event tables because
    online systems usually score users after a feature store has already
    materialized session, ad, wallet, and transaction aggregates.
    """

    user_id: str
    features: dict[str, float] = Field(..., description="Feature dictionary matching config clustering_columns")


class BatchFeaturePayload(BaseModel):
    """Batch request body for multiple users."""

    records: list[FeaturePayload]


app = FastAPI(
    title="GreedyGame User Segmentation API",
    description="K-Means macro segmentation with DBSCAN anomaly detection for ad targeting.",
    version="1.0.0",
)

CONFIG_PATH = os.getenv("SEGMENTATION_CONFIG", "configs/config.yaml")
ARTIFACT_PATH = os.getenv("SEGMENTATION_ARTIFACT", "models/segmentation_artifacts.joblib")

CONFIG = None
ARTIFACTS: dict[str, Any] | None = None


@app.on_event("startup")
def startup() -> None:
    """Load config and model once when the API process starts."""

    global CONFIG, ARTIFACTS
    CONFIG = load_config(CONFIG_PATH)
    artifact_path = Path(ARTIFACT_PATH)
    if not artifact_path.exists():
        raise RuntimeError(f"Model artifact not found at {artifact_path}. Train the model before starting the API.")
    ARTIFACTS = load_artifacts(artifact_path)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health endpoint for Docker, Kubernetes, and load balancers."""

    return {"status": "ok"}


@app.post("/predict")
def predict(payload: BatchFeaturePayload) -> dict[str, list[dict[str, Any]]]:
    """Return segment predictions for engineered user features."""

    if CONFIG is None or ARTIFACTS is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    rows: list[dict[str, Any]] = []
    for record in payload.records:
        row = {"user_id": record.user_id}
        row.update(record.features)
        rows.append(row)

    features = pd.DataFrame(rows)

    missing = sorted(set(CONFIG.clustering_columns) - set(features.columns))
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")

    scored = predict_from_feature_frame(features, ARTIFACTS)
    response_columns = ["user_id", "kmeans_cluster", "segment_label", "dbscan_cluster", "is_dbscan_outlier"]
    return {"predictions": scored[response_columns].to_dict(orient="records")}

