"""
Segmentation REST API
──────────────────────
FastAPI application exposing:
  GET  /health                   → health check
  GET  /segments/{user_id}       → fetch segment for one user
  POST /segments/batch           → bulk lookup (list of user_ids)
  GET  /segments/summary         → aggregated segment stats
  POST /predict                  → run inference on posted feature vector
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import API_DEBUG, API_HOST, API_PORT, DATA_PROCESSED, MODELS_DIR

logger = logging.getLogger(__name__)

app = FastAPI(
    title="User Segmentation API",
    description="Serve K-Means cluster labels and segment scores for ad targeting.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load data at startup ───────────────────────────────────────────────────────
_segments_df: Optional[pd.DataFrame] = None
_kmeans_model = None
_scaler       = None


@app.on_event("startup")
def load_artifacts():
    global _segments_df, _kmeans_model, _scaler

    segments_path = DATA_PROCESSED / "user_segments.csv"
    if segments_path.exists():
        _segments_df = pd.read_csv(segments_path)
        logger.info("Loaded %d user segments", len(_segments_df))
    else:
        logger.warning("user_segments.csv not found. Run pipeline first.")

    kmeans_path = MODELS_DIR / "kmeans_model.pkl"
    if kmeans_path.exists():
        with open(kmeans_path, "rb") as f:
            _kmeans_model = pickle.load(f)
        logger.info("K-Means model loaded")

    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)
        logger.info("Scaler loaded")


# ── Schemas ────────────────────────────────────────────────────────────────────

class UserSegmentResponse(BaseModel):
    user_id:            str
    kmeans_cluster:     int
    dbscan_cluster:     int
    segment_label:      str
    is_anomaly:         int
    engagement_score:   float
    monetization_score: float
    churn_risk_score:   float
    last_updated:       str


class BatchRequest(BaseModel):
    user_ids: List[str]


class PredictRequest(BaseModel):
    """
    Raw feature vector for real-time inference.
    Pass normalised values; the API will apply the saved scaler.
    """
    session_count:          float = 0
    avg_session_duration:   float = 0
    total_session_duration: float = 0
    active_days:            float = 0
    recency_days:           float = 30
    pages_per_session:      float = 0
    total_impressions:      float = 0
    total_clicks:           float = 0
    ctr:                    float = 0
    total_earnings:         float = 0
    total_redeemed:         float = 0
    wallet_balance:         float = 0
    redemption_rate:        float = 0
    total_transactions:     float = 0
    total_txn_amount:       float = 0
    avg_txn_amount:         float = 0
    txn_success_rate:       float = 0
    engagement_score:       float = 0
    monetization_score:     float = 0
    churn_risk_score:       float = 0


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": _kmeans_model is not None,
        "segments_loaded": _segments_df is not None and len(_segments_df) > 0,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/segments/{user_id}", response_model=UserSegmentResponse)
def get_segment(user_id: str):
    if _segments_df is None:
        raise HTTPException(503, "Segment data not loaded. Run pipeline first.")

    row = _segments_df[_segments_df["user_id"] == user_id]
    if row.empty:
        raise HTTPException(404, f"User '{user_id}' not found in segment table.")

    r = row.iloc[0].to_dict()
    return UserSegmentResponse(**{k: (v if pd.notna(v) else 0) for k, v in r.items()})


@app.post("/segments/batch")
def get_segments_batch(request: BatchRequest):
    if _segments_df is None:
        raise HTTPException(503, "Segment data not loaded.")

    found = _segments_df[_segments_df["user_id"].isin(request.user_ids)]
    missing = set(request.user_ids) - set(found["user_id"])

    return {
        "found":   found.to_dict(orient="records"),
        "missing": list(missing),
        "total_requested": len(request.user_ids),
        "total_found":     len(found),
    }


@app.get("/segments/summary")
def get_summary():
    if _segments_df is None:
        raise HTTPException(503, "Segment data not loaded.")

    summary = (
        _segments_df
        .groupby("segment_label")
        .agg(
            count               = ("user_id", "count"),
            avg_engagement      = ("engagement_score", "mean"),
            avg_monetization    = ("monetization_score", "mean"),
            avg_churn_risk      = ("churn_risk_score", "mean"),
            anomaly_count       = ("is_anomaly", "sum"),
        )
        .round(2)
        .reset_index()
    )
    return summary.to_dict(orient="records")


@app.post("/predict")
def predict_segment(request: PredictRequest):
    """
    Real-time segment prediction for a single user given raw features.
    Applies saved scaler → K-Means model.
    """
    if _kmeans_model is None or _scaler is None:
        raise HTTPException(503, "Models not loaded. Run pipeline first.")

    feature_vector = np.array([[
        request.session_count, request.avg_session_duration,
        request.total_session_duration, request.active_days,
        request.recency_days, request.pages_per_session,
        request.total_impressions, request.total_clicks,
        request.ctr, request.total_earnings, request.total_redeemed,
        request.wallet_balance, request.redemption_rate,
        request.total_transactions, request.total_txn_amount,
        request.avg_txn_amount, request.txn_success_rate,
        request.engagement_score, request.monetization_score,
        request.churn_risk_score,
    ]])

    X_scaled = _scaler.transform(feature_vector)
    cluster  = int(_kmeans_model.predict(X_scaled)[0])

    return {
        "predicted_cluster": cluster,
        "note": "Run DBSCAN separately for anomaly detection on batch.",
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.app:app", host=API_HOST, port=API_PORT, reload=API_DEBUG)
