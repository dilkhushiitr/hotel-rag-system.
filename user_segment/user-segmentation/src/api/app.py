"""
src/api/app.py

FastAPI REST API for serving user segments to the ad system.

Endpoints:
  GET  /health                  - liveness check
  GET  /segment/{user_id}       - get one user's segment
  POST /segment/batch           - bulk segment lookup
  GET  /segments/summary        - all segment stats
  POST /predict                 - score raw features (real-time)
  GET  /metrics                 - basic metrics for monitoring
"""

import logging
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.data.db import get_engine
from src.features.feature_engineering import MODEL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="User Segmentation API",
    description="Serves user segments for ad targeting. Backed by K-Means + DBSCAN.",
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

# ── Global state ──────────────────────────────────────────────────────────────
_kmeans_model  = None
_scaler        = None
_request_count = 0
_start_time    = time.time()


def load_models():
    global _kmeans_model, _scaler
    models_dir = os.getenv("MODELS_DIR", "data/models")

    km_path  = os.path.join(models_dir, "kmeans_model.pkl")
    sca_path = os.path.join(models_dir, "scaler.pkl")

    if os.path.exists(km_path):
        with open(km_path, "rb") as f:
            _kmeans_model = pickle.load(f)
        log.info("Loaded K-Means model")

    if os.path.exists(sca_path):
        with open(sca_path, "rb") as f:
            _scaler = pickle.load(f)
        log.info("Loaded scaler")


@app.on_event("startup")
def startup_event():
    load_models()
    log.info("API started")


# ── Pydantic models ───────────────────────────────────────────────────────────
class SegmentResponse(BaseModel):
    user_id:       str
    segment_label: str
    bid_strategy:  str
    is_anomaly:    int
    updated_at:    Optional[str]


class BatchRequest(BaseModel):
    user_ids: List[str] = Field(..., max_items=1000, description="Up to 1000 user IDs")


class BatchResponse(BaseModel):
    results:  List[SegmentResponse]
    found:    int
    missing:  int


class PredictRequest(BaseModel):
    sessions_per_user:    float = Field(0, ge=0)
    avg_session_duration: float = Field(0, ge=0)
    active_days:          float = Field(0, ge=0)
    recency_days:         float = Field(30, ge=0)
    ctr:                  float = Field(0, ge=0, le=1)
    total_impressions:    float = Field(0, ge=0)
    ad_revenue:           float = Field(0, ge=0)
    wallet_balance:       float = Field(0, ge=0)
    total_earnings:       float = Field(0, ge=0)
    total_redemptions:    float = Field(0, ge=0)
    avg_transaction_value: float = Field(0, ge=0)
    success_rate:         float = Field(0, ge=0, le=1)
    engagement_score:     float = Field(0, ge=0)
    monetization_score:   float = Field(0, ge=0)
    churn_risk_score:     float = Field(0, ge=0, le=1)


class PredictResponse(BaseModel):
    cluster_id:    int
    segment_label: str
    bid_strategy:  str
    confidence:    Optional[float]


class SegmentSummary(BaseModel):
    segment_label:    str
    n_users:          int
    pct_users:        float
    avg_ctr:          float
    avg_revenue:      float
    avg_engagement:   float
    avg_churn_risk:   float
    bid_strategy:     str


# ── Helpers ───────────────────────────────────────────────────────────────────
BID_STRATEGY = {
    "Whales":         "HIGH",
    "Rising Stars":   "MEDIUM_HIGH",
    "Ad Hunters":     "MEDIUM",
    "Casual Earners": "LOW_MEDIUM",
    "Dormant Users":  "SUPPRESSED",
    "ANOMALY":        "BLOCKED",
}


def query_user_segment(user_id: str) -> Optional[Dict]:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM user_segments WHERE user_id = :uid LIMIT 1"),
                {"uid": user_id},
            )
            row = result.fetchone()
            return dict(row._mapping) if row else None
    except Exception as e:
        log.error(f"DB query failed: {e}")
        return None


def query_batch(user_ids: List[str]) -> Dict[str, Dict]:
    try:
        engine = get_engine()
        ids_str = ",".join([f"'{uid}'" for uid in user_ids])
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT * FROM user_segments WHERE user_id IN ({ids_str})")
            )
            return {row["user_id"]: dict(row._mapping) for row in result.fetchall()}
    except Exception as e:
        log.error(f"Batch DB query failed: {e}")
        return {}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status":        "healthy",
        "model_loaded":  _kmeans_model is not None,
        "scaler_loaded": _scaler is not None,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "timestamp":     datetime.now().isoformat(),
    }


@app.get("/metrics", tags=["System"])
def metrics():
    global _request_count
    return {
        "total_requests":  _request_count,
        "uptime_seconds":  round(time.time() - _start_time, 1),
        "model_loaded":    _kmeans_model is not None,
    }


@app.get("/segment/{user_id}", response_model=SegmentResponse, tags=["Segments"])
def get_user_segment(user_id: str):
    """Get the current segment for a single user."""
    global _request_count
    _request_count += 1

    row = query_user_segment(user_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found in segment table")

    return SegmentResponse(
        user_id       = row["user_id"],
        segment_label = row.get("segment_label", "Unknown"),
        bid_strategy  = row.get("bid_strategy", "LOW_MEDIUM"),
        is_anomaly    = int(row.get("is_anomaly", 0)),
        updated_at    = str(row.get("updated_at", "")),
    )


@app.post("/segment/batch", response_model=BatchResponse, tags=["Segments"])
def get_batch_segments(request: BatchRequest):
    """Bulk segment lookup. Up to 1000 user IDs per request."""
    global _request_count
    _request_count += 1

    results_map = query_batch(request.user_ids)
    results     = []
    missing     = 0

    for uid in request.user_ids:
        if uid in results_map:
            row = results_map[uid]
            results.append(SegmentResponse(
                user_id       = uid,
                segment_label = row.get("segment_label", "Unknown"),
                bid_strategy  = row.get("bid_strategy", "LOW_MEDIUM"),
                is_anomaly    = int(row.get("is_anomaly", 0)),
                updated_at    = str(row.get("updated_at", "")),
            ))
        else:
            missing += 1
            results.append(SegmentResponse(
                user_id       = uid,
                segment_label = "Unknown",
                bid_strategy  = "LOW_MEDIUM",
                is_anomaly    = 0,
                updated_at    = None,
            ))

    return BatchResponse(results=results, found=len(results_map), missing=missing)


@app.get("/segments/summary", response_model=List[SegmentSummary], tags=["Segments"])
def get_segments_summary():
    """Return KPI summary for all segments."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM segment_profiles ORDER BY n_users DESC"))
            rows   = result.fetchall()
            if not rows:
                return []
            return [
                SegmentSummary(
                    segment_label  = r["segment_label"],
                    n_users        = int(r.get("n_users", 0)),
                    pct_users      = float(r.get("pct_users", 0)),
                    avg_ctr        = float(r.get("avg_ctr", 0)),
                    avg_revenue    = float(r.get("avg_revenue", 0)),
                    avg_engagement = float(r.get("avg_engagement", 0)),
                    avg_churn_risk = float(r.get("avg_churn_risk", 0)),
                    bid_strategy   = str(r.get("bid_strategy", "LOW_MEDIUM")),
                )
                for r in rows
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["Predict"])
def predict_segment(request: PredictRequest):
    """
    Real-time segment prediction for a new user based on raw features.
    Does NOT write to DB — for inference only.
    """
    global _request_count
    _request_count += 1

    if _kmeans_model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run the pipeline first.")

    # Build feature vector
    features = {feat: getattr(request, feat, 0) for feat in MODEL_FEATURES}
    X_df     = pd.DataFrame([features])

    # Scale
    X_scaled = _scaler.transform(X_df[MODEL_FEATURES])

    # Predict
    cluster_id = int(_kmeans_model.predict(X_scaled)[0])

    # Map label
    label_map = {
        0: "Whales",
        1: "Casual Earners",
        2: "Ad Hunters",
        3: "Dormant Users",
        4: "Rising Stars",
    }
    segment_label = label_map.get(cluster_id, f"Cluster_{cluster_id}")
    bid_strategy  = BID_STRATEGY.get(segment_label, "LOW_MEDIUM")

    # Distance to centroid as confidence proxy (inverse of normalized distance)
    centers   = _kmeans_model.cluster_centers_
    dists     = np.linalg.norm(X_scaled - centers[cluster_id], axis=1)
    confidence = round(float(1 / (1 + dists[0])), 4)

    return PredictResponse(
        cluster_id    = cluster_id,
        segment_label = segment_label,
        bid_strategy  = bid_strategy,
        confidence    = confidence,
    )


@app.post("/reload-models", tags=["System"])
def reload_models():
    """Hot-reload models from disk without restarting the API."""
    load_models()
    return {"message": "Models reloaded", "model_loaded": _kmeans_model is not None}
