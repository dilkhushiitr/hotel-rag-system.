"""FastAPI application for real-time fraud scoring."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from fraud_detection.config import load_config
from fraud_detection.feature_engineering import build_features
from fraud_detection.model import anomaly_scores, load_model
from fraud_detection.monitoring import log_prediction
from fraud_detection.risk_scoring import combine_scores
from fraud_detection.rule_engine import score_rules
from fraud_detection.schema import RuleDetail, TransactionScoreRequest, TransactionScoreResponse


CONFIG_PATH = os.getenv("FRAUD_CONFIG_PATH", "configs/config.yaml")
CONFIG = load_config(CONFIG_PATH)
MODEL = None

REQUEST_COUNTER = Counter("fraud_score_requests_total", "Total number of fraud score requests.")
BLOCK_COUNTER = Counter("fraud_block_decisions_total", "Total number of BLOCK decisions.")
SCORE_LATENCY = Histogram("fraud_score_latency_seconds", "Latency for fraud scoring requests.")


def load_artifacts() -> None:
    """Load the model once when the API process starts."""
    global MODEL
    model_path = Path(CONFIG["paths"]["model_artifact"])
    if model_path.exists():
        MODEL = load_model(model_path)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan hook used for model loading and future cleanup."""
    load_artifacts()
    yield


app = FastAPI(
    title="Isolation Forest Fraud Detection API",
    version="1.0.0",
    description="Hybrid fraud scoring API using deterministic rules and Isolation Forest.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Health endpoint for load balancers and deployment checks."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_path": CONFIG["paths"]["model_artifact"],
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus-compatible metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/score", response_model=TransactionScoreResponse)
def score_transaction(request: TransactionScoreRequest) -> TransactionScoreResponse:
    """Score one transaction in real time.

    The request is already denormalized because online services often fetch
    feature-store values before calling a model API. For offline/batch scoring,
    use `scripts/batch_score.py`.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not loaded. Train the model first using scripts/train_model.py.",
        )

    REQUEST_COUNTER.inc()
    with SCORE_LATENCY.time():
        row = pd.DataFrame([request.model_dump()])
        features = build_features(row, CONFIG)
        scored = score_rules(features, CONFIG)
        scored["ml_anomaly_score"] = anomaly_scores(MODEL, scored).round(2)
        scored = combine_scores(scored, CONFIG)
        result = scored.iloc[0]

    if result["decision"] == "BLOCK":
        BLOCK_COUNTER.inc()

    log_prediction(
        {
            "txn_id": result["txn_id"],
            "user_id": result["user_id"],
            "rule_score": float(result["rule_score"]),
            "ml_anomaly_score": float(result["ml_anomaly_score"]),
            "final_fraud_score": float(result["final_fraud_score"]),
            "risk_level": result["risk_level"],
            "decision": result["decision"],
            "rules_fired": result["rules_fired"],
        },
        logs_dir=CONFIG["paths"]["logs_dir"],
    )

    return TransactionScoreResponse(
        txn_id=result["txn_id"],
        user_id=result["user_id"],
        rule_score=float(result["rule_score"]),
        ml_anomaly_score=float(result["ml_anomaly_score"]),
        final_fraud_score=float(result["final_fraud_score"]),
        risk_level=result["risk_level"],
        decision=result["decision"],
        rules_fired=list(result["rules_fired"]),
        rule_details=[
            RuleDetail(rule=rule, weight=float(weight))
            for rule, weight in result["rule_details"].items()
        ],
        explanation=result["explanation"],
    )
