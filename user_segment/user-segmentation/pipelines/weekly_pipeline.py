"""
pipelines/weekly_pipeline.py

Orchestrates the full end-to-end segmentation pipeline.
Can be triggered manually or by Airflow.

Steps:
  1. Extract 30-day data from DB
  2. Feature engineering
  3. Find optimal K (first run) or use configured K
  4. Fit K-Means
  5. Fit DBSCAN
  6. Profile segments + auto-label
  7. Write user_segments to DB
  8. Save models to disk
  9. Log run metrics
"""

import logging
import os
import pickle
import sys
import time
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import yaml

from src.data.db import get_engine, test_connection
from src.data.sql_queries import extract_all_features
from src.features.feature_engineering import build_feature_matrix, MODEL_FEATURES
from src.models.kmeans_model import KMeansSegmenter
from src.models.dbscan_model import DBSCANSegmenter
from src.analysis.segment_profiler import SegmentProfiler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/pipeline_run.log"),
    ],
)
log = logging.getLogger("pipeline")


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict = None, find_k: bool = False):
    """
    Main pipeline entry point.

    Args:
        config:   Config dict (loads from YAML if None)
        find_k:   If True, run K search before fitting. Slow but thorough.
    """
    config    = config or load_config()
    start_t   = time.time()
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("=" * 70)
    log.info(f"PIPELINE START | run_id={run_id}")
    log.info("=" * 70)

    # ── 1. DB connectivity check ───────────────────────────────────────────────
    log.info("Step 1/9: Checking DB connection...")
    if not test_connection():
        log.error("DB connection failed. Aborting pipeline.")
        return False

    # ── 2. Data extraction ────────────────────────────────────────────────────
    log.info("Step 2/9: Extracting features from DB...")
    lookback_days = config.get("data", {}).get("lookback_days", 30)
    raw_df = extract_all_features(lookback_days=lookback_days)
    log.info(f"  Extracted {len(raw_df):,} users")

    if len(raw_df) == 0:
        log.error("No data returned. Check DB tables. Aborting.")
        return False

    # ── 3. Feature engineering ────────────────────────────────────────────────
    log.info("Step 3/9: Running feature engineering pipeline...")
    features_df, enriched_df, scaler = build_feature_matrix(
        raw_df,
        fit_scaler=True,
        save_processed=True,
    )

    # Drop user_id from feature matrix before clustering
    X = features_df.drop(columns=["user_id"], errors="ignore").values
    log.info(f"  Feature matrix shape: {X.shape}")

    # ── 4. K-Means ────────────────────────────────────────────────────────────
    log.info("Step 4/9: Fitting K-Means...")
    km_config = config.get("kmeans", {})
    segmenter = KMeansSegmenter(
        n_clusters   = km_config.get("n_clusters", 5),
        random_state = km_config.get("random_state", 42),
        n_init       = km_config.get("n_init", 10),
        max_iter     = km_config.get("max_iter", 300),
        k_range      = range(*km_config.get("k_range", [2, 11])),
    )

    if find_k:
        log.info("  Searching for optimal K (this may take a few minutes)...")
        best_k = segmenter.find_optimal_k(X)
        log.info(f"  Optimal K = {best_k}")
    else:
        best_k = km_config.get("n_clusters", 5)
        log.info(f"  Using configured K = {best_k}")

    segmenter.fit(X, n_clusters=best_k)
    kmeans_labels = segmenter.predict(X)
    segmenter.plot_clusters_pca(X, kmeans_labels)

    # ── 5. DBSCAN ─────────────────────────────────────────────────────────────
    log.info("Step 5/9: Fitting DBSCAN for anomaly detection...")
    db_config   = config.get("dbscan", {})
    db_segmenter = DBSCANSegmenter(
        eps        = db_config.get("eps", 0.5),
        min_samples= db_config.get("min_samples", 5),
    )
    db_segmenter.plot_kdistance(X, save=True)
    db_segmenter.fit(X)
    dbscan_labels = db_segmenter.labels_
    db_segmenter.plot_clusters_pca(X)

    anomalies = db_segmenter.get_anomalies(enriched_df)
    log.info(f"  Anomalous users flagged: {len(anomalies):,}")

    # ── 6. Segment profiling ───────────────────────────────────────────────────
    log.info("Step 6/9: Profiling segments...")
    profiler = SegmentProfiler()
    user_segments_df = profiler.profile(
        enriched_df   = enriched_df,
        kmeans_labels = kmeans_labels,
        dbscan_labels = dbscan_labels,
        auto_label    = True,
    )

    # ── 7. Write to DB ────────────────────────────────────────────────────────
    log.info("Step 7/9: Writing results to DB...")
    profiler.write_to_db(user_segments_df)

    # ── 8. Save models ────────────────────────────────────────────────────────
    log.info("Step 8/9: Saving models to disk...")
    os.makedirs("data/models", exist_ok=True)
    segmenter.save()
    db_segmenter.save()

    # Save scaler separately for API use
    scaler_path = "data/models/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"  Saved scaler -> {scaler_path}")

    # Save run metadata
    run_meta = {
        "run_id":        run_id,
        "n_users":       len(raw_df),
        "n_clusters":    best_k,
        "n_anomalies":   len(anomalies),
        "duration_sec":  round(time.time() - start_t, 2),
        "timestamp":     datetime.now().isoformat(),
        "features":      MODEL_FEATURES,
        "segments":      user_segments_df["segment_label"].value_counts().to_dict(),
    }
    meta_path = f"data/models/run_meta_{run_id}.json"
    import json
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    # ── 9. Summary ────────────────────────────────────────────────────────────
    elapsed = round(time.time() - start_t, 2)
    log.info("=" * 70)
    log.info(f"PIPELINE COMPLETE | run_id={run_id} | {elapsed}s")
    log.info(f"  Users processed : {len(raw_df):,}")
    log.info(f"  K-Means clusters: {best_k}")
    log.info(f"  DBSCAN anomalies: {len(anomalies):,}")
    log.info("  Segment distribution:")
    for seg, cnt in user_segments_df["segment_label"].value_counts().items():
        log.info(f"    {seg:<20} {cnt:>7,} users")
    log.info("=" * 70)

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="User Segmentation Pipeline")
    parser.add_argument("--find-k",  action="store_true",
                        help="Search for optimal K (slower, use on first run)")
    args = parser.parse_args()

    success = run_pipeline(find_k=args.find_k)
    sys.exit(0 if success else 1)
