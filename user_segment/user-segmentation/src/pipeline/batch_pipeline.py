"""
Batch Pipeline
──────────────
End-to-end orchestration:
  Load → Features → Preprocess → KMeans → DBSCAN → Segment → Store → Visualise

Designed to run:
  - Locally: `python -m src.pipeline.batch_pipeline`
  - Via Airflow DAG: imported as a callable task
  - Via Docker / cron: wrapped in scripts/run_pipeline.sh
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED, KMEANS_FINAL_K, LOG_FORMAT, LOG_LEVEL, MODELS_DIR
from src.data.ingestion import load_all, write_segments_to_db
from src.data.preprocessing import full_preprocessing_pipeline
from src.features.engineering import FEATURE_COLS, build_feature_matrix
from src.models.dbscan_model import evaluate_dbscan, train_dbscan
from src.models.kmeans_model import evaluate_kmeans, find_optimal_k, train_kmeans
from src.models.segmentation import (
    auto_label_clusters,
    build_user_segments,
    profile_clusters,
    segment_summary,
)
from src.visualization.plots import generate_all_plots

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_pipeline(
    source: str = "csv",
    fit_scaler: bool = True,
    find_k: bool = False,
    write_db: bool = False,
    plots_dir: str = "reports/plots",
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Execute the full segmentation pipeline.

    Parameters
    ----------
    source     : "csv" or "db" — data source
    fit_scaler : True  → fit + save a new scaler
                 False → load existing scaler (inference mode)
    find_k     : if True, run elbow analysis before training (slower)
    write_db   : if True, persist user_segments to PostgreSQL
    plots_dir  : directory to write visualisation plots
    save_csv   : if True, save user_segments.csv to data/processed/

    Returns
    -------
    user_segments DataFrame
    """
    run_start = time.time()
    run_date  = datetime.utcnow()
    logger.info("=" * 60)
    logger.info("Pipeline START  |  %s  |  source=%s", run_date.isoformat(), source)
    logger.info("=" * 60)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    logger.info("[1/7] Loading raw data …")
    tables = load_all(source=source)

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    logger.info("[2/7] Building feature matrix …")
    # In CSV/dev mode, use the latest date in the actual data so the
    # lookback window captures sample data (which may be in the past).
    from src.features.engineering import _filter_window
    sessions_max = tables["sessions"]["session_start"].max()
    data_end_date = sessions_max if pd.notna(sessions_max) else run_date
    logger.info("  Feature window ends: %s", data_end_date)
    feature_df = build_feature_matrix(tables, end_date=data_end_date)

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    logger.info("[3/7] Preprocessing features …")
    feature_cols_present = [c for c in FEATURE_COLS if c in feature_df.columns]
    df_clean, X_scaled, scaler = full_preprocessing_pipeline(
        feature_df,
        feature_cols=feature_cols_present,
        fit_scaler=fit_scaler,
    )

    # ── Step 4: K-Means ───────────────────────────────────────────────────────
    logger.info("[4/7] K-Means clustering …")

    if find_k:
        logger.info("  Running elbow analysis …")
        k_results = find_optimal_k(
            X_scaled,
            plot_path=str(Path(plots_dir) / "elbow_silhouette.png"),
        )
        k = k_results["suggested_k"]
        logger.info("  Auto-selected K=%d", k)
    else:
        k = KMEANS_FINAL_K

    kmeans_model   = train_kmeans(X_scaled, k=k, save=True)
    kmeans_labels  = kmeans_model.predict(X_scaled)
    kmeans_metrics = evaluate_kmeans(kmeans_model, X_scaled)
    logger.info("  K-Means metrics: %s", kmeans_metrics)

    # ── Step 5: DBSCAN ────────────────────────────────────────────────────────
    logger.info("[5/7] DBSCAN micro-segmentation …")
    dbscan_model, dbscan_labels = train_dbscan(X_scaled, save=True)
    dbscan_metrics = evaluate_dbscan(dbscan_labels, X_scaled)
    logger.info("  DBSCAN metrics: %s", dbscan_metrics)

    # ── Step 6: Segment Profiling & Assembly ──────────────────────────────────
    logger.info("[6/7] Profiling segments …")
    profile_df       = profile_clusters(df_clean, kmeans_labels)
    cluster_label_map = auto_label_clusters(profile_df)

    segments_df = build_user_segments(
        feature_df      = df_clean,
        kmeans_labels   = kmeans_labels,
        dbscan_labels   = dbscan_labels,
        cluster_label_map = cluster_label_map,
        run_date        = run_date,
    )

    summary = segment_summary(segments_df)
    logger.info("Segment summary:\n%s", summary.to_string(index=False))

    # ── Step 7: Output ────────────────────────────────────────────────────────
    logger.info("[7/7] Saving outputs …")

    if save_csv:
        out_path = DATA_PROCESSED / "user_segments.csv"
        segments_df.to_csv(out_path, index=False)
        logger.info("  Saved user_segments.csv → %s", out_path)

        summary_path = DATA_PROCESSED / "segment_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("  Saved segment_summary.csv → %s", summary_path)

    if write_db:
        write_segments_to_db(segments_df)

    # Visualisations
    try:
        generate_all_plots(
            X_scaled      = X_scaled,
            feature_df    = df_clean,
            kmeans_labels = kmeans_labels,
            dbscan_labels = dbscan_labels,
            segments_df   = segments_df,
            profile_df    = profile_df,
            label_names   = cluster_label_map,
            output_dir    = plots_dir,
        )
    except Exception as exc:
        logger.warning("Plot generation failed (non-fatal): %s", exc)

    elapsed = time.time() - run_start
    logger.info("=" * 60)
    logger.info("Pipeline COMPLETE  |  %.1f seconds", elapsed)
    logger.info("=" * 60)

    return segments_df


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the User Segmentation Pipeline")
    parser.add_argument("--source",    default="csv",    choices=["csv", "db"])
    parser.add_argument("--find-k",    action="store_true", help="Run elbow analysis")
    parser.add_argument("--write-db",  action="store_true", help="Write segments to PostgreSQL")
    parser.add_argument("--no-scaler-fit", action="store_true", help="Load existing scaler (inference mode)")
    parser.add_argument("--plots-dir", default="reports/plots")
    args = parser.parse_args()

    segments = run_pipeline(
        source     = args.source,
        fit_scaler = not args.no_scaler_fit,
        find_k     = args.find_k,
        write_db   = args.write_db,
        plots_dir  = args.plots_dir,
    )

    print(f"\nDone. {len(segments):,} users segmented.")
    print(segments["segment_label"].value_counts().to_string())
