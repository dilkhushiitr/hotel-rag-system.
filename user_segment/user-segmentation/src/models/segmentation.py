"""
Segmentation Orchestrator
──────────────────────────
Combines K-Means + DBSCAN outputs into a final user_segments table.

Responsibilities:
  1. Assign human-readable segment labels based on cluster profiles
  2. Generate per-segment statistics (used by the monitoring dashboard)
  3. Build the final output DataFrame ready for the ad targeting system
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import SEGMENT_LABELS
from src.features.engineering import FEATURE_COLS

logger = logging.getLogger(__name__)


# ── Cluster Profiling ──────────────────────────────────────────────────────────

def profile_clusters(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    label_col: str = "kmeans_cluster",
) -> pd.DataFrame:
    """
    Compute mean statistics per cluster.

    Parameters
    ----------
    feature_df : user-level feature DataFrame (with user_id)
    labels     : cluster label array (same row order as feature_df)
    label_col  : name to give the label column in the profile

    Returns
    -------
    DataFrame: one row per cluster, columns = feature means + cluster size
    """
    df = feature_df.copy()
    df[label_col] = labels

    numeric_cols = [c for c in FEATURE_COLS if c in df.columns]
    profile = df.groupby(label_col)[numeric_cols].mean().round(3)
    profile["cluster_size"] = df.groupby(label_col).size()
    profile["cluster_pct"]  = (profile["cluster_size"] / len(df) * 100).round(2)

    logger.info("Cluster profile computed:\n%s", profile[["cluster_size", "cluster_pct", "engagement_score", "monetization_score", "churn_risk_score"]].to_string())
    return profile.reset_index()


def auto_label_clusters(profile_df: pd.DataFrame, label_col: str = "kmeans_cluster") -> Dict[int, str]:
    """
    Automatically assign business labels based on cluster profiles.

    Rules (applied in priority order):
    - Highest monetization_score  → "High-Value Power Users"
    - Highest ctr                 → "Ad Hunters"
    - Highest churn_risk_score    → "Dormant Users"
    - Highest engagement_score    → "Active Casual Users"
    - Remainder                   → "At-Risk Users"

    Returns a dict mapping cluster_id → label string.
    """
    df = profile_df.set_index(label_col).copy()
    labels: Dict[int, str] = {}
    assigned = set()

    def assign(cluster_id: int, label: str):
        labels[cluster_id] = label
        assigned.add(cluster_id)

    remaining = list(df.index)

    # 1. High-Value Power Users
    top_mon = df["monetization_score"].idxmax()
    assign(top_mon, "High-Value Power Users")
    remaining = [c for c in remaining if c not in assigned]

    # 2. Ad Hunters
    if remaining:
        top_ctr = df.loc[remaining, "ctr"].idxmax()
        assign(top_ctr, "Ad Hunters")
        remaining = [c for c in remaining if c not in assigned]

    # 3. Dormant Users
    if remaining:
        top_churn = df.loc[remaining, "churn_risk_score"].idxmax()
        assign(top_churn, "Dormant Users")
        remaining = [c for c in remaining if c not in assigned]

    # 4. Active Casual Users
    if remaining:
        top_eng = df.loc[remaining, "engagement_score"].idxmax()
        assign(top_eng, "Active Casual Users")
        remaining = [c for c in remaining if c not in assigned]

    # 5. Everything else
    for c in remaining:
        assign(c, "At-Risk Users")

    logger.info("Auto-assigned segment labels: %s", labels)
    return labels


# ── Final Table Assembly ───────────────────────────────────────────────────────

def build_user_segments(
    feature_df: pd.DataFrame,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    cluster_label_map: Optional[Dict[int, str]] = None,
    run_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Build the final user_segments DataFrame.

    Columns
    -------
    user_id | kmeans_cluster | dbscan_cluster | segment_label |
    is_anomaly | engagement_score | monetization_score | churn_risk_score |
    last_updated

    Parameters
    ----------
    feature_df        : user-level features
    kmeans_labels     : K-Means cluster ids
    dbscan_labels     : DBSCAN labels (-1 = anomaly)
    cluster_label_map : {cluster_id: "Human Label"} (auto-derived if None)
    run_date          : pipeline run timestamp

    Returns
    -------
    user_segments DataFrame
    """
    if run_date is None:
        run_date = datetime.utcnow()

    # Auto-derive labels if not provided
    if cluster_label_map is None:
        profile = profile_clusters(feature_df, kmeans_labels)
        cluster_label_map = auto_label_clusters(profile)

    df = feature_df[["user_id"]].copy()
    df["kmeans_cluster"] = kmeans_labels
    df["dbscan_cluster"] = dbscan_labels
    df["is_anomaly"]     = (dbscan_labels == -1).astype(int)

    # Segment label: anomalies get special label
    df["segment_label"] = df["kmeans_cluster"].map(cluster_label_map)
    df.loc[df["is_anomaly"] == 1, "segment_label"] = "Anomaly / Fraud Suspect"

    # Carry over the three most important scores for ad system
    for col in ["engagement_score", "monetization_score", "churn_risk_score"]:
        if col in feature_df.columns:
            df[col] = feature_df[col].values

    df["last_updated"] = run_date

    logger.info("User segments built: %d rows", len(df))
    logger.info("Segment distribution:\n%s", df["segment_label"].value_counts().to_string())

    return df


def segment_summary(segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-segment summary table for the monitoring dashboard / reporting.
    """
    summary = (
        segments_df
        .groupby("segment_label")
        .agg(
            users               = ("user_id", "count"),
            pct                 = ("user_id", lambda x: round(100 * len(x) / len(segments_df), 2)),
            avg_engagement      = ("engagement_score", "mean"),
            avg_monetization    = ("monetization_score", "mean"),
            avg_churn_risk      = ("churn_risk_score", "mean"),
            anomaly_count       = ("is_anomaly", "sum"),
        )
        .round(2)
        .reset_index()
        .sort_values("users", ascending=False)
    )
    return summary
