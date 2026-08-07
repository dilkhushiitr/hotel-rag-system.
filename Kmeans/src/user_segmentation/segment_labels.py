"""Business profiling and human-readable segment labels."""

from __future__ import annotations

import pandas as pd


def profile_segments(features: pd.DataFrame, cluster_column: str = "kmeans_cluster") -> pd.DataFrame:
    """Aggregate business metrics for every segment."""

    profile = features.groupby(cluster_column).agg(
        users=("user_id", "nunique"),
        avg_ctr=("ctr", "mean"),
        avg_revenue=("total_earnings", "mean"),
        avg_retention=("retention_rate", "mean"),
        avg_sessions=("session_count", "mean"),
        avg_active_days=("active_days", "mean"),
        avg_recency_days=("recency_days", "mean"),
        avg_churn_risk=("churn_risk_score", "mean"),
        avg_monetization=("monetization_score", "mean"),
        avg_engagement=("engagement_score", "mean"),
    ).reset_index()

    return profile.sort_values("users", ascending=False)


def assign_business_labels(features: pd.DataFrame, cluster_column: str = "kmeans_cluster") -> dict[int, str]:
    """Assign readable labels like High-value, Casual, Low engagement, Ad clickers.

    Labels are generated from actual segment behavior instead of assuming cluster
    0 always means the same thing. K-Means cluster IDs are arbitrary, so this
    mapping is essential for production interpretability.
    """

    profile = profile_segments(features, cluster_column)
    labels: dict[int, str] = {}

    high_value_cluster = profile.sort_values(["avg_revenue", "avg_monetization"], ascending=False).iloc[0][cluster_column]
    ad_clicker_cluster = profile.sort_values("avg_ctr", ascending=False).iloc[0][cluster_column]
    low_engagement_cluster = profile.sort_values(["avg_engagement", "avg_sessions"], ascending=True).iloc[0][cluster_column]

    for cluster_id in profile[cluster_column].tolist():
        cluster_id_int = int(cluster_id)
        if cluster_id == high_value_cluster:
            labels[cluster_id_int] = "High-value users"
        elif cluster_id == ad_clicker_cluster:
            labels[cluster_id_int] = "Ad clickers"
        elif cluster_id == low_engagement_cluster:
            labels[cluster_id_int] = "Low engagement / churn risk"
        else:
            labels[cluster_id_int] = "Casual users"

    return labels

