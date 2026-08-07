"""
src/analysis/segment_profiler.py

Converts raw cluster IDs into business-meaningful segment profiles.
ML clusters are useless unless business teams understand them.

Outputs:
  - Segment labels (Whales, Casual Earners, etc.)
  - Per-segment KPI summary
  - Bid strategy recommendation
  - user_segments table write
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.db import get_engine

log = logging.getLogger(__name__)

# ── Default label map (overridden after profiling) ────────────────────────────
DEFAULT_LABELS = {
    0: "Whales",
    1: "Casual Earners",
    2: "Ad Hunters",
    3: "Dormant Users",
    4: "Rising Stars",
}

BID_STRATEGY = {
    "Whales":         "HIGH",
    "Rising Stars":   "MEDIUM_HIGH",
    "Ad Hunters":     "MEDIUM",
    "Casual Earners": "LOW_MEDIUM",
    "Dormant Users":  "SUPPRESSED",
    "ANOMALY":        "BLOCKED",
}

PROFILE_FEATURES = [
    "sessions_per_user",
    "avg_session_duration",
    "active_days",
    "ctr",
    "ad_revenue",
    "total_earnings",
    "wallet_balance",
    "engagement_score",
    "monetization_score",
    "churn_risk_score",
]


class SegmentProfiler:
    """
    Takes enriched DataFrame with cluster labels and produces:
      1. segment_summary  - KPI table per segment
      2. user_segments    - user_id | segment_label | bid_strategy | ...
      3. anomaly_users    - users flagged by DBSCAN (noise)
    """

    def __init__(self, label_map: Dict[int, str] = None):
        self.label_map     = label_map or DEFAULT_LABELS
        self.segment_summary: Optional[pd.DataFrame] = None

    # ── Auto-label clusters based on KPI ranking ──────────────────────────────
    def auto_label_clusters(self, df: pd.DataFrame, cluster_col: str = "kmeans_cluster") -> Dict[int, str]:
        """
        Automatically assign human labels based on:
          - engagement_score  → activity
          - monetization_score → revenue
          - churn_risk_score  → at risk

        Cluster with highest monetization → "Whales"
        Cluster with highest churn risk   → "Dormant Users"
        Cluster with highest CTR          → "Ad Hunters"
        Cluster with youngest users (low sessions, growing) → "Rising Stars"
        Rest → "Casual Earners"
        """
        if cluster_col not in df.columns:
            log.warning(f"Column {cluster_col} not found. Using default labels.")
            return self.label_map

        summary = df.groupby(cluster_col).agg(
            monetization=("monetization_score", "mean"),
            churn_risk=("churn_risk_score",   "mean"),
            ctr=("ctr",                        "mean"),
            engagement=("engagement_score",    "mean"),
            sessions=("sessions_per_user",     "mean"),
        )

        labels: Dict[int, str] = {}
        used   = set()

        def assign(cluster_id: int, label: str):
            labels[cluster_id] = label
            used.add(cluster_id)
            log.info(f"  Cluster {cluster_id} → {label}")

        log.info("Auto-labeling clusters:")

        # Whales = highest monetization
        whale = summary["monetization"].idxmax()
        assign(whale, "Whales")

        # Dormant = highest churn risk (from remaining)
        remaining = summary.drop(index=list(used))
        dormant   = remaining["churn_risk"].idxmax()
        assign(dormant, "Dormant Users")

        # Ad Hunters = highest CTR (from remaining)
        remaining = summary.drop(index=list(used))
        ad_hunt   = remaining["ctr"].idxmax()
        assign(ad_hunt, "Ad Hunters")

        # Rising Stars = moderate engagement + low churn risk (from remaining)
        remaining = summary.drop(index=list(used))
        if len(remaining) >= 2:
            rising = (remaining["engagement"] - remaining["churn_risk"]).idxmax()
            assign(rising, "Rising Stars")

        # Rest = Casual Earners
        for cid in summary.index:
            if cid not in labels:
                assign(cid, "Casual Earners")

        self.label_map = labels
        return labels

    # ── Profile segments ──────────────────────────────────────────────────────
    def profile(
        self,
        enriched_df: pd.DataFrame,
        kmeans_labels: np.ndarray,
        dbscan_labels: np.ndarray = None,
        auto_label: bool = True,
    ) -> pd.DataFrame:
        """
        Main profiling function.
        Returns user_segments DataFrame ready for DB write.
        """
        df = enriched_df.copy()
        df["kmeans_cluster"] = kmeans_labels

        # DBSCAN anomaly flag
        if dbscan_labels is not None:
            df["is_anomaly"] = (dbscan_labels == -1).astype(int)
        else:
            df["is_anomaly"] = 0

        # Auto or manual labeling
        if auto_label:
            self.auto_label_clusters(df)

        df["segment_label"] = df["kmeans_cluster"].map(self.label_map).fillna("Unknown")
        df.loc[df["is_anomaly"] == 1, "segment_label"] = "ANOMALY"
        df["bid_strategy"]  = df["segment_label"].map(BID_STRATEGY).fillna("LOW_MEDIUM")
        df["updated_at"]    = datetime.now()

        # Build summary
        self.segment_summary = self._compute_summary(df)
        self._print_summary()
        self._plot_segment_overview(df)

        return df

    def _compute_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        agg_dict = {}
        for feat in PROFILE_FEATURES:
            if feat in df.columns:
                agg_dict[feat] = ("mean")

        summary = df.groupby("segment_label").agg(
            n_users        =("user_id",           "count"),
            avg_ctr        =("ctr",                lambda x: round(x.mean(), 4)),
            avg_revenue    =("ad_revenue",         lambda x: round(x.mean(), 2)),
            avg_engagement =("engagement_score",   lambda x: round(x.mean(), 2)),
            avg_monetization=("monetization_score",lambda x: round(x.mean(), 2)),
            avg_churn_risk =("churn_risk_score",   lambda x: round(x.mean(), 4)),
            avg_sessions   =("sessions_per_user",  lambda x: round(x.mean(), 1)),
            avg_active_days=("active_days",        lambda x: round(x.mean(), 1)),
        ).reset_index()

        summary["pct_users"]     = (summary["n_users"] / summary["n_users"].sum() * 100).round(1)
        summary["bid_strategy"]  = summary["segment_label"].map(BID_STRATEGY).fillna("LOW_MEDIUM")
        return summary

    def _print_summary(self):
        log.info("\n" + "=" * 70)
        log.info("SEGMENT PROFILE SUMMARY")
        log.info("=" * 70)
        if self.segment_summary is not None:
            for _, row in self.segment_summary.iterrows():
                log.info(f"\n  Segment: {row['segment_label']} ({row['pct_users']}% of users)")
                log.info(f"    Users         : {row['n_users']:,}")
                log.info(f"    Avg CTR       : {row['avg_ctr']:.4f}")
                log.info(f"    Avg Revenue   : ${row['avg_revenue']:.2f}")
                log.info(f"    Engagement    : {row['avg_engagement']:.1f}/10")
                log.info(f"    Churn Risk    : {row['avg_churn_risk']:.2f}")
                log.info(f"    Bid Strategy  : {row['bid_strategy']}")
        log.info("=" * 70)

    def _plot_segment_overview(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Segment Overview Dashboard", fontsize=16, fontweight="bold")

        # 1. User distribution
        seg_counts = df["segment_label"].value_counts()
        axes[0, 0].pie(seg_counts.values, labels=seg_counts.index, autopct="%1.1f%%",
                       colors=plt.cm.Set3.colors[:len(seg_counts)])
        axes[0, 0].set_title("User Distribution by Segment")

        # 2. CTR by segment
        if "ctr" in df.columns:
            seg_ctr = df.groupby("segment_label")["ctr"].mean().sort_values(ascending=False)
            axes[0, 1].bar(seg_ctr.index, seg_ctr.values, color="steelblue", alpha=0.8)
            axes[0, 1].set_title("Average CTR by Segment")
            axes[0, 1].set_xlabel("Segment"); axes[0, 1].set_ylabel("CTR")
            axes[0, 1].tick_params(axis="x", rotation=30)

        # 3. Engagement vs Monetization scatter
        if "engagement_score" in df.columns and "monetization_score" in df.columns:
            sample = df.sample(min(2000, len(df)), random_state=42)
            unique_segs = sample["segment_label"].unique()
            colors_map  = dict(zip(unique_segs, plt.cm.tab10.colors[:len(unique_segs)]))
            for seg in unique_segs:
                mask = sample["segment_label"] == seg
                axes[1, 0].scatter(sample.loc[mask, "engagement_score"],
                                   sample.loc[mask, "monetization_score"],
                                   label=seg, alpha=0.6, s=20,
                                   color=colors_map[seg])
            axes[1, 0].set_xlabel("Engagement Score"); axes[1, 0].set_ylabel("Monetization Score")
            axes[1, 0].set_title("Engagement vs Monetization"); axes[1, 0].legend(fontsize=7)

        # 4. Churn risk distribution
        if "churn_risk_score" in df.columns:
            for seg in df["segment_label"].unique():
                mask = df["segment_label"] == seg
                axes[1, 1].hist(df.loc[mask, "churn_risk_score"], alpha=0.5,
                                label=seg, bins=30, density=True)
            axes[1, 1].set_xlabel("Churn Risk Score"); axes[1, 1].set_ylabel("Density")
            axes[1, 1].set_title("Churn Risk Distribution"); axes[1, 1].legend(fontsize=7)

        plt.tight_layout()
        path = "data/models/segment_overview.png"
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"Saved segment overview -> {path}")

    # ── DB write ──────────────────────────────────────────────────────────────
    def write_to_db(self, user_segments_df: pd.DataFrame):
        """Write user_segments and segment_profiles to Postgres."""
        engine = get_engine()

        # user_segments table
        cols = ["user_id", "segment_label", "bid_strategy", "is_anomaly",
                "kmeans_cluster", "updated_at"]
        available = [c for c in cols if c in user_segments_df.columns]
        user_segments_df[available].to_sql(
            "user_segments", engine, if_exists="replace", index=False, chunksize=1000
        )
        log.info(f"Wrote {len(user_segments_df):,} rows -> user_segments table")

        # segment_profiles table
        if self.segment_summary is not None:
            self.segment_summary["updated_at"] = datetime.now()
            self.segment_summary.to_sql(
                "segment_profiles", engine, if_exists="replace", index=False
            )
            log.info(f"Wrote {len(self.segment_summary)} rows -> segment_profiles table")
