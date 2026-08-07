"""
Feature Engineering
────────────────────
Transforms 5 raw tables into a single user-level feature matrix.

Feature Groups
--------------
1. Engagement    — sessions, duration, active_days, recency
2. Ad Behaviour  — impressions, clicks, CTR, preferred category
3. Monetization  — earnings, redemptions, wallet balance
4. Transactions  — count, avg amount, success rate
5. Derived Scores — engagement_score, monetization_score, churn_risk_score

All aggregations use a configurable LOOKBACK_DAYS rolling window.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import LOOKBACK_DAYS

logger = logging.getLogger(__name__)

# ── Feature column list (used in preprocessing & model training) ───────────────
FEATURE_COLS = [
    # Engagement
    "session_count",
    "avg_session_duration",
    "total_session_duration",
    "active_days",
    "recency_days",
    "pages_per_session",
    # Ad Behaviour
    "total_impressions",
    "total_clicks",
    "ctr",
    # Monetization
    "total_earnings",
    "total_redeemed",
    "wallet_balance",
    "redemption_rate",
    # Transactions
    "total_transactions",
    "total_txn_amount",
    "avg_txn_amount",
    "txn_success_rate",
    # Derived composite scores
    "engagement_score",
    "monetization_score",
    "churn_risk_score",
]


def _filter_window(df: pd.DataFrame, date_col: str, end_date: datetime) -> pd.DataFrame:
    """Keep only rows within the LOOKBACK_DAYS window ending at end_date."""
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]


# ── Individual feature builders ────────────────────────────────────────────────

def build_engagement_features(sessions: pd.DataFrame, end_date: datetime) -> pd.DataFrame:
    """
    Per-user session-level aggregations within the lookback window.
    """
    sess = _filter_window(sessions, "session_start", end_date).copy()

    # Recency: days since last session (lower = more recent = less likely to churn)
    last_session = sess.groupby("user_id")["session_start"].max().reset_index()
    last_session["recency_days"] = (end_date - last_session["session_start"]).dt.days
    last_session = last_session[["user_id", "recency_days"]]

    agg = sess.groupby("user_id").agg(
        session_count          = ("session_id", "count"),
        avg_session_duration   = ("duration_seconds", "mean"),
        total_session_duration = ("duration_seconds", "sum"),
        active_days            = ("session_start", lambda x: x.dt.date.nunique()),
        pages_per_session      = ("pages_viewed", "mean"),
    ).reset_index()

    feat = agg.merge(last_session, on="user_id", how="left")
    feat["recency_days"] = feat["recency_days"].fillna(LOOKBACK_DAYS)
    return feat


def build_ad_features(ad_events: pd.DataFrame, end_date: datetime) -> pd.DataFrame:
    """
    Per-user ad impression / click aggregations.
    """
    ads = _filter_window(ad_events, "timestamp", end_date).copy()

    impressions = (
        ads[ads["event_type"] == "impression"]
        .groupby("user_id")
        .agg(total_impressions=("event_id", "count"))
        .reset_index()
    )

    clicks = (
        ads[ads["event_type"] == "click"]
        .groupby("user_id")
        .agg(total_clicks=("event_id", "count"))
        .reset_index()
    )

    # Preferred ad category (mode)
    pref_cat = (
        ads[ads["event_type"] == "click"]
        .groupby("user_id")["ad_category"]
        .agg(lambda x: x.mode()[0] if len(x) else "none")
        .reset_index()
        .rename(columns={"ad_category": "preferred_ad_category"})
    )

    feat = impressions.merge(clicks, on="user_id", how="outer").fillna(0)
    feat = feat.merge(pref_cat, on="user_id", how="left")
    feat["preferred_ad_category"] = feat["preferred_ad_category"].fillna("none")
    feat["ctr"] = (feat["total_clicks"] / feat["total_impressions"].replace(0, np.nan)).fillna(0)
    return feat


def build_monetization_features(wallet: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user wallet / earnings features (snapshot, no window needed).
    """
    feat = wallet[["user_id", "total_earnings", "total_redeemed", "wallet_balance"]].copy()
    feat["redemption_rate"] = (
        feat["total_redeemed"] / feat["total_earnings"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    return feat


def build_transaction_features(transactions: pd.DataFrame, end_date: datetime) -> pd.DataFrame:
    """
    Per-user transaction aggregations within the lookback window.
    """
    txns = _filter_window(transactions, "created_at", end_date).copy()

    agg = txns.groupby("user_id").agg(
        total_transactions = ("txn_id", "count"),
        total_txn_amount   = ("amount", "sum"),
        avg_txn_amount     = ("amount", "mean"),
    ).reset_index()

    _sr = (
        txns.groupby("user_id")
        .apply(lambda x: (x["status"] == "success").sum() / len(x))
        .reset_index()
    )
    _computed = [c for c in _sr.columns if c != "user_id"][0]
    success_rate = _sr.rename(columns={_computed: "txn_success_rate"})

    feat = agg.merge(success_rate, on="user_id", how="left")
    feat["txn_success_rate"] = feat["txn_success_rate"].fillna(0)
    return feat


# ── Derived / Composite Scores ─────────────────────────────────────────────────

def _minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]."""
    rng = series.max() - series.min()
    return (series - series.min()) / rng if rng > 0 else series * 0


def build_derived_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compose three interpretable business scores from raw features.

    engagement_score   = weighted combo of sessions, active days, recency
    monetization_score = weighted combo of earnings, redemptions, CTR
    churn_risk_score   = high recency + low sessions + low earnings → high risk
    """
    df = df.copy()

    # Engagement score (0–100)
    eng = (
        0.35 * _minmax_norm(df["session_count"])
        + 0.35 * _minmax_norm(df["active_days"])
        + 0.30 * (1 - _minmax_norm(df["recency_days"]))  # invert: low recency = good
    )
    df["engagement_score"] = (eng * 100).round(2)

    # Monetization score (0–100)
    mon = (
        0.40 * _minmax_norm(df["total_earnings"])
        + 0.30 * _minmax_norm(df["total_redeemed"])
        + 0.30 * _minmax_norm(df["ctr"])
    )
    df["monetization_score"] = (mon * 100).round(2)

    # Churn risk score (0–100); higher = more likely to churn
    churn = (
        0.50 * _minmax_norm(df["recency_days"])
        + 0.30 * (1 - _minmax_norm(df["session_count"]))
        + 0.20 * (1 - _minmax_norm(df["total_earnings"]))
    )
    df["churn_risk_score"] = (churn * 100).round(2)

    return df


# ── Master function ────────────────────────────────────────────────────────────

def build_feature_matrix(
    tables: dict,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Orchestrates all feature builders and joins into a single user-level DataFrame.

    Parameters
    ----------
    tables   : dict returned by ingestion.load_all()
    end_date : snapshot date; defaults to today

    Returns
    -------
    DataFrame with columns = [user_id] + FEATURE_COLS
    """
    if end_date is None:
        end_date = datetime.utcnow()

    logger.info("Building feature matrix for window: %d days ending %s", LOOKBACK_DAYS, end_date.date())

    users = tables["users"][["user_id"]].drop_duplicates()

    eng_feat  = build_engagement_features(tables["sessions"],     end_date)
    ad_feat   = build_ad_features(tables["ad_events"],            end_date)
    mon_feat  = build_monetization_features(tables["wallet"])
    txn_feat  = build_transaction_features(tables["transactions"], end_date)

    # Left join everything onto users so every user has a row
    df = (
        users
        .merge(eng_feat,  on="user_id", how="left")
        .merge(ad_feat,   on="user_id", how="left")
        .merge(mon_feat,  on="user_id", how="left")
        .merge(txn_feat,  on="user_id", how="left")
    )

    # Fill users who had zero activity
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df["preferred_ad_category"] = df.get("preferred_ad_category", pd.Series("none")).fillna("none")

    # Derived composite scores
    df = build_derived_scores(df)

    # Enforce correct column order
    final_cols = ["user_id"] + [c for c in FEATURE_COLS if c in df.columns]
    df = df[final_cols]

    logger.info("Feature matrix ready: %d users × %d features", len(df), len(FEATURE_COLS))
    return df
