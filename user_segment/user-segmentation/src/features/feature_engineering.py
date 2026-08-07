"""
src/features/feature_engineering.py

Full feature engineering pipeline:
  Raw extracted DataFrame → model-ready feature matrix

Steps:
  1. Impute missing values
  2. Cap outliers (IQR)
  3. Log-transform skewed features
  4. Compute composite scores
  5. Standardize (StandardScaler)
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ─── Feature groups ───────────────────────────────────────────────────────────
ENGAGEMENT_FEATURES = [
    "sessions_per_user",
    "avg_session_duration",
    "active_days",
    "total_page_views",
]

MONETIZATION_FEATURES = [
    "total_earnings",
    "total_redemptions",
    "wallet_balance",
    "ad_revenue",
    "ctr",
]

AD_BEHAVIOR_FEATURES = [
    "total_impressions",
    "total_clicks",
    "ctr",
    "ad_type_diversity",
]

TRANSACTION_FEATURES = [
    "total_transactions",
    "avg_transaction_value",
    "total_transaction_value",
    "success_rate",
]

LOG_TRANSFORM_COLS = [
    "sessions_per_user",
    "avg_session_duration",
    "total_page_views",
    "total_earnings",
    "total_redemptions",
    "wallet_balance",
    "ad_revenue",
    "total_impressions",
    "total_clicks",
    "total_transaction_value",
    "avg_transaction_value",
]

# Final model features (after engineering)
MODEL_FEATURES = [
    "sessions_per_user",
    "avg_session_duration",
    "active_days",
    "recency_days",
    "ctr",
    "total_impressions",
    "ad_revenue",
    "wallet_balance",
    "total_earnings",
    "total_redemptions",
    "avg_transaction_value",
    "success_rate",
    "engagement_score",
    "monetization_score",
    "churn_risk_score",
]


# ─── Step 1: Imputation ───────────────────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with median for numeric, 0 for binary/rate columns."""
    df = df.copy()

    # Zero-fill for counts / rates where null means "no activity"
    zero_fill_cols = [
        "sessions_per_user", "active_days", "total_page_views",
        "total_impressions", "total_clicks", "ctr", "ad_revenue",
        "total_transactions", "success_rate", "ad_type_diversity",
    ]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Median fill for value-based columns
    median_fill_cols = [
        "avg_session_duration", "wallet_balance", "total_earnings",
        "total_redemptions", "avg_transaction_value", "total_transaction_value",
    ]
    for col in median_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    log.info(f"Imputation complete. Remaining NaNs: {df.isnull().sum().sum()}")
    return df


# ─── Step 2: Outlier capping ──────────────────────────────────────────────────
def cap_outliers(df: pd.DataFrame, iqr_factor: float = 1.5) -> pd.DataFrame:
    """IQR-based capping. Keeps outliers in data but bounds their influence."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_capped > 0:
            log.debug(f"  {col}: capped {n_capped} outliers [{lower:.2f}, {upper:.2f}]")

    log.info("Outlier capping complete")
    return df


# ─── Step 3: Log transform ────────────────────────────────────────────────────
def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """log1p transform to reduce skewness in heavy-tailed distributions."""
    df = df.copy()
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    log.info(f"Log-transformed {len(LOG_TRANSFORM_COLS)} columns")
    return df


# ─── Step 4: Recency feature ──────────────────────────────────────────────────
def add_recency(df: pd.DataFrame) -> pd.DataFrame:
    """Convert last_active timestamp → recency_days (lower = more recent)."""
    df = df.copy()
    now = datetime.now()
    if "last_active" in df.columns:
        df["last_active"] = pd.to_datetime(df["last_active"], errors="coerce")
        df["recency_days"] = (now - df["last_active"]).dt.days.fillna(30)
    else:
        df["recency_days"] = 30  # default: assume inactive if no session data
    return df


# ─── Step 5: Composite scores ─────────────────────────────────────────────────
def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engagement, monetization, and churn_risk composite scores.
    Each score is 0–1 normalized within the batch.
    """
    df = df.copy()

    def minmax(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        if rng == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.min()) / rng

    # ── Engagement Score (0–10)
    eng = (
        0.30 * minmax(df.get("sessions_per_user",     pd.Series(np.zeros(len(df)))))
      + 0.30 * minmax(df.get("avg_session_duration",  pd.Series(np.zeros(len(df)))))
      + 0.25 * minmax(df.get("active_days",           pd.Series(np.zeros(len(df)))))
      + 0.15 * (1 - minmax(df.get("recency_days",     pd.Series(np.zeros(len(df))))))
    )
    df["engagement_score"] = (eng * 10).round(2)

    # ── Monetization Score (0–10)
    total_earn  = df.get("total_earnings",    pd.Series(np.zeros(len(df))))
    total_redm  = df.get("total_redemptions", pd.Series(np.zeros(len(df))))
    ctr_col     = df.get("ctr",               pd.Series(np.zeros(len(df))))

    mon = (
        0.40 * minmax(total_earn)
      + 0.30 * minmax(total_redm)
      + 0.30 * minmax(ctr_col)
    )
    df["monetization_score"] = (mon * 10).round(2)

    # ── Churn Risk Score (0–1, higher = more risk)
    recency   = df.get("recency_days",      pd.Series(np.full(len(df), 30)))
    act_days  = df.get("active_days",       pd.Series(np.zeros(len(df))))
    avg_dur   = df.get("avg_session_duration", pd.Series(np.zeros(len(df))))

    churn = (
        0.50 * minmax(recency)
      + 0.30 * (1 - minmax(act_days))
      + 0.20 * (1 - minmax(avg_dur))
    )
    df["churn_risk_score"] = churn.round(4)

    log.info("Composite scores computed: engagement, monetization, churn_risk")
    return df


# ─── Step 6: Rename & consolidate ────────────────────────────────────────────
def consolidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw wallet columns to match feature names."""
    df = df.copy()
    rename_map = {
        "total_earned":    "total_earnings",
        "total_redeemed":  "total_redemptions",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


# ─── Step 7: Scale ────────────────────────────────────────────────────────────
def scale_features(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScaler on MODEL_FEATURES.
    If fit=True: fit and transform (training).
    If fit=False: transform only (inference).
    Returns: scaled DataFrame, fitted scaler
    """
    available = [f for f in MODEL_FEATURES if f in df.columns]
    missing   = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        log.warning(f"Missing model features (filling 0): {missing}")
        for col in missing:
            df[col] = 0.0

    if scaler is None:
        scaler = StandardScaler()

    X = df[available].copy()
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=available, index=df.index)
    log.info(f"Scaled {len(available)} features. Shape: {df_scaled.shape}")
    return df_scaled, scaler


# ─── Master pipeline function ─────────────────────────────────────────────────
def build_feature_matrix(
    raw_df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit_scaler: bool = True,
    save_processed: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Full pipeline: raw → model-ready.

    Returns:
        features_df   : scaled feature matrix (for clustering)
        full_df       : unscaled enriched df (for profiling / storage)
        scaler        : fitted StandardScaler
    """
    log.info("Starting feature engineering pipeline...")

    df = raw_df.copy()
    df = consolidate_columns(df)
    df = impute_missing(df)
    df = add_recency(df)
    df = cap_outliers(df)
    df = log_transform(df)
    df = add_composite_scores(df)

    features_df, scaler = scale_features(df, scaler=scaler, fit=fit_scaler)

    # Keep user_id in features_df for downstream joins
    features_df["user_id"] = df["user_id"].values

    if save_processed:
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/features_unscaled.csv", index=False)
        features_df.to_csv("data/processed/features_scaled.csv", index=False)
        log.info("Saved processed features to data/processed/")

    log.info(f"Feature matrix ready: {features_df.shape}")
    return features_df, df, scaler
