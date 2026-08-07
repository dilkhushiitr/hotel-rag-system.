"""
Data Preprocessing
──────────────────
Handles everything between raw features and model-ready matrices:
  1. Missing value imputation
  2. Outlier handling (IQR-based capping)
  3. Log transformation for skewed columns
  4. StandardScaler normalization
  5. Save / load scaler for inference parity
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Columns that receive log1p transform before scaling (typically right-skewed)
LOG_TRANSFORM_COLS = [
    "total_earnings",
    "total_redeemed",
    "wallet_balance",
    "total_impressions",
    "total_clicks",
    "session_count",
    "total_session_duration",
    "total_transactions",
    "total_txn_amount",
]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaNs:
    - Numeric columns → median
    - Categorical columns → mode
    """
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Imputed %s with median=%.4f", col, median_val)

    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug("Imputed %s with mode=%s", col, mode_val)

    return df


def cap_outliers(df: pd.DataFrame, cols: Optional[List[str]] = None, factor: float = 3.0) -> pd.DataFrame:
    """
    Winsorize numeric columns using IQR × factor.
    Outliers are CAPPED (not removed) so DBSCAN can still detect them as noise.
    """
    df = df.copy()
    num_cols = cols or df.select_dtypes(include="number").columns.tolist()

    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        if n_capped:
            logger.debug("Capped %d outliers in '%s'  [%.2f, %.2f]", n_capped, col, lower, upper)

    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply np.log1p to known right-skewed feature columns."""
    df = df.copy()
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            logger.debug("Log-transformed '%s'", col)
    return df


def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    fit: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardise features (zero mean, unit variance).

    Parameters
    ----------
    df          : DataFrame containing feature_cols
    feature_cols: columns to scale
    fit         : if True, fit a new scaler and save it;
                  if False, load the saved scaler (for inference)

    Returns
    -------
    X_scaled    : np.ndarray shape (n_users, n_features)
    scaler      : fitted StandardScaler instance
    """
    X = df[feature_cols].values

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Scaler fitted and saved to %s", SCALER_PATH)
    else:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run pipeline with fit=True first.")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
        logger.info("Scaler loaded from %s", SCALER_PATH)

    logger.info("Scaled feature matrix: %s", X_scaled.shape)
    return X_scaled, scaler


def full_preprocessing_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    fit_scaler: bool = True,
    cap_outliers_flag: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    End-to-end preprocessing:
      impute → cap outliers → log transform → scale

    Returns
    -------
    df_clean  : cleaned DataFrame (with user_id intact)
    X_scaled  : numpy array ready for sklearn models
    scaler    : fitted or loaded scaler
    """
    logger.info("Starting preprocessing pipeline …")
    df_clean = impute_missing(df)

    if cap_outliers_flag:
        df_clean = cap_outliers(df_clean, cols=feature_cols)

    df_clean = log_transform(df_clean)
    X_scaled, scaler = scale_features(df_clean, feature_cols, fit=fit_scaler)

    logger.info("Preprocessing complete. Output shape: %s", X_scaled.shape)
    return df_clean, X_scaled, scaler
