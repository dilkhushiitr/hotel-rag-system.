"""Feature engineering and preprocessing for user segmentation.

The output of this module is one row per user. K-Means and DBSCAN require a
numeric matrix, so we aggregate raw event-level data into behavior, ad,
monetization, transaction, and derived score features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from user_segmentation.config import ProjectConfig


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series and return 0 when the denominator is 0."""

    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)


def _reference_date(tables: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Choose a stable reference date for recency and user age calculations."""

    candidates: list[pd.Timestamp] = []
    if not tables["sessions"].empty:
        candidates.append(tables["sessions"]["timestamp"].max())
    if "timestamp" in tables["transactions"].columns and not tables["transactions"].empty:
        candidates.append(tables["transactions"]["timestamp"].max())
    if not tables["users"].empty:
        candidates.append(tables["users"]["signup_date"].max())

    valid_candidates = [value for value in candidates if pd.notna(value)]
    if valid_candidates:
        return max(valid_candidates)

    # Fallback is only used for extremely small inference payloads.
    return pd.Timestamp.utcnow()


def build_user_features(tables: dict[str, pd.DataFrame], config: ProjectConfig) -> pd.DataFrame:
    """Create a production-ready user-level feature table.

    The function deliberately uses left joins from the full user universe so that
    inactive users are retained instead of disappearing from the segmentation
    output. Missing behavior later becomes 0 or a high recency value.
    """

    window_days = int(config.raw["features"]["activity_window_days"])
    ref_date = _reference_date(tables)
    window_start = ref_date - pd.Timedelta(days=window_days)

    users = tables["users"].copy()
    if users.empty:
        # In batch scoring, a user file is preferred. For defensive inference, we
        # can still build a user universe from any table containing user_id.
        user_ids = pd.concat(
            [df[["user_id"]] for df in tables.values() if "user_id" in df.columns],
            ignore_index=True,
        ).drop_duplicates()
        users = user_ids.assign(signup_date=pd.NaT, country="unknown")

    users = users.drop_duplicates("user_id")
    base = users[["user_id", "signup_date", "country"]].copy()
    base["user_age_days"] = (ref_date - base["signup_date"]).dt.days.clip(lower=0)

    # Engagement features from sessions table.
    sessions = tables["sessions"].copy()
    sessions_30d = sessions[sessions["timestamp"].between(window_start, ref_date)] if not sessions.empty else sessions
    if "pages" in sessions_30d.columns:
        sessions_30d["pages"] = pd.to_numeric(sessions_30d["pages"], errors="coerce").fillna(0).clip(lower=0)
    else:
        # If content-depth data is unavailable, use 0. The column still exists so
        # downstream code has a stable schema.
        sessions_30d["pages"] = 0

    if sessions_30d.empty:
        session_features = pd.DataFrame(columns=["user_id"])
    else:
        session_features = sessions_30d.assign(activity_date=sessions_30d["timestamp"].dt.date).groupby("user_id").agg(
            session_count=("session_id", "nunique"),
            avg_session_duration=("duration", "mean"),
            total_session_duration=("duration", "sum"),
            active_days=("activity_date", "nunique"),
            last_session_at=("timestamp", "max"),
            pages_total=("pages", "sum"),
        ).reset_index()
        session_features["recency_days"] = (ref_date - session_features["last_session_at"]).dt.days.clip(lower=0)
        session_features["pages_per_session"] = _safe_divide(
            session_features["pages_total"], session_features["session_count"]
        )
        session_features = session_features.drop(columns=["last_session_at", "pages_total"])

    # Ad behavior features.
    ad_events = tables["ad_events"]
    if ad_events.empty:
        ad_features = pd.DataFrame(columns=["user_id"])
    else:
        ad_features = ad_events.groupby("user_id").agg(
            total_impressions=("impressions", "sum"),
            total_clicks=("clicks", "sum"),
        ).reset_index()
        ad_features["ctr"] = _safe_divide(ad_features["total_clicks"], ad_features["total_impressions"])

    # Monetization features from wallet.
    wallet = tables["wallet"]
    if wallet.empty:
        wallet_features = pd.DataFrame(columns=["user_id"])
    else:
        wallet_features = wallet.groupby("user_id").agg(
            total_earnings=("earnings", "sum"),
            total_redeemed=("redeemed", "sum"),
        ).reset_index()
        wallet_features["wallet_balance"] = (wallet_features["total_earnings"] - wallet_features["total_redeemed"]).clip(lower=0)
        wallet_features["redemption_rate"] = _safe_divide(wallet_features["total_redeemed"], wallet_features["total_earnings"])

    # Transaction features.
    transactions = tables["transactions"]
    if transactions.empty:
        txn_features = pd.DataFrame(columns=["user_id"])
    else:
        transactions = transactions.assign(is_success=(transactions["txn_status"] == "success").astype(int))
        txn_features = transactions.groupby("user_id").agg(
            total_transactions=("txn_amount", "size"),
            total_txn_amount=("txn_amount", "sum"),
            avg_txn_amount=("txn_amount", "mean"),
            txn_success_rate=("is_success", "mean"),
        ).reset_index()

    # Merge all feature families into one user-level table.
    features = base.merge(session_features, on="user_id", how="left")
    features = features.merge(ad_features, on="user_id", how="left")
    features = features.merge(wallet_features, on="user_id", how="left")
    features = features.merge(txn_features, on="user_id", how="left")

    numeric_columns = [column for column in features.columns if column not in {"user_id", "signup_date", "country"}]
    features[numeric_columns] = features[numeric_columns].fillna(0)

    # Users with no session in the 30-day window should look risky. Set recency
    # to the full window instead of 0, because 0 means "active today".
    features.loc[features["session_count"] == 0, "recency_days"] = window_days

    # Retention proxy used for segment reporting. It is not in the cluster list
    # by default, but business teams can monitor it by segment.
    features["retention_rate"] = (features["active_days"] / window_days).clip(lower=0, upper=1)

    # Derived scores use min-max normalized raw features so weights are
    # interpretable. These are business-friendly scores, not arbitrary model
    # internals.
    score_inputs = features[["session_count", "active_days", "recency_days", "total_earnings", "total_redeemed", "ctr"]]
    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(score_inputs),
        columns=[f"norm_{column}" for column in score_inputs.columns],
        index=features.index,
    )

    features["engagement_score"] = (
        0.35 * normalized["norm_session_count"]
        + 0.35 * normalized["norm_active_days"]
        + 0.30 * (1 - normalized["norm_recency_days"])
    )
    features["monetization_score"] = (
        0.40 * normalized["norm_total_earnings"]
        + 0.30 * normalized["norm_total_redeemed"]
        + 0.30 * normalized["norm_ctr"]
    )
    features["churn_risk_score"] = (
        0.50 * normalized["norm_recency_days"]
        + 0.30 * (1 - normalized["norm_session_count"])
        + 0.20 * (1 - normalized["norm_total_earnings"])
    )

    return features


@dataclass
class UserSegmentationPreprocessor:
    """Impute, cap outliers, log-transform, and standardize model features.

    The object stores all training-time statistics so the exact same
    transformation is applied during inference. This is critical: training and
    serving must never compute different scaling or capping thresholds.
    """

    feature_columns: list[str]
    outlier_iqr_multiplier: float = 3.0
    fill_values_: dict[str, float] = field(default_factory=dict)
    lower_bounds_: dict[str, float] = field(default_factory=dict)
    upper_bounds_: dict[str, float] = field(default_factory=dict)
    scaler_: StandardScaler | None = None

    def fit(self, features: pd.DataFrame) -> "UserSegmentationPreprocessor":
        """Learn imputation values, IQR caps, and scaler parameters."""

        matrix = features[self.feature_columns].copy()
        self.fill_values_ = matrix.median(numeric_only=True).fillna(0).to_dict()
        matrix = matrix.fillna(self.fill_values_)

        for column in self.feature_columns:
            q1 = matrix[column].quantile(0.25)
            q3 = matrix[column].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[column] = float(q1 - self.outlier_iqr_multiplier * iqr)
            self.upper_bounds_[column] = float(q3 + self.outlier_iqr_multiplier * iqr)

        transformed = self._cap_and_log(matrix)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(transformed)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Transform user features into the scaled numeric matrix for models."""

        if self.scaler_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        matrix = features[self.feature_columns].copy()
        matrix = matrix.fillna(self.fill_values_)
        transformed = self._cap_and_log(matrix)
        return self.scaler_.transform(transformed)

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        """Fit preprocessing on training data and return transformed values."""

        return self.fit(features).transform(features)

    def _cap_and_log(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier capping and log1p transform.

        log1p compresses heavy-tailed variables such as earnings and duration.
        Values are clipped to at least 0 because negative post-IQR lower bounds
        are not meaningful for these count and monetary features.
        """

        capped = matrix.copy()
        for column in self.feature_columns:
            lower = max(0.0, self.lower_bounds_.get(column, 0.0))
            upper = self.upper_bounds_.get(column, capped[column].max())
            capped[column] = capped[column].clip(lower=lower, upper=upper)

        return np.log1p(capped)

