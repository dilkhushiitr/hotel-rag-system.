"""Feature engineering for hybrid fraud detection.

The model works at transaction level. We merge user, device, network,
behavioral, and historical transaction features into one master table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "amount_log",
    "account_age_days",
    "is_new_account",
    "is_android",
    "referral_count",
    "referral_depth",
    "session_duration_min",
    "click_rate",
    "click_entropy",
    "behavior_score",
    "txn_count_1h",
    "txn_count_24h",
    "velocity_ratio",
    "high_velocity_1h",
    "failure_rate_24h",
    "wallet_change_log",
    "amount_deviation_ratio",
    "deviation_flag",
    "device_risk_score",
    "network_risk_score",
    "emulator_flag",
    "rooted_flag",
    "accounts_per_device",
    "vpn_flag",
    "proxy_flag",
    "geo_mismatch_flag",
    "accounts_per_ip",
    "unverified_kyc",
]


def build_historical_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create time-based transaction aggregates per user.

    For production, these features should usually be computed by a streaming
    feature store or analytics warehouse. Here we compute them in pandas so the
    project is runnable end to end.
    """
    df = transactions.sort_values(["user_id", "txn_timestamp"]).copy()
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"], utc=True)
    df = df.set_index("txn_timestamp")

    frames = []
    for user_id, group in df.groupby("user_id", sort=False):
        rolling_1h = group["txn_amount"].rolling("1h").count().rename("txn_count_1h")
        rolling_24h = group["txn_amount"].rolling("24h").count().rename("txn_count_24h")
        avg_7d = group["txn_amount"].rolling("7d").mean().rename("avg_txn_amount_7d")
        failed = (group["txn_status"].str.lower() == "failed").astype(int)
        failure_24h = failed.rolling("24h").mean().rename("failure_rate_24h")
        wallet_change_24h = group["txn_amount"].rolling("24h").sum().rename("wallet_balance_change_24h")

        user_features = pd.concat([rolling_1h, rolling_24h, avg_7d, failure_24h, wallet_change_24h], axis=1)
        user_features["user_id"] = user_id
        user_features["txn_id"] = group["txn_id"].values
        frames.append(user_features.reset_index(drop=True))

    if not frames:
        return pd.DataFrame(
            columns=[
                "txn_id",
                "user_id",
                "txn_count_1h",
                "txn_count_24h",
                "avg_txn_amount_7d",
                "failure_rate_24h",
                "wallet_balance_change_24h",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def merge_tables(
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    devices: pd.DataFrame,
    network: pd.DataFrame,
    behavioral: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cleaned tables into one transaction-level master table."""
    historical = build_historical_aggregates(transactions)
    master = transactions.merge(users, on="user_id", how="left")
    master = master.merge(devices, on="user_id", how="left")
    master = master.merge(network, on="user_id", how="left")
    master = master.merge(behavioral, on="user_id", how="left")
    master = master.merge(historical, on=["txn_id", "user_id"], how="left")
    return master


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convert the master table into a model-ready feature table.

    The function intentionally preserves IDs and labels alongside numeric
    features so downstream scripts can evaluate and explain predictions.
    """
    thresholds = config["thresholds"]
    features = df.copy()

    numeric_defaults = {
        "txn_amount": 0,
        "account_age_days": 1,
        "referral_count": 0,
        "referral_depth": 0,
        "session_duration": 1,
        "click_count": 0,
        "time_between_actions": 0.01,
        "screen_flow_length": 1,
        "click_entropy": 0,
        "behavior_score": 0,
        "txn_count_1h": 0,
        "txn_count_24h": 0,
        "avg_txn_amount_7d": 0,
        "failure_rate_24h": 0,
        "wallet_balance_change_24h": 0,
        "emulator_flag": 0,
        "rooted_flag": 0,
        "accounts_per_device": 1,
        "vpn_flag": 0,
        "proxy_flag": 0,
        "geo_mismatch_flag": 0,
        "accounts_per_ip": 1,
    }
    for column, default in numeric_defaults.items():
        if column not in features.columns:
            features[column] = default
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(default)

    features["kyc_status"] = features.get("kyc_status", "unverified")
    features["os_type"] = features.get("os_type", "android")

    features["amount_log"] = np.log1p(features["txn_amount"])
    features["is_new_account"] = (features["account_age_days"] < thresholds["new_account_days"]).astype(int)
    features["is_android"] = (features["os_type"].astype(str).str.lower() == "android").astype(int)
    features["unverified_kyc"] = (features["kyc_status"].astype(str).str.lower() != "verified").astype(int)

    features["session_duration_min"] = features["session_duration"] / 60.0
    features["click_rate"] = features["click_count"] / features["session_duration"].clip(lower=1)

    features["velocity_ratio"] = features["txn_count_1h"] / (features["txn_count_24h"] + 1)
    features["high_velocity_1h"] = (features["txn_count_1h"] > thresholds["high_velocity_1h"]).astype(int)

    features["amount_deviation_ratio"] = features["txn_amount"] / (features["avg_txn_amount_7d"] + 1)
    features["deviation_flag"] = (
        features["amount_deviation_ratio"] > thresholds["amount_deviation_multiplier"]
    ).astype(int)
    features["wallet_change_log"] = np.sign(features["wallet_balance_change_24h"]) * np.log1p(
        np.abs(features["wallet_balance_change_24h"])
    )

    features["device_risk_score"] = (
        features["emulator_flag"]
        + features["rooted_flag"]
        + (features["accounts_per_device"] > thresholds["accounts_per_device"]).astype(int)
    )
    features["network_risk_score"] = (
        features["vpn_flag"]
        + features["proxy_flag"]
        + features["geo_mismatch_flag"]
        + (features["accounts_per_ip"] > thresholds["accounts_per_ip"]).astype(int)
    )

    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)

    return features


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric feature matrix used by Isolation Forest."""
    return df[FEATURE_COLUMNS].copy()

