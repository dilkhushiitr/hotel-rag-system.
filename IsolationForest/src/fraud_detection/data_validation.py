"""Data validation layer for raw CSV inputs.

Validation catches broken upstream data before it silently damages training or
production scoring. In real deployments this layer can be replaced or extended
with Great Expectations, Pandera, or database constraints.
"""

from __future__ import annotations

import pandas as pd


def _require_columns(df: pd.DataFrame, required_columns: set[str], table_name: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing)}")


def validate_transactions(df: pd.DataFrame) -> bool:
    """Validate the transaction table.

    This is the most important table because each output fraud score is at
    transaction level. Labels are optional for scoring but useful for training
    evaluation, so `is_fraud` is not enforced here.
    """
    required = {"txn_id", "user_id", "txn_amount", "txn_type", "txn_timestamp", "txn_status"}
    _require_columns(df, required, "transactions")

    if df["txn_amount"].isnull().any():
        raise ValueError("transactions contains missing txn_amount values.")
    if (df["txn_amount"] <= 0).any():
        raise ValueError("transactions contains txn_amount <= 0.")
    if not df["txn_id"].is_unique:
        raise ValueError("transactions contains duplicate txn_id values.")
    if df["txn_timestamp"].isnull().any():
        raise ValueError("transactions contains missing txn_timestamp values.")

    return True


def validate_users(df: pd.DataFrame) -> bool:
    """Validate user-level profile data."""
    required = {"user_id", "account_age_days", "kyc_status", "signup_country"}
    _require_columns(df, required, "users")
    if df["user_id"].isnull().any():
        raise ValueError("users contains missing user_id values.")
    return True


def validate_devices(df: pd.DataFrame) -> bool:
    """Validate device-level fraud signals."""
    required = {"device_id", "user_id", "emulator_flag", "rooted_flag", "accounts_per_device"}
    _require_columns(df, required, "devices")
    return True


def validate_network(df: pd.DataFrame) -> bool:
    """Validate IP/network-level fraud signals."""
    required = {"user_id", "ip_address", "vpn_flag", "proxy_flag", "geo_mismatch_flag", "accounts_per_ip"}
    _require_columns(df, required, "network")
    return True


def validate_behavioral(df: pd.DataFrame) -> bool:
    """Validate session and behavior-level fraud signals."""
    required = {
        "session_id",
        "user_id",
        "session_duration",
        "click_count",
        "time_between_actions",
        "screen_flow_length",
        "click_entropy",
        "behavior_score",
    }
    _require_columns(df, required, "behavioral")
    return True


def validate_all_tables(
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    devices: pd.DataFrame,
    network: pd.DataFrame,
    behavioral: pd.DataFrame,
) -> bool:
    """Validate all raw input tables together."""
    validate_users(users)
    validate_transactions(transactions)
    validate_devices(devices)
    validate_network(network)
    validate_behavioral(behavioral)
    return True

