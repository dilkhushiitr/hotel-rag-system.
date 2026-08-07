"""Data cleaning functions for each raw table.

Each table is cleaned separately before merging. This keeps data quality rules
easy to test and avoids hiding table-specific issues inside one large function.
"""

from __future__ import annotations

import pandas as pd


FLAG_COLUMNS = [
    "emulator_flag",
    "rooted_flag",
    "vpn_flag",
    "proxy_flag",
    "geo_mismatch_flag",
]


def _clean_binary_flag(series: pd.Series) -> pd.Series:
    """Convert noisy flag values into strict 0/1 integers."""
    return (
        series.fillna(0)
        .astype(str)
        .str.lower()
        .map({"1": 1, "true": 1, "yes": 1, "y": 1, "0": 0, "false": 0, "no": 0, "n": 0})
        .fillna(0)
        .astype(int)
    )


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transaction data.

    Steps:
    1. Remove invalid transaction amounts.
    2. Drop rows without timestamps.
    3. Standardize timestamps to UTC.
    4. Remove duplicate transaction IDs.
    """
    cleaned = df.copy()
    cleaned = cleaned[cleaned["txn_amount"] > 0]
    cleaned = cleaned.dropna(subset=["txn_timestamp"])
    cleaned["txn_timestamp"] = pd.to_datetime(cleaned["txn_timestamp"], utc=True, errors="coerce")
    cleaned = cleaned.dropna(subset=["txn_timestamp"])
    cleaned = cleaned.drop_duplicates(subset=["txn_id"], keep="last")
    cleaned["txn_status"] = cleaned["txn_status"].fillna("failed").str.lower()
    cleaned["txn_type"] = cleaned["txn_type"].fillna("earn").str.lower()
    return cleaned


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """Clean user profile data."""
    cleaned = df.copy()
    cleaned["account_age_days"] = pd.to_numeric(cleaned["account_age_days"], errors="coerce").fillna(1)
    cleaned["account_age_days"] = cleaned["account_age_days"].clip(lower=1).astype(int)
    cleaned["kyc_status"] = cleaned["kyc_status"].fillna("unverified").str.lower()
    cleaned["referral_count"] = pd.to_numeric(cleaned.get("referral_count", 0), errors="coerce").fillna(0)
    cleaned["referral_depth"] = pd.to_numeric(cleaned.get("referral_depth", 0), errors="coerce").fillna(0)
    cleaned = cleaned.drop_duplicates(subset=["user_id"], keep="last")
    return cleaned


def clean_devices(df: pd.DataFrame) -> pd.DataFrame:
    """Clean device data and keep one latest device row per user."""
    cleaned = df.copy()
    for column in ("emulator_flag", "rooted_flag"):
        cleaned[column] = _clean_binary_flag(cleaned[column])
    cleaned["accounts_per_device"] = (
        pd.to_numeric(cleaned["accounts_per_device"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    )
    cleaned = cleaned.drop_duplicates(subset=["user_id"], keep="last")
    return cleaned


def clean_network(df: pd.DataFrame) -> pd.DataFrame:
    """Clean network/IP data and keep one row per user."""
    cleaned = df.copy()
    for column in ("vpn_flag", "proxy_flag", "geo_mismatch_flag"):
        cleaned[column] = _clean_binary_flag(cleaned[column])
    cleaned["accounts_per_ip"] = (
        pd.to_numeric(cleaned["accounts_per_ip"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    )
    cleaned = cleaned.drop_duplicates(subset=["user_id"], keep="last")
    return cleaned


def clean_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """Clean behavioral/session data and keep one row per user."""
    cleaned = df.copy()
    cleaned["session_duration"] = pd.to_numeric(cleaned["session_duration"], errors="coerce").fillna(1).clip(lower=1)
    cleaned["click_count"] = pd.to_numeric(cleaned["click_count"], errors="coerce").fillna(0).clip(lower=0)
    cleaned["time_between_actions"] = (
        pd.to_numeric(cleaned["time_between_actions"], errors="coerce").fillna(0.01).clip(lower=0.01)
    )
    cleaned["screen_flow_length"] = (
        pd.to_numeric(cleaned["screen_flow_length"], errors="coerce").fillna(1).clip(lower=1)
    )
    cleaned["click_entropy"] = pd.to_numeric(cleaned["click_entropy"], errors="coerce").fillna(0)
    cleaned["behavior_score"] = pd.to_numeric(cleaned["behavior_score"], errors="coerce").fillna(0).clip(0, 100)
    cleaned = cleaned.drop_duplicates(subset=["user_id"], keep="last")
    return cleaned

