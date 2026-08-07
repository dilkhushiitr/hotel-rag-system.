"""CSV loading and data-cleaning layer.

This module is intentionally separate from feature engineering. Cleaning answers
"is this raw record valid?", while feature engineering answers "what user-level
signal should the model learn from?" Keeping them separate makes debugging much
easier in production.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from user_segmentation.config import ProjectConfig
from user_segmentation.schemas import SCHEMAS, validate_columns


def _read_optional_csv(path: Path, table_name: str) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise return an empty table with schema columns.

    Location and device files are useful for profiling, but the core clustering
    pipeline should still run if those optional dimensions are unavailable.
    """

    if path.exists():
        data = pd.read_csv(path)
        validate_columns(table_name, data)
        return data

    return pd.DataFrame(columns=SCHEMAS[table_name].required_columns)


def load_raw_tables(config: ProjectConfig, data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all seven CSV blocks into a dictionary of DataFrames."""

    root = Path(data_dir).expanduser().resolve() if data_dir else config.raw_data_dir
    filenames = config.raw["data"]

    tables: dict[str, pd.DataFrame] = {}
    for table_name, filename in filenames.items():
        tables[table_name] = _read_optional_csv(root / filename, table_name)

    return tables


def clean_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean raw CSV tables using business rules from the problem statement."""

    cleaned = {name: df.copy() for name, df in tables.items()}

    # Users: signup date defines user age, so invalid dates are treated as
    # missing. They will later be imputed using safe defaults.
    cleaned["users"]["signup_date"] = pd.to_datetime(
        cleaned["users"]["signup_date"], errors="coerce", utc=True
    )

    # Sessions: remove missing timestamps and impossible/non-positive durations.
    # Clustering is distance-based, so even a small number of impossible records
    # can pull centroids in the wrong direction.
    sessions = cleaned["sessions"]
    sessions["timestamp"] = pd.to_datetime(sessions["timestamp"], errors="coerce", utc=True)
    sessions["duration"] = pd.to_numeric(sessions["duration"], errors="coerce")
    cleaned["sessions"] = sessions.dropna(subset=["timestamp", "duration"]).query("duration > 0")

    # Ad events: missing impressions/clicks are treated as 0, because no event
    # means no observed exposure or click. Negative values are invalid and clipped.
    ad_events = cleaned["ad_events"]
    for column in ["impressions", "clicks"]:
        ad_events[column] = pd.to_numeric(ad_events[column], errors="coerce").fillna(0).clip(lower=0)
    cleaned["ad_events"] = ad_events

    # Wallet: business requirement says missing earnings and redeemed should be 0.
    wallet = cleaned["wallet"]
    for column in ["earnings", "redeemed"]:
        wallet[column] = pd.to_numeric(wallet[column], errors="coerce").fillna(0).clip(lower=0)
    cleaned["wallet"] = wallet

    # Transactions: timestamp is optional in the schema but supported when
    # present. Successful transaction rate is based on normalized status values.
    transactions = cleaned["transactions"]
    transactions["txn_amount"] = pd.to_numeric(transactions["txn_amount"], errors="coerce").fillna(0).clip(lower=0)
    transactions["txn_status"] = transactions["txn_status"].fillna("unknown").astype(str).str.lower().str.strip()
    if "timestamp" in transactions.columns:
        transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], errors="coerce", utc=True)
    cleaned["transactions"] = transactions

    return cleaned

