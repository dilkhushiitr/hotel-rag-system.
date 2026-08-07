"""
Data Ingestion Layer
────────────────────
Loads raw data from either:
  - CSV files (development / local mode)
  - PostgreSQL database (production mode)

All public functions return clean DataFrames ready for feature engineering.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, DATABASE_URL

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_engine():
    from sqlalchemy import create_engine
    """Return a SQLAlchemy engine. Raises if DB is unreachable."""
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def _load_csv(filename: str) -> pd.DataFrame:
    path = DATA_RAW / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"Run `python data/sample/generate_sample_data.py` first."
        )
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %s  →  %d rows, %d cols", filename, len(df), df.shape[1])
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def load_users(source: str = "csv") -> pd.DataFrame:
    """
    Load users master table.
    Columns: user_id, created_at, country, device_type, age_bucket, referral_src
    """
    if source == "db":
        engine = _get_engine()
        from sqlalchemy import text
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM users"), conn)
    else:
        df = _load_csv("users.csv")

    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def load_sessions(source: str = "csv") -> pd.DataFrame:
    """
    Load session events.
    Columns: session_id, user_id, session_start, duration_seconds, pages_viewed, platform
    """
    if source == "db":
        engine = _get_engine()
        from sqlalchemy import text
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM sessions"), conn)
    else:
        df = _load_csv("sessions.csv")

    df["session_start"] = pd.to_datetime(df["session_start"])
    return df


def load_ad_events(source: str = "csv") -> pd.DataFrame:
    """
    Load ad impression / click events.
    Columns: event_id, user_id, ad_id, event_type, timestamp, ad_category
    """
    if source == "db":
        engine = _get_engine()
        from sqlalchemy import text
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM ad_events"), conn)
    else:
        df = _load_csv("ad_events.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_wallet(source: str = "csv") -> pd.DataFrame:
    """
    Load wallet / earnings data.
    Columns: user_id, total_earnings, total_redeemed, wallet_balance, last_redemption
    """
    if source == "db":
        engine = _get_engine()
        from sqlalchemy import text
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM wallet"), conn)
    else:
        df = _load_csv("wallet.csv")

    df["last_redemption"] = pd.to_datetime(df["last_redemption"], errors="coerce")
    return df


def load_transactions(source: str = "csv") -> pd.DataFrame:
    """
    Load transaction records.
    Columns: txn_id, user_id, amount, status, txn_type, created_at
    """
    if source == "db":
        engine = _get_engine()
        from sqlalchemy import text
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM transactions"), conn)
    else:
        df = _load_csv("transactions.csv")

    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def load_all(source: str = "csv") -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper — loads ALL tables in one call.

    Returns
    -------
    dict with keys: "users", "sessions", "ad_events", "wallet", "transactions"
    """
    logger.info("Loading all tables from source='%s' …", source)
    tables = {
        "users":        load_users(source),
        "sessions":     load_sessions(source),
        "ad_events":    load_ad_events(source),
        "wallet":       load_wallet(source),
        "transactions": load_transactions(source),
    }
    for name, df in tables.items():
        logger.info("  %-15s  %d rows", name, len(df))
    return tables


def write_segments_to_db(segments_df: pd.DataFrame) -> None:
    """
    Persist the final user_segments table back to PostgreSQL.
    Upserts on user_id so re-runs are idempotent.
    """
    engine = _get_engine()
    segments_df.to_sql(
        "user_segments",
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )
    logger.info("Wrote %d rows to user_segments table.", len(segments_df))
