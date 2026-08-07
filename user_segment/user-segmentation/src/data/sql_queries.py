"""
src/data/sql_queries.py
All SQL queries for the 30-day feature extraction window.
Returns pandas DataFrames.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text

from src.data.db import get_engine

log = logging.getLogger(__name__)


def get_date_range(lookback_days: int = 30):
    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=lookback_days)
    return start_date, end_date


# ────────────────────────────────────────────────
# 1. User base
# ────────────────────────────────────────────────
USER_BASE_SQL = """
SELECT
    user_id,
    created_at,
    country,
    age_group,
    gender,
    is_active
FROM users
WHERE is_active = TRUE
"""


# ────────────────────────────────────────────────
# 2. Session aggregates (30-day window)
# ────────────────────────────────────────────────
SESSION_AGG_SQL = """
SELECT
    user_id,
    COUNT(*)                        AS sessions_per_user,
    AVG(session_duration)           AS avg_session_duration,
    COUNT(DISTINCT DATE(timestamp)) AS active_days,
    MAX(timestamp)                  AS last_active,
    SUM(page_views)                 AS total_page_views
FROM sessions
WHERE timestamp >= :start_date
  AND timestamp <  :end_date
GROUP BY user_id
"""


# ────────────────────────────────────────────────
# 3. Ad event aggregates
# ────────────────────────────────────────────────
AD_AGG_SQL = """
SELECT
    user_id,
    SUM(impressions)                        AS total_impressions,
    SUM(clicks)                             AS total_clicks,
    CASE
        WHEN SUM(impressions) > 0
        THEN CAST(SUM(clicks) AS FLOAT) / SUM(impressions)
        ELSE 0
    END                                     AS ctr,
    SUM(revenue)                            AS ad_revenue,
    COUNT(DISTINCT ad_type)                 AS ad_type_diversity
FROM ad_events
WHERE date >= :start_date
  AND date <  :end_date
GROUP BY user_id
"""


# ────────────────────────────────────────────────
# 4. Wallet snapshot
# ────────────────────────────────────────────────
WALLET_SQL = """
SELECT
    user_id,
    balance             AS wallet_balance,
    total_earned,
    total_redeemed,
    last_redemption
FROM wallet
"""


# ────────────────────────────────────────────────
# 5. Transaction aggregates
# ────────────────────────────────────────────────
TXN_AGG_SQL = """
SELECT
    user_id,
    COUNT(*)                                         AS total_transactions,
    AVG(amount)                                      AS avg_transaction_value,
    SUM(amount)                                      AS total_transaction_value,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
        / CAST(COUNT(*) AS FLOAT)                    AS success_rate
FROM transactions
WHERE timestamp >= :start_date
  AND timestamp <  :end_date
GROUP BY user_id
"""


# ────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────
def run_query(sql: str, params: dict = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=result.keys())


# ────────────────────────────────────────────────
# Main extractor
# ────────────────────────────────────────────────
def extract_all_features(lookback_days: int = 30) -> pd.DataFrame:
    """
    Joins all feature tables on user_id.
    Returns one row per user with all raw features.
    """
    start_date, end_date = get_date_range(lookback_days)
    params = {"start_date": str(start_date), "end_date": str(end_date)}

    log.info(f"Extracting data from {start_date} to {end_date}")

    users    = run_query(USER_BASE_SQL)
    sessions = run_query(SESSION_AGG_SQL, params)
    ads      = run_query(AD_AGG_SQL, params)
    wallet   = run_query(WALLET_SQL)
    txns     = run_query(TXN_AGG_SQL, params)

    log.info(f"Users: {len(users):,} | Sessions: {len(sessions):,} | "
             f"Ads: {len(ads):,} | Wallet: {len(wallet):,} | Txns: {len(txns):,}")

    # Left joins — keep all active users
    df = users.merge(sessions, on="user_id", how="left")
    df = df.merge(ads,     on="user_id", how="left")
    df = df.merge(wallet,  on="user_id", how="left")
    df = df.merge(txns,    on="user_id", how="left")

    log.info(f"Merged feature table: {df.shape}")
    return df
