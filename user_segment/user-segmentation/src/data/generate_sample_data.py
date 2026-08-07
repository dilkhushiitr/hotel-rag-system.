"""
src/data/generate_sample_data.py
Generates realistic synthetic data for all raw tables and saves to CSV + DB.
Run: python src/data/generate_sample_data.py
"""

import numpy as np
import pandas as pd
import os
import sys
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.db import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

np.random.seed(42)

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

N_USERS = 10_000
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=90)


def random_dates(n, start=START_DATE, end=END_DATE):
    delta = (end - start).days
    return [start + timedelta(days=np.random.randint(0, delta)) for _ in range(n)]


def generate_users(n=N_USERS) -> pd.DataFrame:
    """users table — one row per user"""
    log.info("Generating users table...")
    return pd.DataFrame({
        "user_id":      [f"user_{i:05d}" for i in range(n)],
        "created_at":   random_dates(n, START_DATE - timedelta(days=180), END_DATE),
        "country":      np.random.choice(["IN", "US", "BR", "ID", "NG"], n,
                                         p=[0.35, 0.20, 0.15, 0.15, 0.15]),
        "age_group":    np.random.choice(["18-24", "25-34", "35-44", "45+"], n),
        "gender":       np.random.choice(["M", "F", "Other"], n, p=[0.52, 0.45, 0.03]),
        "is_active":    np.random.choice([True, False], n, p=[0.75, 0.25]),
    })


def generate_sessions(users_df: pd.DataFrame) -> pd.DataFrame:
    """sessions table — multiple rows per user"""
    log.info("Generating sessions table...")
    rows = []
    for uid in users_df["user_id"]:
        # Power-law: most users have few sessions, a few have many
        n_sessions = max(1, int(np.random.exponential(scale=8)))
        for _ in range(n_sessions):
            rows.append({
                "user_id":          uid,
                "session_id":       f"sess_{np.random.randint(1e9):010d}",
                "session_duration": max(10, int(np.random.lognormal(mean=4.5, sigma=1.2))),  # seconds
                "timestamp":        random_dates(1)[0],
                "platform":         np.random.choice(["android", "ios", "web"], p=[0.55, 0.35, 0.10]),
                "page_views":       np.random.randint(1, 20),
            })
    return pd.DataFrame(rows)


def generate_ad_events(users_df: pd.DataFrame) -> pd.DataFrame:
    """ad_events table — impressions and clicks per user per day"""
    log.info("Generating ad_events table...")
    rows = []
    for uid in users_df["user_id"]:
        n_days = np.random.randint(0, 30)
        for _ in range(n_days):
            impressions = np.random.randint(0, 50)
            # CTR varies by user type — add natural heterogeneity
            true_ctr = np.clip(np.random.beta(a=1.5, b=10), 0.01, 0.40)
            clicks = np.random.binomial(impressions, true_ctr)
            rows.append({
                "user_id":     uid,
                "date":        random_dates(1)[0].date(),
                "impressions": impressions,
                "clicks":      clicks,
                "ad_type":     np.random.choice(["banner", "interstitial", "rewarded"],
                                                p=[0.4, 0.35, 0.25]),
                "revenue":     round(clicks * np.random.uniform(0.01, 0.15), 4),
            })
    return pd.DataFrame(rows)


def generate_wallet(users_df: pd.DataFrame) -> pd.DataFrame:
    """wallet table — current wallet state per user"""
    log.info("Generating wallet table...")
    n = len(users_df)
    return pd.DataFrame({
        "user_id":          users_df["user_id"].values,
        "balance":          np.round(np.random.exponential(scale=5.0, size=n), 2),
        "total_earned":     np.round(np.random.exponential(scale=15.0, size=n), 2),
        "total_redeemed":   np.round(np.random.exponential(scale=10.0, size=n), 2),
        "last_redemption":  [d.date() if np.random.rand() > 0.3 else None
                             for d in random_dates(n)],
    })


def generate_transactions(users_df: pd.DataFrame) -> pd.DataFrame:
    """transactions table"""
    log.info("Generating transactions table...")
    rows = []
    for uid in users_df["user_id"]:
        n_txns = np.random.poisson(lam=3)
        for _ in range(n_txns):
            rows.append({
                "user_id":   uid,
                "txn_id":    f"txn_{np.random.randint(1e9):010d}",
                "amount":    round(np.random.lognormal(mean=1.5, sigma=1.0), 2),
                "status":    np.random.choice(["success", "failed", "pending"],
                                              p=[0.85, 0.10, 0.05]),
                "timestamp": random_dates(1)[0],
                "category":  np.random.choice(["topup", "reward", "withdrawal"]),
            })
    return pd.DataFrame(rows)


def generate_device(users_df: pd.DataFrame) -> pd.DataFrame:
    """device table — one row per user"""
    log.info("Generating device table...")
    n = len(users_df)
    return pd.DataFrame({
        "user_id":    users_df["user_id"].values,
        "os":         np.random.choice(["android", "ios"], n, p=[0.65, 0.35]),
        "os_version": np.random.choice(["12", "13", "14", "15"], n),
        "model":      np.random.choice(["budget", "mid-range", "flagship"], n,
                                       p=[0.5, 0.35, 0.15]),
        "ram_gb":     np.random.choice([2, 4, 6, 8, 12], n),
    })


def save_to_csv(df: pd.DataFrame, name: str):
    path = os.path.join(RAW_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    log.info(f"Saved {name}.csv — {len(df):,} rows → {path}")


def save_to_db(df: pd.DataFrame, table_name: str):
    try:
        engine = get_engine()
        df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=1000)
        log.info(f"Loaded {table_name} → DB ({len(df):,} rows)")
    except Exception as e:
        log.warning(f"DB write failed for {table_name}: {e} — CSV only")


def main():
    log.info("=" * 60)
    log.info("Generating synthetic dataset...")
    log.info("=" * 60)

    users        = generate_users()
    sessions     = generate_sessions(users)
    ad_events    = generate_ad_events(users)
    wallet       = generate_wallet(users)
    transactions = generate_transactions(users)
    device       = generate_device(users)

    tables = {
        "users":        users,
        "sessions":     sessions,
        "ad_events":    ad_events,
        "wallet":       wallet,
        "transactions": transactions,
        "device":       device,
    }

    for name, df in tables.items():
        save_to_csv(df, name)
        save_to_db(df, name)

    log.info("=" * 60)
    log.info("Done! Summary:")
    for name, df in tables.items():
        log.info(f"  {name:<15} {len(df):>8,} rows")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
