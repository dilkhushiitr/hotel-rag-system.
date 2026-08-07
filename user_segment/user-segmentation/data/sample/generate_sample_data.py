"""
Generate realistic sample data for the User Segmentation project.
Run this once to populate the data/raw/ directory with CSVs.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

N_USERS = 10_000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 1, 31)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def random_dates(start, end, n):
    delta = (end - start).days
    return [start + timedelta(days=random.randint(0, delta)) for _ in range(n)]


# ── 1. USERS TABLE ─────────────────────────────────────────────────────────────
print("Generating users table...")
user_ids = [f"U{str(i).zfill(6)}" for i in range(1, N_USERS + 1)]
created_dates = random_dates(datetime(2023, 1, 1), START_DATE, N_USERS)

users_df = pd.DataFrame({
    "user_id":      user_ids,
    "created_at":   created_dates,
    "country":      np.random.choice(["IN", "US", "GB", "DE", "BR", "AU"], N_USERS, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1]),
    "device_type":  np.random.choice(["android", "ios", "web"], N_USERS, p=[0.5, 0.3, 0.2]),
    "age_bucket":   np.random.choice(["18-24", "25-34", "35-44", "45+"], N_USERS, p=[0.3, 0.4, 0.2, 0.1]),
    "referral_src": np.random.choice(["organic", "paid_ad", "social", "referral"], N_USERS, p=[0.3, 0.3, 0.2, 0.2]),
})
users_df.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)
print(f"  users.csv: {len(users_df):,} rows")


# ── 2. SESSIONS TABLE ──────────────────────────────────────────────────────────
print("Generating sessions table...")

# Power-law distribution: most users have few sessions, some have many
session_counts = np.random.negative_binomial(2, 0.15, N_USERS).clip(0, 200)

rows = []
for uid, n_sess in zip(user_ids, session_counts):
    for _ in range(n_sess):
        dt = random.choice(random_dates(START_DATE, END_DATE, 1))
        rows.append({
            "session_id":       f"S{random.randint(100000, 999999)}",
            "user_id":          uid,
            "session_start":    dt,
            "duration_seconds": max(10, int(np.random.exponential(180))),
            "pages_viewed":     np.random.randint(1, 20),
            "platform":         np.random.choice(["android", "ios", "web"], p=[0.5, 0.3, 0.2]),
        })

sessions_df = pd.DataFrame(rows)
sessions_df.to_csv(os.path.join(OUTPUT_DIR, "sessions.csv"), index=False)
print(f"  sessions.csv: {len(sessions_df):,} rows")


# ── 3. AD EVENTS TABLE ────────────────────────────────────────────────────────
print("Generating ad_events table...")

ad_rows = []
for uid, n_sess in zip(user_ids, session_counts):
    n_impressions = n_sess * np.random.randint(2, 8)
    n_clicks      = int(n_impressions * np.random.beta(1, 15))  # ~6% CTR average
    for _ in range(n_impressions):
        dt = random.choice(random_dates(START_DATE, END_DATE, 1))
        ad_rows.append({
            "event_id":   f"AE{random.randint(1000000, 9999999)}",
            "user_id":    uid,
            "ad_id":      f"AD{np.random.randint(1, 500)}",
            "event_type": "impression",
            "timestamp":  dt,
            "ad_category": np.random.choice(["gaming", "finance", "ecom", "travel", "health"]),
        })
    for _ in range(n_clicks):
        dt = random.choice(random_dates(START_DATE, END_DATE, 1))
        ad_rows.append({
            "event_id":   f"AE{random.randint(1000000, 9999999)}",
            "user_id":    uid,
            "ad_id":      f"AD{np.random.randint(1, 500)}",
            "event_type": "click",
            "timestamp":  dt,
            "ad_category": np.random.choice(["gaming", "finance", "ecom", "travel", "health"]),
        })

ad_df = pd.DataFrame(ad_rows)
ad_df.to_csv(os.path.join(OUTPUT_DIR, "ad_events.csv"), index=False)
print(f"  ad_events.csv: {len(ad_df):,} rows")


# ── 4. WALLET TABLE ───────────────────────────────────────────────────────────
print("Generating wallet table...")

wallet_df = pd.DataFrame({
    "user_id":          user_ids,
    "total_earnings":   np.abs(np.random.exponential(50, N_USERS)).round(2),
    "total_redeemed":   np.abs(np.random.exponential(20, N_USERS)).round(2),
    "wallet_balance":   np.abs(np.random.exponential(30, N_USERS)).round(2),
    "last_redemption":  [random.choice(random_dates(START_DATE, END_DATE, 1)) if random.random() > 0.3 else None for _ in range(N_USERS)],
})
wallet_df.to_csv(os.path.join(OUTPUT_DIR, "wallet.csv"), index=False)
print(f"  wallet.csv: {len(wallet_df):,} rows")


# ── 5. TRANSACTIONS TABLE ─────────────────────────────────────────────────────
print("Generating transactions table...")

txn_counts = np.random.negative_binomial(1, 0.3, N_USERS).clip(0, 50)
txn_rows = []
for uid, n_txn in zip(user_ids, txn_counts):
    for _ in range(n_txn):
        dt = random.choice(random_dates(START_DATE, END_DATE, 1))
        status = np.random.choice(["success", "failed", "pending"], p=[0.8, 0.15, 0.05])
        txn_rows.append({
            "txn_id":     f"T{random.randint(1000000, 9999999)}",
            "user_id":    uid,
            "amount":     round(abs(np.random.exponential(200)), 2),
            "status":     status,
            "txn_type":   np.random.choice(["purchase", "withdrawal", "deposit"]),
            "created_at": dt,
        })

txn_df = pd.DataFrame(txn_rows)
txn_df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)
print(f"  transactions.csv: {len(txn_df):,} rows")

print("\n✅ All sample data generated successfully!")
print(f"   Output: {os.path.abspath(OUTPUT_DIR)}")
