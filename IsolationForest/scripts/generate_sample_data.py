"""Generate realistic sample data for local testing and interviews."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
RAW_DIR = Path("data/raw")


def main() -> None:
    """Create five CSV files that follow the project schema."""
    rng = np.random.default_rng(RANDOM_SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    n_users = 1200
    n_transactions = 5000
    user_ids = np.array([f"U{i:05d}" for i in range(n_users)])

    fraud_user_mask = rng.random(n_users) < 0.06

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "account_age_days": rng.integers(1, 365, n_users),
            "kyc_status": np.where(fraud_user_mask, "unverified", rng.choice(["verified", "unverified"], n_users, p=[0.8, 0.2])),
            "signup_country": rng.choice(["IN", "US", "ID", "BR"], n_users, p=[0.7, 0.1, 0.1, 0.1]),
            "signup_city": rng.choice(["Bengaluru", "Delhi", "Mumbai", "Jakarta", "New York"], n_users),
            "referral_count": np.where(fraud_user_mask, rng.integers(5, 30, n_users), rng.integers(0, 5, n_users)),
            "referral_depth": np.where(fraud_user_mask, rng.integers(2, 6, n_users), rng.integers(0, 2, n_users)),
        }
    )

    devices = pd.DataFrame(
        {
            "device_id": [f"D{i:05d}" for i in range(n_users)],
            "user_id": user_ids,
            "os_type": rng.choice(["android", "ios"], n_users, p=[0.85, 0.15]),
            "app_version": rng.choice(["1.0.0", "1.1.0", "1.2.0"], n_users),
            "emulator_flag": np.where(fraud_user_mask, rng.binomial(1, 0.45, n_users), rng.binomial(1, 0.02, n_users)),
            "rooted_flag": np.where(fraud_user_mask, rng.binomial(1, 0.35, n_users), rng.binomial(1, 0.03, n_users)),
            "accounts_per_device": np.where(fraud_user_mask, rng.integers(4, 15, n_users), rng.integers(1, 4, n_users)),
        }
    )

    network = pd.DataFrame(
        {
            "user_id": user_ids,
            "ip_address": [f"10.0.{i // 255}.{i % 255}" for i in range(n_users)],
            "ip_country": rng.choice(["IN", "US", "ID", "BR"], n_users),
            "ip_city": rng.choice(["Bengaluru", "Delhi", "Mumbai", "Jakarta", "New York"], n_users),
            "vpn_flag": np.where(fraud_user_mask, rng.binomial(1, 0.55, n_users), rng.binomial(1, 0.04, n_users)),
            "proxy_flag": np.where(fraud_user_mask, rng.binomial(1, 0.35, n_users), rng.binomial(1, 0.02, n_users)),
            "geo_mismatch_flag": np.where(fraud_user_mask, rng.binomial(1, 0.5, n_users), rng.binomial(1, 0.05, n_users)),
            "accounts_per_ip": np.where(fraud_user_mask, rng.integers(6, 25, n_users), rng.integers(1, 5, n_users)),
        }
    )

    behavioral = pd.DataFrame(
        {
            "session_id": [f"S{i:05d}" for i in range(n_users)],
            "user_id": user_ids,
            "session_duration": np.where(fraud_user_mask, rng.uniform(5, 60, n_users), rng.uniform(120, 1800, n_users)),
            "click_count": np.where(fraud_user_mask, rng.integers(100, 500, n_users), rng.integers(5, 100, n_users)),
            "time_between_actions": np.where(fraud_user_mask, rng.uniform(0.01, 0.2, n_users), rng.uniform(0.5, 8, n_users)),
            "screen_flow_length": np.where(fraud_user_mask, rng.integers(1, 5, n_users), rng.integers(4, 20, n_users)),
            "click_entropy": np.where(fraud_user_mask, rng.uniform(0.01, 0.35, n_users), rng.uniform(0.45, 0.95, n_users)),
            "behavior_score": np.where(fraud_user_mask, rng.uniform(70, 100, n_users), rng.uniform(0, 45, n_users)),
        }
    )

    txn_user_ids = rng.choice(user_ids, n_transactions)
    user_fraud_lookup = dict(zip(user_ids, fraud_user_mask))
    txn_fraud = np.array([user_fraud_lookup[user_id] for user_id in txn_user_ids])
    timestamps = pd.date_range("2026-01-01", periods=n_transactions, freq="10min", tz="UTC")

    transactions = pd.DataFrame(
        {
            "txn_id": [f"TXN{i:06d}" for i in range(n_transactions)],
            "user_id": txn_user_ids,
            "txn_amount": np.where(txn_fraud, rng.lognormal(6.2, 1.0, n_transactions), rng.lognormal(4.2, 0.7, n_transactions)).round(2),
            "txn_type": rng.choice(["earn", "redeem", "transfer"], n_transactions, p=[0.55, 0.35, 0.10]),
            "txn_timestamp": rng.choice(timestamps, n_transactions),
            "txn_status": np.where(txn_fraud, rng.choice(["success", "failed"], n_transactions, p=[0.55, 0.45]), rng.choice(["success", "failed"], n_transactions, p=[0.95, 0.05])),
            "is_fraud": txn_fraud.astype(int),
        }
    )

    users.to_csv(RAW_DIR / "users.csv", index=False)
    transactions.to_csv(RAW_DIR / "transactions.csv", index=False)
    devices.to_csv(RAW_DIR / "devices.csv", index=False)
    network.to_csv(RAW_DIR / "network.csv", index=False)
    behavioral.to_csv(RAW_DIR / "behavioral.csv", index=False)

    print(f"Sample data written to {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()

