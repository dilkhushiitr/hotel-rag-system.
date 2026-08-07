"""Generate realistic sample CSVs for local testing.

Run:
    python scripts/generate_sample_data.py --output-dir data/raw --users 1000

The generated data intentionally contains different user behavior patterns:
high-value users, casual users, low-engagement users, and ad clickers. This
makes the clustering output easy to inspect even before real production data is
connected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(output_dir: Path, users: int, seed: int = 42) -> None:
    """Create seven CSV files matching the project schema."""

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_ids = np.array([f"user_{i:05d}" for i in range(users)])
    countries = rng.choice(["IN", "US", "BR", "ID", "PH"], size=users, p=[0.55, 0.15, 0.1, 0.1, 0.1])
    segments = rng.choice(["high_value", "casual", "low_engagement", "ad_clicker"], size=users, p=[0.18, 0.42, 0.25, 0.15])
    today = pd.Timestamp("2026-05-15", tz="UTC")

    signup_offsets = rng.integers(5, 365, size=users)
    pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_date": (today - pd.to_timedelta(signup_offsets, unit="D")).date.astype(str),
            "country": countries,
        }
    ).to_csv(output_dir / "users.csv", index=False)

    session_rows = []
    ad_rows = []
    wallet_rows = []
    txn_rows = []
    location_rows = []
    device_rows = []

    for user_id, segment, country in zip(user_ids, segments, countries):
        if segment == "high_value":
            session_count = rng.poisson(22) + 5
            duration_mean = 480
            impressions = rng.poisson(240)
            ctr = 0.05
            earnings = rng.gamma(8, 35)
            txn_count = rng.poisson(8)
        elif segment == "ad_clicker":
            session_count = rng.poisson(14) + 3
            duration_mean = 260
            impressions = rng.poisson(320)
            ctr = 0.16
            earnings = rng.gamma(4, 18)
            txn_count = rng.poisson(3)
        elif segment == "low_engagement":
            session_count = rng.poisson(2)
            duration_mean = 90
            impressions = rng.poisson(35)
            ctr = 0.01
            earnings = rng.gamma(1.5, 8)
            txn_count = rng.poisson(1)
        else:
            session_count = rng.poisson(7) + 1
            duration_mean = 180
            impressions = rng.poisson(100)
            ctr = 0.035
            earnings = rng.gamma(2.5, 12)
            txn_count = rng.poisson(2)

        for session_index in range(session_count):
            ts = today - pd.Timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))
            session_rows.append(
                {
                    "session_id": f"{user_id}_s{session_index}",
                    "user_id": user_id,
                    "duration": max(1, rng.normal(duration_mean, duration_mean * 0.25)),
                    "timestamp": ts.isoformat(),
                    "pages": int(max(1, rng.poisson(4))),
                }
            )

        clicks = rng.binomial(max(0, int(impressions)), min(0.95, ctr))
        ad_rows.append({"user_id": user_id, "impressions": impressions, "clicks": clicks})

        redeemed = earnings * rng.uniform(0.15, 0.85)
        wallet_rows.append({"user_id": user_id, "earnings": earnings, "redeemed": redeemed})

        for txn_index in range(txn_count):
            txn_rows.append(
                {
                    "user_id": user_id,
                    "txn_amount": float(rng.gamma(2.5, 30)),
                    "txn_status": rng.choice(["success", "fail"], p=[0.88, 0.12]),
                    "timestamp": (today - pd.Timedelta(days=int(rng.integers(0, 60)))).isoformat(),
                }
            )

        location_rows.append({"user_id": user_id, "city": rng.choice(["Bengaluru", "Delhi", "Mumbai", "Pune", "Hyderabad"]), "country": country})
        device_rows.append({"user_id": user_id, "device_type": rng.choice(["android", "ios", "tablet"], p=[0.78, 0.18, 0.04]), "os": rng.choice(["Android 14", "Android 13", "iOS 17", "iOS 18"])})

    pd.DataFrame(session_rows).to_csv(output_dir / "sessions.csv", index=False)
    pd.DataFrame(ad_rows).to_csv(output_dir / "ad_events.csv", index=False)
    pd.DataFrame(wallet_rows).to_csv(output_dir / "wallet.csv", index=False)
    pd.DataFrame(txn_rows).to_csv(output_dir / "transactions.csv", index=False)
    pd.DataFrame(location_rows).to_csv(output_dir / "location.csv", index=False)
    pd.DataFrame(device_rows).to_csv(output_dir / "device.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--users", type=int, default=1000)
    args = parser.parse_args()
    generate_sample_data(Path(args.output_dir), args.users)


if __name__ == "__main__":
    main()

