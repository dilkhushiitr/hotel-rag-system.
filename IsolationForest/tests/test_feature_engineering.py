from __future__ import annotations

import pandas as pd

from fraud_detection.config import load_config
from fraud_detection.feature_engineering import FEATURE_COLUMNS, build_features


def test_build_features_contains_required_columns() -> None:
    config = load_config("configs/config.yaml")
    df = pd.DataFrame(
        [
            {
                "txn_id": "TXN1",
                "user_id": "U1",
                "txn_amount": 100,
                "account_age_days": 3,
                "kyc_status": "unverified",
                "os_type": "android",
                "session_duration": 10,
                "click_count": 100,
                "txn_count_1h": 6,
                "txn_count_24h": 10,
                "avg_txn_amount_7d": 20,
                "wallet_balance_change_24h": 100,
            }
        ]
    )

    features = build_features(df, config)

    for column in FEATURE_COLUMNS:
        assert column in features.columns
    assert features.loc[0, "is_new_account"] == 1
    assert features.loc[0, "high_velocity_1h"] == 1

