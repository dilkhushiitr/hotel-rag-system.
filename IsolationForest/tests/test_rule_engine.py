from __future__ import annotations

import pandas as pd

from fraud_detection.config import load_config
from fraud_detection.rule_engine import evaluate_rules


def test_evaluate_rules_fires_expected_rules() -> None:
    config = load_config("configs/config.yaml")
    row = pd.Series(
        {
            "emulator_flag": 1,
            "vpn_flag": 1,
            "txn_count_1h": 9,
            "proxy_flag": 0,
            "geo_mismatch_flag": 1,
            "amount_deviation_ratio": 4,
            "accounts_per_device": 5,
            "rooted_flag": 1,
            "txn_count_24h": 30,
            "accounts_per_ip": 10,
            "failure_rate_24h": 0.7,
            "unverified_kyc": 1,
            "is_new_account": 1,
        }
    )

    score, rules_fired, details = evaluate_rules(row, config)

    assert score > 0
    assert "emulator_flag" in rules_fired
    assert "vpn_flag" in rules_fired
    assert details["emulator_flag"] == 30

