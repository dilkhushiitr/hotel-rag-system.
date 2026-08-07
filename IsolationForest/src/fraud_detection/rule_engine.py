"""Weighted deterministic fraud rule engine."""

from __future__ import annotations

import pandas as pd


def _is_triggered(value: bool) -> bool:
    """Convert numpy/pandas boolean values into a Python bool."""
    return bool(value)


def evaluate_rules(row: pd.Series, config: dict) -> tuple[float, list[str], dict[str, float]]:
    """Evaluate all 13 fraud rules for one transaction.

    Returns:
        rule_score: Normalized 0-100 rule risk score.
        rules_fired: Names of triggered rules.
        rule_details: Mapping of rule name to configured weight.
    """
    weights = config["rules"]
    thresholds = config["thresholds"]

    checks = {
        "emulator_flag": row.get("emulator_flag", 0) == 1,
        "vpn_flag": row.get("vpn_flag", 0) == 1,
        "high_velocity_1h": row.get("txn_count_1h", 0) > thresholds["high_velocity_1h"],
        "proxy_flag": row.get("proxy_flag", 0) == 1,
        "geo_mismatch": row.get("geo_mismatch_flag", 0) == 1,
        "high_deviation": row.get("amount_deviation_ratio", 0) > thresholds["amount_deviation_multiplier"],
        "device_sharing": row.get("accounts_per_device", 1) > thresholds["accounts_per_device"],
        "rooted_flag": row.get("rooted_flag", 0) == 1,
        "high_velocity_24h": row.get("txn_count_24h", 0) > thresholds["high_velocity_24h"],
        "ip_sharing": row.get("accounts_per_ip", 1) > thresholds["accounts_per_ip"],
        "high_failure_rate": row.get("failure_rate_24h", 0) > thresholds["high_failure_rate"],
        "unverified_kyc": row.get("unverified_kyc", 0) == 1,
        "new_account": row.get("is_new_account", 0) == 1,
    }

    rules_fired = [name for name, triggered in checks.items() if _is_triggered(triggered)]
    rule_details = {name: float(weights[name]) for name in rules_fired}
    raw_score = sum(rule_details.values())
    max_score = sum(float(weight) for weight in weights.values())
    normalized_score = 100.0 * raw_score / max_score if max_score else 0.0
    return round(normalized_score, 2), rules_fired, rule_details


def score_rules(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the weighted rule engine to a full dataframe."""
    scored = df.copy()
    outputs = scored.apply(lambda row: evaluate_rules(row, config), axis=1)
    scored["rule_score"] = [item[0] for item in outputs]
    scored["rules_fired"] = [item[1] for item in outputs]
    scored["rule_details"] = [item[2] for item in outputs]
    return scored

