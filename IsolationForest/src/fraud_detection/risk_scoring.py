"""Final score combination and decision engine."""

from __future__ import annotations

import pandas as pd


def risk_level_and_decision(score: float, config: dict) -> tuple[str, str]:
    """Map final fraud score to business action."""
    scoring = config["scoring"]
    if score < scoring["low_threshold"]:
        return "LOW", "ALLOW"
    if score < scoring["high_threshold"]:
        return "MEDIUM", "MANUAL_REVIEW"
    return "HIGH", "BLOCK"


def combine_scores(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Combine deterministic rule score and ML anomaly score."""
    scored = df.copy()
    rule_weight = config["scoring"]["rule_weight"]
    ml_weight = config["scoring"]["ml_weight"]

    scored["final_fraud_score"] = (
        rule_weight * scored["rule_score"] + ml_weight * scored["ml_anomaly_score"]
    ).round(2)

    decisions = scored["final_fraud_score"].apply(lambda score: risk_level_and_decision(score, config))
    scored["risk_level"] = [item[0] for item in decisions]
    scored["decision"] = [item[1] for item in decisions]
    scored["explanation"] = scored.apply(_build_explanation, axis=1)
    return scored


def _build_explanation(row: pd.Series) -> str:
    """Create a short human-readable reason for the score."""
    rules = row.get("rules_fired", [])
    if rules:
        rule_text = ", ".join(rules[:5])
        return (
            f"{row['decision']} because final score is {row['final_fraud_score']} "
            f"with rules fired: {rule_text}."
        )
    return (
        f"{row['decision']} because final score is {row['final_fraud_score']} "
        "with no deterministic fraud rules fired."
    )

