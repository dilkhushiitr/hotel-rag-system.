"""End-to-end training and scoring pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_detection.data_cleaning import clean_behavioral, clean_devices, clean_network, clean_transactions, clean_users
from fraud_detection.data_validation import validate_all_tables
from fraud_detection.feature_engineering import build_features, merge_tables
from fraud_detection.model import anomaly_scores, load_model
from fraud_detection.risk_scoring import combine_scores
from fraud_detection.rule_engine import score_rules


def load_raw_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read all raw CSV files declared in config."""
    paths = config["paths"]
    users = pd.read_csv(paths["users"])
    transactions = pd.read_csv(paths["transactions"])
    devices = pd.read_csv(paths["devices"])
    network = pd.read_csv(paths["network"])
    behavioral = pd.read_csv(paths["behavioral"])
    return users, transactions, devices, network, behavioral


def prepare_feature_table(config: dict) -> pd.DataFrame:
    """Validate, clean, merge, and engineer features from raw tables."""
    users, transactions, devices, network, behavioral = load_raw_tables(config)
    validate_all_tables(users, transactions, devices, network, behavioral)

    users = clean_users(users)
    transactions = clean_transactions(transactions)
    devices = clean_devices(devices)
    network = clean_network(network)
    behavioral = clean_behavioral(behavioral)

    master = merge_tables(users, transactions, devices, network, behavioral)
    features = build_features(master, config)
    return features


def score_feature_table(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply rule scoring, ML scoring, and final decisioning."""
    model = load_model(config["paths"]["model_artifact"])
    scored = score_rules(features, config)
    scored["ml_anomaly_score"] = anomaly_scores(model, scored).round(2)
    scored = combine_scores(scored, config)
    return scored


def save_version(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a dataframe version to disk for reproducibility."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path

