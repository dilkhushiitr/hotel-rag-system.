"""Batch scoring script for offline fraud detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud_detection.config import ensure_directories, load_config
from fraud_detection.pipeline import prepare_feature_table, save_version, score_feature_table


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch score transactions using trained fraud model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--output",
        default="data/processed/batch_scored_transactions.csv",
        help="Output CSV path for scored transactions.",
    )
    return parser.parse_args()


def main() -> None:
    """Run validation, feature engineering, and scoring for a batch."""
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)

    features = prepare_feature_table(config)
    scored = score_feature_table(features, config)
    output_path = save_version(scored, args.output)
    print(f"Batch scoring complete. Output written to: {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()

