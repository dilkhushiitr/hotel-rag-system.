"""Train and evaluate the Isolation Forest fraud model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_detection.config import ensure_directories, load_config
from fraud_detection.model import evaluate_model, save_model, train_isolation_forest
from fraud_detection.pipeline import prepare_feature_table, save_version
from fraud_detection.risk_scoring import combine_scores
from fraud_detection.rule_engine import score_rules
from fraud_detection.model import anomaly_scores


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Isolation Forest fraud detection model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)

    features = prepare_feature_table(config)
    save_version(features, Path(config["paths"]["processed_data_dir"]) / "master_features.csv")

    model = train_isolation_forest(features, config)
    metrics = evaluate_model(model, features)
    save_model(model, config["paths"]["model_artifact"])

    scored = score_rules(features, config)
    scored["ml_anomaly_score"] = anomaly_scores(model, scored).round(2)
    scored = combine_scores(scored, config)
    save_version(scored, Path(config["paths"]["processed_data_dir"]) / "scored_training_data.csv")

    metrics_path = Path(config["paths"]["logs_dir"]) / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Training complete.")
    print(f"Model saved to: {Path(config['paths']['model_artifact']).resolve()}")
    print(f"Metrics saved to: {metrics_path.resolve()}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

