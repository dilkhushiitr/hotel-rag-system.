"""Command-line interface for training and batch inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from user_segmentation.config import load_config
from user_segmentation.pipeline import batch_predict_pipeline, train_pipeline
from user_segmentation.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint installed as ``user-segmentation``."""

    parser = argparse.ArgumentParser(description="GreedyGame user segmentation pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML configuration file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train K-Means and DBSCAN models")
    train_parser.add_argument("--data-dir", default=None, help="Directory containing the seven input CSV files")

    predict_parser = subparsers.add_parser("predict", help="Run batch prediction from CSV files")
    predict_parser.add_argument("--artifact", default="models/segmentation_artifacts.joblib", help="Trained artifact path")
    predict_parser.add_argument("--data-dir", default=None, help="Directory containing CSV files to score")
    predict_parser.add_argument("--output", default=None, help="Output CSV path")

    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)

    if args.command == "train":
        outputs = train_pipeline(config, data_dir=args.data_dir)
        for name, path in outputs.items():
            LOGGER.info("%s written to %s", name, path)

    if args.command == "predict":
        output = batch_predict_pipeline(
            config=config,
            artifact_path=Path(args.artifact),
            data_dir=args.data_dir,
            output_path=args.output,
        )
        LOGGER.info("Predictions written to %s", output)


if __name__ == "__main__":
    main()

