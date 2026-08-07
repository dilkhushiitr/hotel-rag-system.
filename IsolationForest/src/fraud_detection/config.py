"""Configuration loading utilities.

Production systems should avoid hard-coded thresholds and file paths.
This module loads the YAML config once and gives the rest of the code a
validated dictionary-like object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load project configuration from YAML.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing paths, model parameters, thresholds, and weights.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML file is empty or invalid for this project.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    required_sections = {"paths", "model", "scoring", "rules", "thresholds"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Config missing required sections: {sorted(missing)}")

    return config


def ensure_directories(config: dict[str, Any]) -> None:
    """Create output directories declared in the config.

    This keeps scripts idempotent: training and scoring can run on a fresh
    machine without manual folder creation.
    """
    paths = config["paths"]
    for key in ("processed_data_dir", "model_dir", "logs_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)

