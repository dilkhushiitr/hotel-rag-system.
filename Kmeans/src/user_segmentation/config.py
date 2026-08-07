"""Configuration loading utilities for the user segmentation project.

Production projects should avoid hard-coded paths and model parameters inside
training code. This module reads a YAML file and exposes a typed dataclass, which
makes the rest of the pipeline easier to test and safer to deploy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Typed wrapper around the YAML configuration file.

    The original YAML dictionary is retained in ``raw`` so newly added config
    keys can be used without changing this dataclass immediately.
    """

    root_dir: Path
    raw: dict[str, Any]

    @property
    def raw_data_dir(self) -> Path:
        return self.root_dir / self.raw["paths"]["raw_data_dir"]

    @property
    def processed_data_dir(self) -> Path:
        return self.root_dir / self.raw["paths"]["processed_data_dir"]

    @property
    def model_dir(self) -> Path:
        return self.root_dir / self.raw["paths"]["model_dir"]

    @property
    def report_dir(self) -> Path:
        return self.root_dir / self.raw["paths"]["report_dir"]

    @property
    def clustering_columns(self) -> list[str]:
        return list(self.raw["features"]["clustering_columns"])


def load_config(config_path: str | Path = "configs/config.yaml") -> ProjectConfig:
    """Load project configuration from YAML.

    Parameters
    ----------
    config_path:
        Path to the YAML config. It can be absolute or relative to the project
        root. The project root is inferred as the parent of the ``configs`` dir.
    """

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    # For a path like /project/configs/config.yaml, project root is /project.
    root_dir = path.parent.parent
    return ProjectConfig(root_dir=root_dir, raw=raw_config)

