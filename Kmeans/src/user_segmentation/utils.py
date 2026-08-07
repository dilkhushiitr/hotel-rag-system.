"""Shared utility helpers for logging and filesystem operations."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure consistent logs for CLI, training jobs, and API startup."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories(*paths: Path) -> None:
    """Create output directories when they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

