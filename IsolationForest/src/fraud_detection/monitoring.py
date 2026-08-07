"""Lightweight monitoring utilities.

The API writes prediction logs as JSON Lines. In production these records should
be shipped to a warehouse, observability stack, or model monitoring platform.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_prediction(record: dict[str, Any], logs_dir: str = "logs") -> None:
    """Append one prediction event to the daily JSONL log."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    event = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    log_path = Path(logs_dir) / "prediction_events.jsonl"
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, default=str) + "\n")

