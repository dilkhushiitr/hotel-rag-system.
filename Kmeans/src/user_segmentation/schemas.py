"""Input schema definitions and validation helpers.

The project ingests seven CSV blocks. In real production, these would often come
from a warehouse, feature store, or event stream. For this assignment, CSV
validation catches common data-contract issues before the model trains on bad
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TableSchema:
    """Simple schema object for required CSV columns."""

    name: str
    required_columns: tuple[str, ...]


SCHEMAS: dict[str, TableSchema] = {
    "users": TableSchema("users", ("user_id", "signup_date", "country")),
    "sessions": TableSchema("sessions", ("session_id", "user_id", "duration", "timestamp")),
    "ad_events": TableSchema("ad_events", ("user_id", "impressions", "clicks")),
    "wallet": TableSchema("wallet", ("user_id", "earnings", "redeemed")),
    "transactions": TableSchema("transactions", ("user_id", "txn_amount", "txn_status")),
    "location": TableSchema("location", ("user_id", "city", "country")),
    "device": TableSchema("device", ("user_id", "device_type", "os")),
}


def validate_columns(table_name: str, data: pd.DataFrame) -> None:
    """Raise a clear error when a table is missing required columns."""

    schema = SCHEMAS[table_name]
    missing = sorted(set(schema.required_columns) - set(data.columns))
    if missing:
        raise ValueError(f"{table_name}.csv is missing required columns: {missing}")

