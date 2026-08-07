"""
src/data/db.py
Database connection factory using SQLAlchemy.
Reads credentials from environment variables.
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)


def get_connection_string() -> str:
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5432")
    db       = os.getenv("POSTGRES_DB", "segmentation")
    user     = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "changeme")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def get_engine() -> Engine:
    conn_str = get_connection_string()
    engine = create_engine(
        conn_str,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,      # auto-reconnect on stale connections
    )
    return engine


def test_connection() -> bool:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("DB connection OK")
        return True
    except Exception as e:
        log.error(f"DB connection failed: {e}")
        return False
