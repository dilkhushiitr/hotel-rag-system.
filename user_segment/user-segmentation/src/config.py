"""
Central configuration for the User Segmentation project.
All tuneable knobs live here — override via environment variables or .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR     = BASE_DIR / "models" / "saved"
LOGS_DIR       = BASE_DIR / "logs"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "postgres")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_NAME     = os.getenv("DB_NAME", "segmentation_db")
DB_USER     = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "secret")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Feature Engineering ────────────────────────────────────────────────────────
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", 30))   # rolling window for features

# ── K-Means ────────────────────────────────────────────────────────────────────
KMEANS_K_MIN     = int(os.getenv("KMEANS_K_MIN", 2))
KMEANS_K_MAX     = int(os.getenv("KMEANS_K_MAX", 12))
KMEANS_FINAL_K   = int(os.getenv("KMEANS_FINAL_K", 5))   # set after elbow analysis
KMEANS_INIT      = "k-means++"
KMEANS_N_INIT    = 10
KMEANS_MAX_ITER  = 300
RANDOM_STATE     = 42

# ── DBSCAN ─────────────────────────────────────────────────────────────────────
DBSCAN_EPS       = float(os.getenv("DBSCAN_EPS", 2.5))
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", 10))

# ── Segment Labels (must match cluster indices from profiling) ─────────────────
SEGMENT_LABELS = {
    0: "High-Value Power Users",
    1: "Casual Earners",
    2: "Ad Hunters",
    3: "Dormant Users",
    4: "At-Risk Users",
   -1: "Anomalies / Fraud Suspects",   # DBSCAN noise label
}

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

# ── Monitoring ─────────────────────────────────────────────────────────────────
DRIFT_THRESHOLD      = float(os.getenv("DRIFT_THRESHOLD", 0.15))   # PSI threshold
RETRAIN_TRIGGER_DAYS = int(os.getenv("RETRAIN_TRIGGER_DAYS", 7))

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
