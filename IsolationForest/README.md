# Isolation Forest Fraud Detection

Production-style hybrid fraud detection project for wallet-based gaming transactions.

The system combines:

- 13 deterministic fraud rules for known fraud patterns.
- Isolation Forest anomaly detection for unknown or evolving fraud behavior.
- Final risk scoring with LOW, MEDIUM, and HIGH risk decisions.
- Batch training, batch scoring, REST API serving, Docker deployment, and monitoring logs.

## Project Structure

```text
IsolationForest/
  configs/config.yaml              # Business thresholds, model params, file paths
  data/raw/                        # Input CSV files
  data/processed/                  # Cleaned datasets, feature tables, scored outputs
  models/                          # Trained model artifacts
  scripts/train_model.py           # Train + evaluate + save model
  scripts/batch_score.py           # Score a batch CSV/table
  scripts/generate_sample_data.py  # Create demo data with the expected schema
  src/fraud_detection/             # Production Python package
  tests/                           # Unit tests
  Dockerfile                       # API container
  docker-compose.yml               # Local API deployment
```

## Setup

```bash
cd /Users/Dilkhush1/Downloads/projects/IsolationForest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Create Demo Data

```bash
python scripts/generate_sample_data.py
```

This creates realistic CSVs in `data/raw/`:

- `users.csv`
- `transactions.csv`
- `devices.csv`
- `network.csv`
- `behavioral.csv`

## Train Model

```bash
python scripts/train_model.py --config configs/config.yaml
```

Outputs:

- `models/fraud_iforest_pipeline.joblib`
- `data/processed/master_features.csv`
- `data/processed/scored_training_data.csv`
- `logs/training_metrics.json`

## Batch Score

```bash
python scripts/batch_score.py --config configs/config.yaml
```

Output:

- `data/processed/batch_scored_transactions.csv`

## Run API Locally

```bash
uvicorn fraud_detection.api:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Score one transaction:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "txn_id": "TXN999",
    "user_id": "U999",
    "txn_amount": 1500,
    "txn_type": "redeem",
    "txn_timestamp": "2026-05-14T10:00:00Z",
    "txn_status": "success",
    "account_age_days": 2,
    "kyc_status": "unverified",
    "signup_country": "IN",
    "signup_city": "Bengaluru",
    "referral_count": 8,
    "referral_depth": 3,
    "device_id": "D999",
    "os_type": "android",
    "app_version": "1.0.0",
    "emulator_flag": 1,
    "rooted_flag": 1,
    "accounts_per_device": 5,
    "ip_address": "10.0.0.1",
    "ip_country": "US",
    "ip_city": "New York",
    "vpn_flag": 1,
    "proxy_flag": 0,
    "geo_mismatch_flag": 1,
    "accounts_per_ip": 10,
    "session_duration": 15,
    "click_count": 200,
    "time_between_actions": 0.05,
    "screen_flow_length": 3,
    "click_entropy": 0.1,
    "behavior_score": 90,
    "txn_count_1h": 8,
    "txn_count_24h": 30,
    "avg_txn_amount_7d": 100,
    "failure_rate_24h": 0.6,
    "wallet_balance_change_24h": 2000
  }'
```

## Docker

```bash
docker compose up --build
```

API URL:

```text
http://localhost:8000
```

## DVC and MLflow Notes

This project is ready for DVC and MLflow integration.

Typical production commands:

```bash
dvc init
dvc add data/raw data/processed models
git add data/.gitignore models/.gitignore *.dvc
mlflow ui --host 0.0.0.0 --port 5000
```

The training script logs metrics locally to `logs/training_metrics.json`. If MLflow is installed and configured, you can extend `scripts/train_model.py` to call `mlflow.log_metric`, `mlflow.log_param`, and `mlflow.sklearn.log_model`.

