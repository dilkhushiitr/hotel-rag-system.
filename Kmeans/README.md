# GreedyGame User Segmentation Using K-Means and DBSCAN

This project builds an industrial-style user segmentation pipeline for ad targeting personalization.

It creates user-level behavioral features from seven data blocks, trains K-Means for macro business segments, uses DBSCAN for micro-cluster and anomaly detection, profiles each segment by CTR/revenue/retention, and exposes the trained model through both batch prediction and a FastAPI deployment endpoint.

## Project Structure

```text
Kmeans/
  configs/config.yaml                 # All tunable paths, feature columns, and model settings
  data/raw/                           # Input CSV files: users, sessions, ad_events, wallet, transactions, location, device
  data/processed/                     # Engineered features and predictions
  models/                             # Saved model artifacts
  reports/                            # Segment profile and model-selection reports
  src/user_segmentation/
    api.py                            # FastAPI deployment app
    cli.py                            # CLI for train and predict
    config.py                         # YAML config loader
    data_loader.py                    # CSV validation and cleaning
    features.py                       # Feature engineering and preprocessing
    modeling.py                       # K-Means and DBSCAN training
    pipeline.py                       # End-to-end train and predict workflows
    segment_labels.py                 # Business labels and segment profiling
  scripts/generate_sample_data.py     # Local synthetic data generator
  tests/test_features.py              # Basic feature-engineering test
  Dockerfile                          # API container
  docker-compose.yml                  # Local API deployment
```

## Input CSV Schema

Place these files in `data/raw/`:

| File | Required columns |
|---|---|
| `users.csv` | `user_id`, `signup_date`, `country` |
| `sessions.csv` | `session_id`, `user_id`, `duration`, `timestamp` |
| `ad_events.csv` | `user_id`, `impressions`, `clicks` |
| `wallet.csv` | `user_id`, `earnings`, `redeemed` |
| `transactions.csv` | `user_id`, `txn_amount`, `txn_status` |
| `location.csv` | `user_id`, `city`, `country` |
| `device.csv` | `user_id`, `device_type`, `os` |

## Setup

```bash
cd /Users/Dilkhush1/Downloads/projects/Kmeans
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Generate Sample Data

Use this if you want to test the project before connecting real GreedyGame data.

```bash
python scripts/generate_sample_data.py --output-dir data/raw --users 1000
```

## Train the Model

```bash
user-segmentation --config configs/config.yaml train --data-dir data/raw
```

Training writes:

| Output | Purpose |
|---|---|
| `models/segmentation_artifacts.joblib` | Preprocessor, K-Means, DBSCAN, feature list, business label mapping |
| `data/processed/user_features_with_segments.csv` | User-level features plus K-Means and DBSCAN labels |
| `reports/segment_profile.csv` | Segment-level CTR, revenue, retention, engagement, churn risk |
| `reports/kmeans_model_selection.csv` | Elbow/silhouette values for K choices |
| `reports/dbscan_model_selection.csv` | DBSCAN eps, cluster count, noise rate, silhouette |

## Batch Prediction

```bash
user-segmentation --config configs/config.yaml predict \
  --artifact models/segmentation_artifacts.joblib \
  --data-dir data/raw \
  --output data/processed/segment_predictions.csv
```

## API Deployment

Start locally:

```bash
uvicorn user_segmentation.api:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up --build
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "user_id": "user_001",
        "features": {
          "user_age_days": 120,
          "session_count": 12,
          "avg_session_duration": 220,
          "total_session_duration": 2640,
          "active_days": 8,
          "recency_days": 2,
          "pages_per_session": 4,
          "total_impressions": 140,
          "total_clicks": 9,
          "ctr": 0.064,
          "total_earnings": 88,
          "total_redeemed": 25,
          "wallet_balance": 63,
          "redemption_rate": 0.284,
          "total_transactions": 3,
          "total_txn_amount": 450,
          "avg_txn_amount": 150,
          "txn_success_rate": 0.9,
          "engagement_score": 0.72,
          "monetization_score": 0.58,
          "churn_risk_score": 0.18
        }
      }
    ]
  }'
```

## Business Interpretation

K-Means gives clean macro segments for campaign targeting:

- High-value users
- Casual users
- Low engagement / churn risk
- Ad clickers

DBSCAN is used as a second layer to detect unusual dense groups and outliers, such as extreme spenders, very high ad-click users, or niche behavior patterns that K-Means may force into a broad segment.

## Test

```bash
pytest
```

