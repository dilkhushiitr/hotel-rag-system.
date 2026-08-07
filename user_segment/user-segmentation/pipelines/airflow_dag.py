"""
pipelines/airflow_dag.py

Apache Airflow DAG for weekly user segmentation pipeline.
Schedule: every Sunday at 02:00 UTC

Tasks:
  t1_check_db     → t2_generate_data (if sample mode) → t3_feature_eng
  → t4_kmeans     → t5_dbscan        → t6_profile
  → t7_write_db   → t8_notify
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner":            "ml-team",
    "depends_on_past":  False,
    "start_date":       days_ago(1),
    "email":            ["ml-alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# ── DAG definition ─────────────────────────────────────────────────────────────
dag = DAG(
    dag_id="user_segmentation_weekly",
    description="Weekly user segmentation: K-Means + DBSCAN → ad targeting",
    default_args=default_args,
    schedule_interval="0 2 * * 0",   # every Sunday 02:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["ml", "segmentation", "ads"],
)

# ── Task functions ─────────────────────────────────────────────────────────────
def task_check_db(**context):
    import sys, os
    sys.path.insert(0, "/opt/airflow")
    from src.data.db import test_connection
    ok = test_connection()
    if not ok:
        raise RuntimeError("DB connection failed")
    return "DB OK"


def task_extract_features(**context):
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.data.sql_queries import extract_all_features
    df = extract_all_features(lookback_days=30)
    # Push to XCom (small summary only)
    context["ti"].xcom_push(key="n_users", value=len(df))
    # Save to disk for next tasks
    df.to_csv("/tmp/raw_features.csv", index=False)
    return f"Extracted {len(df)} users"


def task_feature_engineering(**context):
    import sys, pandas as pd, pickle
    sys.path.insert(0, "/opt/airflow")
    from src.features.feature_engineering import build_feature_matrix
    raw_df = pd.read_csv("/tmp/raw_features.csv")
    features_df, enriched_df, scaler = build_feature_matrix(raw_df, fit_scaler=True)
    features_df.to_csv("/tmp/features_scaled.csv", index=False)
    enriched_df.to_csv("/tmp/enriched.csv",        index=False)
    with open("/tmp/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return f"Feature matrix: {features_df.shape}"


def task_fit_kmeans(**context):
    import sys, numpy as np, pandas as pd
    sys.path.insert(0, "/opt/airflow")
    from src.models.kmeans_model import KMeansSegmenter
    features_df  = pd.read_csv("/tmp/features_scaled.csv")
    X            = features_df.drop(columns=["user_id"], errors="ignore").values
    segmenter    = KMeansSegmenter(n_clusters=5)
    segmenter.fit(X)
    segmenter.save("/tmp/kmeans_model.pkl")
    labels = segmenter.predict(X).tolist()
    import json
    with open("/tmp/kmeans_labels.json", "w") as f:
        json.dump(labels, f)
    return "K-Means fitted"


def task_fit_dbscan(**context):
    import sys, numpy as np, pandas as pd, json
    sys.path.insert(0, "/opt/airflow")
    from src.models.dbscan_model import DBSCANSegmenter
    features_df   = pd.read_csv("/tmp/features_scaled.csv")
    X             = features_df.drop(columns=["user_id"], errors="ignore").values
    db_segmenter  = DBSCANSegmenter(eps=0.5, min_samples=5)
    db_segmenter.fit(X)
    db_segmenter.save("/tmp/dbscan_model.pkl")
    with open("/tmp/dbscan_labels.json", "w") as f:
        json.dump(db_segmenter.labels_.tolist(), f)
    return f"DBSCAN: {db_segmenter.n_clusters_} clusters, {db_segmenter.n_noise_} noise"


def task_profile_and_write(**context):
    import sys, numpy as np, pandas as pd, json
    sys.path.insert(0, "/opt/airflow")
    from src.analysis.segment_profiler import SegmentProfiler

    enriched_df   = pd.read_csv("/tmp/enriched.csv")
    with open("/tmp/kmeans_labels.json") as f:
        kmeans_labels = np.array(json.load(f))
    with open("/tmp/dbscan_labels.json") as f:
        dbscan_labels = np.array(json.load(f))

    profiler = SegmentProfiler()
    user_segments = profiler.profile(enriched_df, kmeans_labels, dbscan_labels)
    profiler.write_to_db(user_segments)
    return f"Wrote {len(user_segments)} rows to DB"


def task_copy_models(**context):
    import shutil, os
    os.makedirs("/opt/airflow/data/models", exist_ok=True)
    for fname in ["kmeans_model.pkl", "dbscan_model.pkl", "scaler.pkl"]:
        src = f"/tmp/{fname}"
        dst = f"/opt/airflow/data/models/{fname}"
        if os.path.exists(src):
            shutil.copy(src, dst)
    return "Models copied"


def task_notify(**context):
    n_users = context["ti"].xcom_pull(key="n_users", task_ids="t2_extract_features")
    msg     = f"Segmentation pipeline complete. Processed {n_users} users."
    print(msg)
    # Add Slack / email notification here
    return msg


# ── Tasks ──────────────────────────────────────────────────────────────────────
with dag:
    t1 = PythonOperator(task_id="t1_check_db",         python_callable=task_check_db)
    t2 = PythonOperator(task_id="t2_extract_features",  python_callable=task_extract_features)
    t3 = PythonOperator(task_id="t3_feature_engineering", python_callable=task_feature_engineering)
    t4 = PythonOperator(task_id="t4_fit_kmeans",        python_callable=task_fit_kmeans)
    t5 = PythonOperator(task_id="t5_fit_dbscan",        python_callable=task_fit_dbscan)
    t6 = PythonOperator(task_id="t6_profile_and_write", python_callable=task_profile_and_write)
    t7 = PythonOperator(task_id="t7_copy_models",       python_callable=task_copy_models)
    t8 = PythonOperator(task_id="t8_notify",            python_callable=task_notify)

    t1 >> t2 >> t3 >> [t4, t5] >> t6 >> t7 >> t8
