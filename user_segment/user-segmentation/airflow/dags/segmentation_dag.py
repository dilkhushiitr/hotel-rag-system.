"""
Airflow DAG: Weekly User Segmentation Pipeline
───────────────────────────────────────────────
Schedule: Every Sunday at 02:00 UTC
Tasks:
  1. generate_features     — build feature matrix from DB
  2. run_kmeans            — train / predict clusters
  3. run_dbscan            — anomaly detection
  4. build_segments        — assemble user_segments table
  5. write_to_db           — persist to PostgreSQL
  6. run_monitoring        — drift detection vs last week
  7. send_slack_alert      — notify team if drift detected (optional)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# ── Default args ───────────────────────────────────────────────────────────────
default_args = {
    "owner":            "ml-team",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 7),
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# ── DAG definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id          = "user_segmentation_weekly",
    default_args    = default_args,
    description     = "Weekly user segmentation & ad personalisation pipeline",
    schedule_interval = "0 2 * * 0",   # Every Sunday 02:00 UTC
    catchup         = False,
    tags            = ["ml", "segmentation", "ads"],
) as dag:

    # ── Task 1: Backup previous segments ──────────────────────────────────────
    backup_segments = BashOperator(
        task_id     = "backup_previous_segments",
        bash_command = (
            "cp /app/data/processed/user_segments.csv "
            "/app/data/processed/user_segments_baseline.csv 2>/dev/null || true"
        ),
    )

    # ── Task 2: Generate sample data (dev only — skip in prod) ───────────────
    generate_data = BashOperator(
        task_id     = "generate_sample_data",
        bash_command = "python /app/data/sample/generate_sample_data.py",
    )

    # ── Task 3: Run full pipeline ─────────────────────────────────────────────
    def _run_pipeline(**context):
        from src.pipeline.batch_pipeline import run_pipeline
        segments = run_pipeline(
            source    = "csv",     # Change to "db" in production
            fit_scaler = True,
            find_k    = False,
            write_db  = False,    # Change to True in production
            plots_dir = "/app/reports/plots",
            save_csv  = True,
        )
        context["task_instance"].xcom_push(key="n_users", value=len(segments))
        return len(segments)

    run_segmentation = PythonOperator(
        task_id         = "run_segmentation_pipeline",
        python_callable = _run_pipeline,
    )

    # ── Task 4: Run drift monitoring ─────────────────────────────────────────
    def _run_monitoring(**context):
        from monitoring.drift_detector import run_monitoring
        report = run_monitoring()
        needs_retrain = report.get("needs_retrain", False)
        reason        = report.get("reason", "")
        context["task_instance"].xcom_push(key="needs_retrain", value=needs_retrain)
        context["task_instance"].xcom_push(key="drift_reason",  value=reason)

        if needs_retrain:
            import logging
            logging.getLogger(__name__).warning(
                "DRIFT DETECTED — consider retraining. Reason: %s", reason
            )
        return report

    monitor_drift = PythonOperator(
        task_id         = "monitor_drift",
        python_callable = _run_monitoring,
    )

    # ── Task 5: Log summary ───────────────────────────────────────────────────
    def _log_summary(**context):
        ti = context["task_instance"]
        n_users      = ti.xcom_pull(key="n_users",      task_ids="run_segmentation_pipeline")
        needs_retrain = ti.xcom_pull(key="needs_retrain", task_ids="monitor_drift")
        reason        = ti.xcom_pull(key="drift_reason",  task_ids="monitor_drift")

        summary = (
            f"\n{'='*50}\n"
            f"  Segmentation Pipeline Summary\n"
            f"  Run date   : {datetime.utcnow().date()}\n"
            f"  Users      : {n_users:,}\n"
            f"  Drift?     : {needs_retrain}  ({reason})\n"
            f"{'='*50}"
        )
        print(summary)

    log_summary = PythonOperator(
        task_id         = "log_summary",
        python_callable = _log_summary,
    )

    # ── DAG wiring ────────────────────────────────────────────────────────────
    backup_segments >> generate_data >> run_segmentation >> monitor_drift >> log_summary
