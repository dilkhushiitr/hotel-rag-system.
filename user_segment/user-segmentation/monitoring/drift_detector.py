"""
Segment Drift Monitoring
─────────────────────────
Detects when the user population has shifted enough to warrant retraining.

Methods
-------
1. Population Stability Index (PSI)  — detects feature distribution drift
2. Segment proportion drift          — detects shifts in cluster sizes
3. KPI degradation check             — compares current vs baseline metrics

PSI Thresholds (industry standard)
  PSI < 0.10  → no significant change
  PSI 0.10–0.25 → moderate change, monitor
  PSI > 0.25  → significant change, RETRAIN
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DRIFT_THRESHOLD

logger = logging.getLogger(__name__)


# ── PSI ────────────────────────────────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """
    Population Stability Index between two distributions.

    Parameters
    ----------
    expected : baseline distribution (1D array)
    actual   : current distribution (1D array)
    buckets  : number of bins

    Returns
    -------
    PSI value (float)
    """
    # Build bucket edges on expected; clip actual to same range
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_pct = np.percentile(expected, breakpoints)
    expected_pct = np.unique(expected_pct)     # remove duplicates

    expected_count = np.histogram(expected, bins=expected_pct)[0]
    actual_count   = np.histogram(actual,   bins=expected_pct)[0]

    # Avoid division by zero / log(0)
    expected_pct_dist = (expected_count / len(expected)).clip(1e-6)
    actual_pct_dist   = (actual_count   / len(actual)).clip(1e-6)

    psi = np.sum((actual_pct_dist - expected_pct_dist) * np.log(actual_pct_dist / expected_pct_dist))
    return round(float(psi), 4)


def compute_feature_psi(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, float]:
    """
    Compute PSI for each feature column.

    Returns dict: {feature_name: psi_value}
    """
    results = {}
    for col in feature_cols:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        psi = compute_psi(baseline_df[col].dropna().values, current_df[col].dropna().values)
        results[col] = psi
        level = "OK" if psi < 0.10 else ("MONITOR" if psi < 0.25 else "⚠️  RETRAIN")
        logger.debug("PSI %-30s = %.4f  [%s]", col, psi, level)

    return results


# ── Segment Drift ─────────────────────────────────────────────────────────────

def segment_proportion_drift(
    baseline_segments: pd.DataFrame,
    current_segments: pd.DataFrame,
    segment_col: str = "segment_label",
) -> pd.DataFrame:
    """
    Compare segment size proportions between baseline and current runs.

    Returns a DataFrame with columns:
      segment_label | baseline_pct | current_pct | abs_drift | flag
    """
    def pct(df):
        counts = df[segment_col].value_counts(normalize=True).reset_index()
        counts.columns = [segment_col, "pct"]
        return counts

    base = pct(baseline_segments).rename(columns={"pct": "baseline_pct"})
    curr = pct(current_segments).rename(columns={"pct": "current_pct"})

    merged = base.merge(curr, on=segment_col, how="outer").fillna(0)
    merged["abs_drift"] = (merged["current_pct"] - merged["baseline_pct"]).abs().round(4)
    merged["flag"]      = merged["abs_drift"].apply(
        lambda x: "⚠️  DRIFT" if x > DRIFT_THRESHOLD else "OK"
    )

    logger.info("Segment proportion drift:\n%s", merged.to_string(index=False))
    return merged


# ── Retrain Decision ──────────────────────────────────────────────────────────

def should_retrain(
    feature_psi: Dict[str, float],
    segment_drift_df: pd.DataFrame,
    psi_threshold: float = 0.25,
    drift_threshold: float = None,
) -> Tuple[bool, str]:
    """
    Decide whether retraining is required.

    Returns
    -------
    (bool, reason_string)
    """
    drift_thr = drift_threshold or DRIFT_THRESHOLD

    # Check PSI
    high_psi_features = {k: v for k, v in feature_psi.items() if v > psi_threshold}
    if high_psi_features:
        reason = f"High PSI detected in features: {high_psi_features}"
        logger.warning("RETRAIN TRIGGERED: %s", reason)
        return True, reason

    # Check segment drift
    flagged = segment_drift_df[segment_drift_df["flag"] != "OK"]
    if not flagged.empty:
        reason = f"Segment proportion drift > {drift_thr} for: {flagged['segment_label'].tolist()}"
        logger.warning("RETRAIN TRIGGERED: %s", reason)
        return True, reason

    logger.info("No retraining needed. Distributions are stable.")
    return False, "Stable"


# ── Full Monitoring Run ────────────────────────────────────────────────────────

def run_monitoring(
    baseline_path: Optional[str] = None,
    current_path: Optional[str] = None,
) -> Dict:
    """
    Compare baseline vs current user_segments CSVs and report drift.

    Defaults to reading from data/processed/ directory.
    """
    baseline_path = baseline_path or str(DATA_PROCESSED / "user_segments_baseline.csv")
    current_path  = current_path  or str(DATA_PROCESSED / "user_segments.csv")

    if not Path(baseline_path).exists():
        logger.warning("Baseline not found at %s. Copy current to baseline first.", baseline_path)
        return {"status": "no_baseline"}

    baseline = pd.read_csv(baseline_path)
    current  = pd.read_csv(current_path)

    score_cols = ["engagement_score", "monetization_score", "churn_risk_score"]

    feature_psi    = compute_feature_psi(baseline, current, score_cols)
    segment_drift  = segment_proportion_drift(baseline, current)
    needs_retrain, reason = should_retrain(feature_psi, segment_drift)

    report = {
        "run_at":          datetime.utcnow().isoformat(),
        "feature_psi":     feature_psi,
        "segment_drift":   segment_drift.to_dict(orient="records"),
        "needs_retrain":   needs_retrain,
        "reason":          reason,
    }

    return report


if __name__ == "__main__":
    report = run_monitoring()
    print("\n=== Drift Monitoring Report ===")
    print(f"Run at: {report.get('run_at')}")
    print(f"Needs retrain: {report.get('needs_retrain')}  |  Reason: {report.get('reason')}")
    print("\nFeature PSI:")
    for feat, psi in (report.get("feature_psi") or {}).items():
        print(f"  {feat:30s} {psi:.4f}")
