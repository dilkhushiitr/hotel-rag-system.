"""
Unit Tests
──────────
Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

# ── Feature engineering tests ──────────────────────────────────────────────────

class TestEngagementFeatures:
    def _make_sessions(self):
        return pd.DataFrame({
            "session_id":       ["S1", "S2", "S3"],
            "user_id":          ["U1", "U1", "U2"],
            "session_start":    pd.to_datetime(["2024-01-10", "2024-01-15", "2024-01-20"]),
            "duration_seconds": [120, 300, 60],
            "pages_viewed":     [5, 10, 3],
            "platform":         ["android", "ios", "web"],
        })

    def test_session_count(self):
        from src.features.engineering import build_engagement_features
        sessions = self._make_sessions()
        result = build_engagement_features(sessions, end_date=datetime(2024, 1, 31))
        u1_row = result[result["user_id"] == "U1"]
        assert u1_row["session_count"].values[0] == 2

    def test_avg_duration(self):
        from src.features.engineering import build_engagement_features
        sessions = self._make_sessions()
        result = build_engagement_features(sessions, end_date=datetime(2024, 1, 31))
        u1_row = result[result["user_id"] == "U1"]
        assert abs(u1_row["avg_session_duration"].values[0] - 210.0) < 0.01

    def test_recency_days_computed(self):
        from src.features.engineering import build_engagement_features
        sessions = self._make_sessions()
        result = build_engagement_features(sessions, end_date=datetime(2024, 1, 31))
        u1_row = result[result["user_id"] == "U1"]
        # Last session of U1 is Jan 15; end_date is Jan 31 → 16 days
        assert u1_row["recency_days"].values[0] == 16


class TestAdFeatures:
    def _make_ads(self):
        return pd.DataFrame({
            "event_id":    ["A1", "A2", "A3", "A4"],
            "user_id":     ["U1", "U1", "U1", "U2"],
            "ad_id":       ["AD1", "AD1", "AD2", "AD3"],
            "event_type":  ["impression", "impression", "click", "impression"],
            "timestamp":   pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-10", "2024-01-15"]),
            "ad_category": ["gaming", "finance", "gaming", "ecom"],
        })

    def test_impression_count(self):
        from src.features.engineering import build_ad_features
        ads = self._make_ads()
        result = build_ad_features(ads, end_date=datetime(2024, 1, 31))
        u1 = result[result["user_id"] == "U1"]
        assert u1["total_impressions"].values[0] == 2

    def test_ctr_calculation(self):
        from src.features.engineering import build_ad_features
        ads = self._make_ads()
        result = build_ad_features(ads, end_date=datetime(2024, 1, 31))
        u1 = result[result["user_id"] == "U1"]
        # 1 click / 2 impressions = 0.5
        assert abs(u1["ctr"].values[0] - 0.5) < 0.001


class TestDerivedScores:
    def _make_feature_df(self):
        return pd.DataFrame({
            "user_id":          ["U1", "U2", "U3"],
            "session_count":    [10, 1, 5],
            "active_days":      [20, 2, 10],
            "recency_days":     [1, 25, 10],
            "total_earnings":   [100, 5, 50],
            "total_redeemed":   [50, 0, 25],
            "ctr":              [0.1, 0.01, 0.05],
        })

    def test_scores_in_range(self):
        from src.features.engineering import build_derived_scores
        df = self._make_feature_df()
        result = build_derived_scores(df)
        for col in ["engagement_score", "monetization_score", "churn_risk_score"]:
            assert result[col].between(0, 100).all(), f"{col} out of [0,100]"

    def test_high_value_user_has_high_engagement(self):
        from src.features.engineering import build_derived_scores
        df = self._make_feature_df()
        result = build_derived_scores(df)
        u1 = result[result["user_id"] == "U1"]["engagement_score"].values[0]
        u2 = result[result["user_id"] == "U2"]["engagement_score"].values[0]
        assert u1 > u2, "U1 should have higher engagement than U2"


# ── Preprocessing tests ────────────────────────────────────────────────────────

class TestPreprocessing:
    def _df(self):
        return pd.DataFrame({
            "user_id": ["U1", "U2", "U3", "U4"],
            "feat_a":  [1.0, 2.0, np.nan, 100.0],  # one NaN, one outlier
            "feat_b":  [0.5, 0.3, 0.8, 0.1],
        })

    def test_impute_fills_na(self):
        from src.data.preprocessing import impute_missing
        result = impute_missing(self._df())
        assert result["feat_a"].isna().sum() == 0

    def test_cap_outliers(self):
        from src.data.preprocessing import cap_outliers
        df = pd.DataFrame({"feat": [1, 2, 3, 1000]})
        result = cap_outliers(df, cols=["feat"], factor=1.5)
        assert result["feat"].max() < 1000

    def test_log_transform_non_negative(self):
        from src.data.preprocessing import log_transform
        df = pd.DataFrame({"total_earnings": [0, 10, 100, 1000]})
        result = log_transform(df)
        assert (result["total_earnings"] >= 0).all()


# ── Clustering tests ───────────────────────────────────────────────────────────

class TestKMeans:
    def _X(self):
        np.random.seed(42)
        return np.vstack([
            np.random.randn(50, 3) + [0, 0, 0],
            np.random.randn(50, 3) + [10, 10, 10],
            np.random.randn(50, 3) + [20, 0, 20],
        ])

    def test_train_returns_model(self):
        from src.models.kmeans_model import train_kmeans
        model = train_kmeans(self._X(), k=3, save=False)
        assert hasattr(model, "labels_")
        assert model.n_clusters == 3

    def test_predict_length(self):
        from src.models.kmeans_model import train_kmeans, predict_clusters
        model = train_kmeans(self._X(), k=3, save=False)
        labels = predict_clusters(model, self._X())
        assert len(labels) == 150


class TestDBSCAN:
    def _X(self):
        np.random.seed(42)
        core = np.random.randn(100, 3)
        noise = np.array([[50, 50, 50], [-50, -50, -50]])  # outliers
        return np.vstack([core, noise])

    def test_detects_noise(self):
        from src.models.dbscan_model import train_dbscan
        _, labels = train_dbscan(self._X(), eps=1.0, min_samples=5, save=False)
        assert -1 in labels, "DBSCAN should flag noise points"

    def test_noise_count_reasonable(self):
        from src.models.dbscan_model import train_dbscan
        X = self._X()
        _, labels = train_dbscan(X, eps=1.0, min_samples=5, save=False)
        noise_pct = (labels == -1).mean()
        assert noise_pct < 0.5, "More than 50% noise — eps is too small?"


# ── Segmentation tests ─────────────────────────────────────────────────────────

class TestSegmentation:
    def _setup(self):
        feature_df = pd.DataFrame({
            "user_id":          [f"U{i}" for i in range(10)],
            "engagement_score":  [80, 20, 50, 90, 10, 60, 30, 70, 40, 5],
            "monetization_score":[90, 10, 40, 85, 5, 55, 25, 65, 35, 2],
            "churn_risk_score":  [5, 90, 50, 10, 95, 40, 70, 20, 60, 99],
            "ctr":               [0.1, 0.01, 0.05, 0.12, 0.005, 0.07, 0.03, 0.09, 0.04, 0.001],
        })
        kmeans_labels = np.array([0, 3, 2, 0, 4, 1, 3, 1, 2, 4])
        dbscan_labels = np.array([0, 0, 0, 0, -1, 0, 0, 0, 0, -1])
        return feature_df, kmeans_labels, dbscan_labels

    def test_segment_table_has_all_users(self):
        from src.models.segmentation import build_user_segments
        feat, km, db = self._setup()
        result = build_user_segments(feat, km, db)
        assert len(result) == 10

    def test_anomalies_correctly_labelled(self):
        from src.models.segmentation import build_user_segments
        feat, km, db = self._setup()
        result = build_user_segments(feat, km, db)
        anomaly_rows = result[result["is_anomaly"] == 1]
        assert all(anomaly_rows["segment_label"] == "Anomaly / Fraud Suspect")
