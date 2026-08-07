# ...existing code...
"""
Single-file production-ready fraud system (all modules combined).
- Contains: config loader, RawDataLoader, FeatureBuilder (and sub-feature classes),
  RuleEngine, IsolationForestModel, Trainer, Scorer, ModelRegistry,
  RiskAggregator, Thresholding, DriftDetector, MetricsLogger, RetrainingPipeline,
  FastAPI app, and CLI (train/score/serve).
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
import os
import shutil
import joblib
import yaml
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve

# -----------------------
# Defaults & logging
# -----------------------
DEFAULT_CONFIG = {
    "model": {"contamination": 0.02, "n_estimators": 300, "random_state": 42},
    "risk": {"rule_weight": 0.5, "anomaly_weight": 0.5, "block_threshold": 0.85, "review_threshold": 0.65},
    "data": {"feature_table": "fraud_user_features_daily"},
    "paths": {"model_dir": "models", "artifact_name": "model.pkl"},
}

LOG = logging.getLogger("fraud_prod")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEED = 42
np.random.seed(SEED)


def load_config(path: Optional[str]) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        merged = DEFAULT_CONFIG.copy()
        # shallow merge dicts
        for k, v in cfg.items():
            if isinstance(v, dict):
                merged[k] = {**merged.get(k, {}), **v}
            else:
                merged[k] = v
        return merged
    return DEFAULT_CONFIG.copy()


# -----------------------
# Data loading
# -----------------------
class RawDataLoader:
    def load_users(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["signup_time"], low_memory=False)

    def load_transactions(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["transaction_time"], low_memory=False)

    def load_postbacks(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["click_time", "postback_time"], low_memory=False)

    def load_devices(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False)

    def load_ip_logs(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False)


# -----------------------
# Feature transforms
# -----------------------
class UserFeatures:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "signup_time" in df.columns:
            df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
            df["account_age_days"] = (pd.Timestamp("now") - df["signup_time"]).dt.days.fillna(0).astype(int)
        else:
            df["account_age_days"] = 0
        df["referral_count"] = pd.to_numeric(df.get("referral_count", 0)).fillna(0).astype(int)
        df["referral_ratio"] = df["referral_count"] / (df["account_age_days"] + 1)
        return df


class TransactionFeatures:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "transaction_time" in df.columns:
            df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")
            df = df.sort_values(["user_id", "transaction_time"])
            df["tx_time_diff"] = df.groupby("user_id")["transaction_time"].diff().dt.total_seconds().fillna(0)
        else:
            df["tx_time_diff"] = 0
        if "transaction_amount" in df.columns:
            df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce").fillna(0)
            df["avg_tx_amount"] = df.groupby("user_id")["transaction_amount"].transform("mean").fillna(0)
        else:
            df["avg_tx_amount"] = 0
        return df


class IPFeatures:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "signup_ip" in df.columns:
            df["users_per_signup_ip"] = df.groupby("signup_ip")["user_id"].transform("count").fillna(0)
        else:
            df["users_per_signup_ip"] = 0
        df["unique_ips_per_user"] = df.groupby("user_id")["signup_ip"].transform("nunique").fillna(0)
        return df


class DeviceFeatures:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "fingerprint_id" in df.columns:
            df["devices_per_user"] = df.groupby("user_id")["fingerprint_id"].transform("nunique").fillna(0)
        else:
            df["devices_per_user"] = 0
        return df


class RuleFeatures:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in ["rule_duplicate_ip", "rule_low_conversion", "rule_vpn"]:
            if c not in df.columns:
                df[c] = 0
        df["rule_unique_types"] = df[["rule_duplicate_ip", "rule_low_conversion", "rule_vpn"]].sum(axis=1)
        return df


class FeatureBuilder:
    def __init__(self):
        self.user_features = UserFeatures()
        self.tx_features = TransactionFeatures()
        self.ip_features = IPFeatures()
        self.device_features = DeviceFeatures()
        self.rule_features = RuleFeatures()

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.user_features.transform(df)
        df = self.tx_features.transform(df)
        df = self.ip_features.transform(df)
        df = self.device_features.transform(df)
        df = self.rule_features.transform(df)
        return df


# -----------------------
# Rules
# -----------------------
RULES = {
    "DUPLICATE_IP": {"threshold": 2},
    "LOW_CONVERSION_TIME": {"threshold": 60},
    "VPN_FLAG": {"threshold": 1},
    "POSTBACK_TIME_FRAUD": {"threshold": 600},
}


class RuleEngine:
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        defaults = {"users_per_signup_ip": 0, "reward_latency": 1e9, "vpn_flag": 0}
        for k, v in defaults.items():
            if k not in df.columns:
                df[k] = v
        df["rule_duplicate_ip"] = (df["users_per_signup_ip"] > RULES["DUPLICATE_IP"]["threshold"]).astype(int)
        df["rule_low_conversion"] = (df["reward_latency"] < RULES["LOW_CONVERSION_TIME"]["threshold"]).astype(int)
        df["rule_vpn"] = (df["vpn_flag"] == RULES["VPN_FLAG"]["threshold"]).astype(int)
        df["rule_total_count"] = df[["rule_duplicate_ip", "rule_low_conversion", "rule_vpn"]].sum(axis=1)
        df["hard_block"] = (df["rule_duplicate_ip"] == 1).astype(int)
        # add short reason for first triggered rule for explainability
        df["rule_reason"] = ""
        conds = [
            df["rule_duplicate_ip"] == 1,
            df["rule_low_conversion"] == 1,
            df["rule_vpn"] == 1,
        ]
        reasons = ["DUPLICATE_IP", "LOW_CONVERSION_TIME", "VPN_FLAG"]
        df["rule_reason"] = np.select(conds, reasons, default="")
        # also set generic rule_flagged for compatibility
        df["rule_flagged"] = (df["rule_total_count"] > 0).astype(int)
        return df


# -----------------------
# Composite engineered features
# -----------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    inputs = [
        "click_entropy",
        "touch_events_per_min",
        "time_btw_actions_sec",
        "account_age_days",
        "referral_count",
        "kyc_verified",
        "txn_velocity",
        "earn_redeem_ratio",
        "txn_failure_rate",
        "device_count",
        "is_rooted",
        "is_emulator",
        "geo_ip_mismatch",
        "ip_change_count_7d",
        "avg_txn_amount",
    ]
    for c in inputs:
        if c not in df.columns:
            df[c] = 0
    df["bot_behavior_index"] = (
        (1 - df["click_entropy"].clip(0, 1)) * 0.4
        + (df["touch_events_per_min"].clip(0, 500) / 500) * 0.35
        + (1 - df["time_btw_actions_sec"].clip(0, 300) / 300) * 0.25
    )
    df["account_age_norm"] = df["account_age_days"].clip(0, 365) / 365
    referral_flag = (df["referral_count"] > 10).astype(int)
    df["account_trust_score"] = df["kyc_verified"] * 0.4 + df["account_age_norm"] * 0.4 + (1 - referral_flag) * 0.2
    df["txn_velocity_norm"] = df["txn_velocity"].clip(0, 70) / 70
    df["txn_suspicion_score"] = df["txn_velocity_norm"] * 0.4 + df["earn_redeem_ratio"] * 0.35 + df["txn_failure_rate"] * 0.25
    df["device_count_norm"] = df["device_count"].clip(0, 20) / 20
    df["ip_change_norm"] = df["ip_change_count_7d"].clip(0, 30) / 30
    df["device_risk_score"] = (
        df["is_rooted"] * 0.25
        + df["is_emulator"] * 0.25
        + df["device_count_norm"] * 0.25
        + df["geo_ip_mismatch"] * 0.15
        + df["ip_change_norm"] * 0.10
    )
    df["manual_fraud_score"] = (
        df["bot_behavior_index"] * 0.35 + df["device_risk_score"] * 0.30 + df["txn_suspicion_score"] * 0.25 + (1 - df["account_trust_score"]) * 0.10
    )
    df["velocity_x_redeem"] = df["txn_velocity"] * df["earn_redeem_ratio"]
    df["bot_x_device_risk"] = df["bot_behavior_index"] * df["device_risk_score"]
    df["unverified_highvalue"] = ((df["kyc_verified"] == 0) & (df["avg_txn_amount"] > 100)).astype(int)
    return df


# -----------------------
# Isolation Forest wrapper
# -----------------------
class IsolationForestModel:
    def __init__(self, contamination=0.02, n_estimators=300, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.score_min = 0.0
        self.score_max = 1.0

    def train(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        raw = -self.model.decision_function(Xs)
        self.score_min = float(raw.min())
        self.score_max = float(raw.max())
        return raw

    def score_raw(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return -self.model.decision_function(Xs)

    def score_norm(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.score_raw(X)
        norm = (raw - self.score_min) / (self.score_max - self.score_min + 1e-12)
        return np.clip(norm, 0.0, 1.0)

    def save(self, path: str) -> None:
        obj = {"model": self.model, "scaler": self.scaler, "contamination": self.contamination, "n_estimators": self.n_estimators, "score_min": self.score_min, "score_max": self.score_max}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        LOG.info("Saved model to %s", path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.contamination = data.get("contamination", self.contamination)
        self.n_estimators = data.get("n_estimators", self.n_estimators)
        self.score_min = data.get("score_min", self.score_min)
        self.score_max = data.get("score_max", self.score_max)
        LOG.info("Loaded model from %s", path)


# -----------------------
# Trainer / Scorer / Registry
# -----------------------
class Trainer:
    def __init__(self, config: dict):
        mc = config.get("model", {})
        self.model = IsolationForestModel(contamination=mc.get("contamination", 0.02), n_estimators=mc.get("n_estimators", 300), random_state=mc.get("random_state", 42))

    def train(self, df: pd.DataFrame, feature_cols: List[str], exclude_rule_flagged: bool = True, auto_tune_threshold: bool = True) -> Tuple[IsolationForestModel, float]:
        df = df.copy()
        if "rule_flagged" not in df.columns:
            df = RuleEngine().apply(df)
        df = engineer_features(df)
        train_df = df[df["rule_flagged"] == 0] if exclude_rule_flagged and "rule_flagged" in df.columns else df
        if train_df.empty:
            LOG.warning("No rule-clean rows found; training on full dataset")
            train_df = df
        X = train_df[feature_cols].fillna(0)
        raw = self.model.train(X)
        norm = (raw - self.model.score_min) / (self.model.score_max - self.model.score_min + 1e-12)
        threshold = None
        if auto_tune_threshold and "true_label" in train_df.columns:
            precision, recall, thresholds = precision_recall_curve(train_df["true_label"], norm)
            if len(thresholds) > 0:
                f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
                best = int(np.argmax(f1))
                threshold = float(thresholds[best])
                LOG.info("Auto-tuned threshold=%.4f", threshold)
        if threshold is None:
            threshold = float(np.quantile(norm, 1 - self.model.contamination))
        LOG.info("Trainer complete. threshold=%.4f", threshold)
        return self.model, threshold


class Scorer:
    def __init__(self, model: IsolationForestModel, feature_cols: List[str], threshold: float):
        self.model = model
        self.feature_cols = feature_cols
        self.threshold = threshold

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = engineer_features(df)
        df = RuleEngine().apply(df)
        X = df[self.feature_cols].fillna(0)
        df["anomaly_score"] = self.model.score_norm(X)
        df["if_score_flagged"] = (df["anomaly_score"] >= self.threshold).astype(int)
        df["final_fraud_flag"] = ((df["rule_flagged"] == 1) | (df["if_score_flagged"] == 1)).astype(int)
        conditions = [
            (df["rule_flagged"] == 1) & (df["if_score_flagged"] == 1),
            (df["rule_flagged"] == 1) & (df["if_score_flagged"] == 0),
            (df["rule_flagged"] == 0) & (df["if_score_flagged"] == 1),
        ]
        choices = ["rule_and_model", "rule_only", "model_only"]
        df["detection_source"] = np.select(conditions, choices, default="not_flagged")
        return df

    def score_record(self, record: Dict) -> Dict:
        df = pd.DataFrame([record])
        out = self.score(df).iloc[0]
        confidence = "low"
        if out["detection_source"] == "rule_and_model":
            confidence = "high"
        elif out["detection_source"] == "rule_only":
            confidence = "high"
        elif out["detection_source"] == "model_only":
            confidence = "medium" if out["anomaly_score"] < 0.8 else "high"
        else:
            confidence = "low" if out["anomaly_score"] > 0.4 else "high"
        return {
            "user_id": out.get("user_id"),
            "anomaly_score": float(round(out["anomaly_score"], 4)),
            "is_fraud": bool(out["final_fraud_flag"]),
            "detection_source": out["detection_source"],
            "rule_triggered": out.get("rule_reason", None),
            "confidence": confidence,
        }


class ModelRegistry:
    def register(self, model_path: str, version: int) -> str:
        dest = f"models/model_v{version}.pkl"
        Path("models").mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, dest)
        LOG.info("Registered model %s -> %s", model_path, dest)
        return dest


# -----------------------
# Risk / thresholding / monitoring
# -----------------------
class RiskAggregator:
    def combine(self, df: pd.DataFrame, rule_weight: float = 0.5, anomaly_weight: float = 0.5) -> pd.DataFrame:
        df = df.copy()
        df["rule_rank"] = df["rule_total_count"].rank(pct=True)
        df["anomaly_rank"] = df["anomaly_score"].rank(pct=True)
        df["final_risk_score"] = rule_weight * df["rule_rank"] + anomaly_weight * df["anomaly_rank"]
        return df


class Thresholding:
    def apply(self, df: pd.DataFrame, block_threshold: float = 0.85, review_threshold: float = 0.65) -> pd.DataFrame:
        df = df.copy()
        df["decision"] = "ALLOW"
        df.loc[df["final_risk_score"] > review_threshold, "decision"] = "REVIEW"
        df.loc[df["final_risk_score"] > block_threshold, "decision"] = "BLOCK"
        df.loc[df.get("hard_block", 0) == 1, "decision"] = "BLOCK"
        return df


class DriftDetector:
    def detect(self, baseline_scores: pd.Series, current_scores: pd.Series) -> float:
        baseline_mean = baseline_scores.mean()
        current_mean = current_scores.mean()
        return abs(current_mean - baseline_mean)


class MetricsLogger:
    def log(self, precision: float, recall: float, fraud_captured: float) -> None:
        LOG.info("Precision: %.4f Recall: %.4f FraudCaptured: %.4f", precision, recall, fraud_captured)


class RetrainingPipeline:
    def __init__(self, config: dict):
        self.config = config

    def retrain(self, df: pd.DataFrame, feature_cols: List[str]) -> IsolationForestModel:
        trainer = Trainer(self.config)
        model, threshold = trainer.train(df, feature_cols)
        model.save(os.path.join(self.config["paths"]["model_dir"], self.config["paths"]["artifact_name"]))
        meta = {"feature_cols": feature_cols, "threshold": threshold}
        joblib.dump(meta, os.path.join(self.config["paths"]["model_dir"], self.config["paths"]["artifact_name"] + ".meta.pkl"))
        LOG.info("Retrained and saved model + meta")
        return model


# -----------------------
# API + CLI
# -----------------------
app = FastAPI(title="FraudProdAPI")
_GLOBAL = {"scorer": None, "feature_cols": None, "threshold": None, "model": None}


@app.on_event("startup")
def startup_event():
    cfg = load_config(None)
    model_path = os.path.join(cfg["paths"]["model_dir"], cfg["paths"]["artifact_name"])
    default_features = [
        "account_age_days",
        "avg_tx_amount",
        "txn_velocity",
        "earn_redeem_ratio",
        "txn_failure_rate",
        "device_count",
        "bot_behavior_index",
        "device_risk_score",
        "txn_suspicion_score",
        "rule_total_count",
    ]
    _GLOBAL["feature_cols"] = default_features
    if Path(model_path).exists():
        m = IsolationForestModel()
        m.load(model_path)
        meta_path = str(Path(model_path).with_suffix(".meta.pkl"))
        if Path(meta_path).exists():
            meta = joblib.load(meta_path)
            feature_cols = meta.get("feature_cols", default_features)
            threshold = meta.get("threshold", float(np.quantile(np.linspace(0, 1, 1000), 1 - m.contamination)))
        else:
            feature_cols = default_features
            threshold = float(np.quantile(np.linspace(0, 1, 1000), 1 - m.contamination))
        _GLOBAL["model"] = m
        _GLOBAL["threshold"] = threshold
        _GLOBAL["scorer"] = Scorer(m, feature_cols, threshold)
        LOG.info("Loaded model for API")
    else:
        LOG.warning("No model found at %s", model_path)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _GLOBAL["model"] is not None}


@app.post("/score")
def score_endpoint(payload: Dict):
    if _GLOBAL.get("scorer") is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if "user_id" not in payload:
        raise HTTPException(status_code=400, detail="user_id required")
    res = _GLOBAL["scorer"].score_record(payload)
    return res


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def main():
    parser = argparse.ArgumentParser(description="Fraud production single-file")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--input", required=True)
    t.add_argument("--config", default=None)
    t.add_argument("--model-path", default=os.path.join(DEFAULT_CONFIG["paths"]["model_dir"], DEFAULT_CONFIG["paths"]["artifact_name"]))
    t.add_argument("--features", nargs="+", required=False)

    s = sub.add_parser("score")
    s.add_argument("--input", required=True)
    s.add_argument("--model-path", required=True)
    s.add_argument("--output", required=False)
    s.add_argument("--config", default=None)

    serve = sub.add_parser("serve")
    serve.add_argument("--model-path", default=os.path.join(DEFAULT_CONFIG["paths"]["model_dir"], DEFAULT_CONFIG["paths"]["artifact_name"]))
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--config", default=None)

    args = parser.parse_args()
    if args.cmd == "train":
        cfg = load_config(args.config)
        df = _load_csv(args.input)
        builder = FeatureBuilder()
        df = builder.build(df)
        rule_engine = RuleEngine()
        df = rule_engine.apply(df)
        feature_cols = args.features or [
            "account_age_days",
            "avg_tx_amount",
            "txn_velocity",
            "earn_redeem_ratio",
            "txn_failure_rate",
            "device_count",
            "bot_behavior_index",
            "device_risk_score",
            "txn_suspicion_score",
            "rule_total_count",
        ]
        trainer = Trainer(cfg)
        model, threshold = trainer.train(df, feature_cols, exclude_rule_flagged=True, auto_tune_threshold=True)
        model.save(args.model_path)
        meta = {"feature_cols": feature_cols, "threshold": threshold}
        joblib.dump(meta, str(Path(args.model_path).with_suffix(".meta.pkl")))
        LOG.info("Saved model and metadata to %s", args.model_path)

    elif args.cmd == "score":
        cfg = load_config(args.config)
        df = _load_csv(args.input)
        builder = FeatureBuilder()
        df = builder.build(df)
        rule_engine = RuleEngine()
        df = rule_engine.apply(df)
        m = IsolationForestModel()
        m.load(args.model_path)
        meta_path = str(Path(args.model_path).with_suffix(".meta.pkl"))
        if Path(meta_path).exists():
            meta = joblib.load(meta_path)
            feature_cols = meta.get("feature_cols")
            threshold = meta.get("threshold")
        else:
            feature_cols = [
                "account_age_days",
                "avg_tx_amount",
                "txn_velocity",
                "earn_redeem_ratio",
                "txn_failure_rate",
                "device_count",
                "bot_behavior_index",
                "device_risk_score",
                "txn_suspicion_score",
                "rule_total_count",
            ]
            threshold = float(np.quantile(np.linspace(0, 1, 1000), 1 - m.contamination))
        scorer = Scorer(m, feature_cols, threshold)
        out_df = scorer.score(df)
        agg = RiskAggregator()
        out_df = agg.combine(out_df, cfg["risk"]["rule_weight"], cfg["risk"]["anomaly_weight"])
        th = Thresholding()
        out_df = th.apply(out_df, cfg["risk"]["block_threshold"], cfg["risk"]["review_threshold"])
        if args.output:
            out_df.to_csv(args.output, index=False)
            LOG.info("Wrote scored output to %s", args.output)
        else:
            print(out_df[["user_id", "anomaly_score", "final_risk_score", "decision"]].head())

    elif args.cmd == "serve":
        cfg = load_config(args.config)
        model_path = args.model_path
        if Path(model_path).exists():
            m = IsolationForestModel()
            m.load(model_path)
            meta_path = str(Path(model_path).with_suffix(".meta.pkl"))
            if Path(meta_path).exists():
                meta = joblib.load(meta_path)
                feature_cols = meta.get("feature_cols")
                threshold = meta.get("threshold")
            else:
                feature_cols = [
                    "account_age_days",
                    "avg_tx_amount",
                    "txn_velocity",
                    "earn_redeem_ratio",
                    "txn_failure_rate",
                    "device_count",
                    "bot_behavior_index",
                    "device_risk_score",
                    "txn_suspicion_score",
                    "rule_total_count",
                ]
                threshold = float(np.quantile(np.linspace(0, 1, 1000), 1 - m.contamination))
            _GLOBAL["model"] = m
            _GLOBAL["threshold"] = threshold
            _GLOBAL["feature_cols"] = feature_cols
            _GLOBAL["scorer"] = Scorer(m, feature_cols, threshold)
            LOG.info("Model loaded for serve.")
        else:
            LOG.warning("Model not found; server will start but /score will return 503.")
        uvicorn.run("fraud_prod_singlefile:app", host=args.host, port=args.port, log_level="info")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
# ...existing code...



# # deployment 
# mkdir fraud-prod-system : mkdir = make directory
#                           It creates a new folder named --> cd fraud-prod-system

# mkdir app : In production systems Code lives inside a folder (best practice)
#             Docker will copy this folder later (paste your code in this)

# touch requirements.txt : It lists Python libraries your app needs for Docker

# touch .gitignore : It tells git which files/folders to ignore (like models, logs, etc.)

# touch Dockerfile : FROM python:3.10-slim
#                    WORKDIR /app
#                    COPY requirements.txt .
#                    RUN pip install --no-cache-dir -r requirements.txt
#                    COPY app/ app/
#                    EXPOSE 8000
#                    CMD ["uvicorn", "app.fraud_prod_singlefile:app", "--host", "0.0.0.0", "--port", "8000"]


# docker build -t fraud-api .(Build image:)
# docker run -p 8000:8000 fraud-api (Run container: maps port 8000 in container to 8000 on host)

# http://localhost:8000/health (Check health endpoint)