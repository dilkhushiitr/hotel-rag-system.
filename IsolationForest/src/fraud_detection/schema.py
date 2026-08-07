"""Input schemas used by the FastAPI service.

The API accepts a denormalized transaction payload. In production this usually
comes from backend services that already join transaction, user, device,
network, behavior, and historical aggregate data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TransactionScoreRequest(BaseModel):
    """Single transaction request body for real-time fraud scoring."""

    txn_id: str
    user_id: str
    txn_amount: float = Field(gt=0)
    txn_type: Literal["earn", "redeem", "transfer"]
    txn_timestamp: datetime
    txn_status: Literal["success", "failed"] = "success"

    account_age_days: int = Field(ge=1)
    kyc_status: Literal["verified", "unverified"] = "unverified"
    signup_country: str
    signup_city: str | None = None
    referral_count: int = Field(default=0, ge=0)
    referral_depth: int = Field(default=0, ge=0)

    device_id: str
    os_type: Literal["android", "ios"]
    app_version: str | None = None
    emulator_flag: int = Field(ge=0, le=1)
    rooted_flag: int = Field(ge=0, le=1)
    accounts_per_device: int = Field(default=1, ge=1)

    ip_address: str
    ip_country: str
    ip_city: str | None = None
    vpn_flag: int = Field(ge=0, le=1)
    proxy_flag: int = Field(ge=0, le=1)
    geo_mismatch_flag: int = Field(ge=0, le=1)
    accounts_per_ip: int = Field(default=1, ge=1)

    session_duration: float = Field(gt=0)
    click_count: int = Field(ge=0)
    time_between_actions: float = Field(gt=0)
    screen_flow_length: int = Field(ge=1)
    click_entropy: float = Field(ge=0)
    behavior_score: float = Field(ge=0, le=100)

    txn_count_1h: int = Field(default=0, ge=0)
    txn_count_24h: int = Field(default=0, ge=0)
    avg_txn_amount_7d: float = Field(default=0, ge=0)
    failure_rate_24h: float = Field(default=0, ge=0, le=1)
    wallet_balance_change_24h: float = 0


class RuleDetail(BaseModel):
    """A single fired rule and its weight."""

    rule: str
    weight: float


class TransactionScoreResponse(BaseModel):
    """Fraud scoring response returned to transaction systems."""

    txn_id: str
    user_id: str
    rule_score: float
    ml_anomaly_score: float
    final_fraud_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["ALLOW", "MANUAL_REVIEW", "BLOCK"]
    rules_fired: list[str]
    rule_details: list[RuleDetail]
    explanation: str

