from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypedDict, List, Optional

class TransactionStatus(Enum):
    success = "success"
    failed = "failed"
    refunded = "refunded"
    disputed = "disputed"

@dataclass
class Transaction:
    txn_id: str
    account_id: str
    timestamp: datetime
    amount: float
    merchant_id: str
    merchant_category: str
    payment_instrument_id: str
    device_id: str
    ip_address: str
    geo_location: str
    status: TransactionStatus
    label_fraud: bool

@dataclass
class Account:
    account_id: str
    created_at: datetime
    kyc_verified: bool
    registration_device_id: str
    registration_ip: str
    historical_avg_amount: float
    label_ring_member: bool

class Layer1Output(TypedDict):
    """Layer 1 output schema placeholder"""
    txn_id: str
    rule_score: float
    ml_score: float

class ClusterOutput(TypedDict):
    """Layer 2 cluster output schema placeholder"""
    cluster_id: str
    account_ids: List[str]
    risk_score: float

class AuditLogEntry(TypedDict):
    """Audit log entry schema placeholder"""
    entry_id: str
    timestamp: datetime
    action: str
    details: str
