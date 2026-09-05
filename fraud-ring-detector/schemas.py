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
    ring_id: Optional[str]
    is_ambiguous: bool

class TopFeature(TypedDict):
    feature: str
    value: float
    contribution: str

class Layer1Output(TypedDict):
    txn_id: str
    layer1_score: float
    triggered_rules: List[str]
    top_features: List[TopFeature]

class ClusterOutput(TypedDict):
    cluster_id: str
    member_accounts: List[str]
    ring_score: float
    shared_signals: List[str]
    flags: List[str]

class FusionOutput(TypedDict):
    txn_id: str
    layer1_score: float
    ring_membership: Optional[str]
    ring_score: Optional[float]
    final_score: float
    action: str

class AuditLogEntry(TypedDict):
    txn_id: str
    timestamp: str
    layer1_score: float
    layer1_triggers: List[str]
    ring_membership: Optional[str]
    ring_score: Optional[float]
    final_score: float
    action_taken: str
    explanation: str
