import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.layer1.features import extract_features
from src.layer1.rules import evaluate_rules
from schemas import TransactionStatus

@pytest.fixture
def sample_accounts():
    return pd.DataFrame([
        {
            "account_id": "acc_normal",
            "created_at": datetime(2023, 1, 1),
            "kyc_verified": True,
            "registration_device_id": "dev_1",
            "registration_ip": "1.1.1.1",
            "historical_avg_amount": 100.0,
            "label_ring_member": False
        },
        {
            "account_id": "acc_attack",
            "created_at": datetime(2023, 1, 1),
            "kyc_verified": True,
            "registration_device_id": "dev_2",
            "registration_ip": "2.2.2.2",
            "historical_avg_amount": 200.0,
            "label_ring_member": False
        }
    ])

@pytest.fixture
def normal_sequence():
    # 5 slow transactions for normal user
    base_time = datetime(2023, 6, 1, 12, 0, 0)
    txns = []
    for i in range(5):
        txns.append({
            "txn_id": f"txn_normal_{i}",
            "account_id": "acc_normal",
            "timestamp": base_time + timedelta(hours=i),
            "amount": 100.0 + i*5, # small variation
            "merchant_id": "merch_1",
            "merchant_category": "retail",
            "payment_instrument_id": "card_1",
            "device_id": "dev_1",
            "ip_address": "1.1.1.1",
            "geo_location": "New York",
            "status": TransactionStatus.success.value,
            "label_fraud": False
        })
    return txns

@pytest.fixture
def attack_sequence():
    # Card testing attack (velocity > 5 in 1 min, amount < 50)
    base_time = datetime(2023, 6, 1, 12, 0, 0)
    txns = []
    for i in range(10):
        txns.append({
            "txn_id": f"txn_attack_{i}",
            "account_id": "acc_attack",
            "timestamp": base_time + timedelta(seconds=i*5), # very fast
            "amount": 5.0, # small amount
            "merchant_id": "merch_2",
            "merchant_category": "digital",
            "payment_instrument_id": "card_2",
            "device_id": "dev_2",
            "ip_address": "2.2.2.2",
            "geo_location": "London",
            "status": TransactionStatus.failed.value if i % 2 == 0 else TransactionStatus.success.value,
            "label_fraud": True
        })
    return txns

def test_extract_features_and_rules(sample_accounts, normal_sequence, attack_sequence):
    df_acc = sample_accounts
    df_txn = pd.DataFrame(normal_sequence + attack_sequence)
    
    features_df = extract_features(df_txn, df_acc)
    
    # Check velocity for normal vs attack
    normal_features = features_df[features_df['account_id'] == 'acc_normal']
    attack_features = features_df[features_df['account_id'] == 'acc_attack']
    
    # Normal should have max 1 min velocity of 1
    assert normal_features['velocity_1min'].max() == 1
    
    # Attack should have max 1 min velocity of 10
    assert attack_features['velocity_1min'].max() == 10
    
    # Check failure retry ratio for attack
    assert attack_features['failure_retry_ratio'].max() > 0.0
    
    # Evaluate Rules
    scored_df = evaluate_rules(features_df)
    normal_scored = scored_df[scored_df['account_id'] == 'acc_normal']
    attack_scored = scored_df[scored_df['account_id'] == 'acc_attack']
    
    # Normal user should not trigger anything
    assert normal_scored['rule_score'].max() == 0.0
    for rules_list in normal_scored['triggered_rules']:
        assert len(rules_list) == 0
        
    # Attack user should trigger card testing on later transactions
    card_testing_triggered = False
    for _, row in attack_scored.iterrows():
        if 'card_testing_flag' in row['triggered_rules']:
            card_testing_triggered = True
            assert row['rule_score'] == 1.0 # hard override
            
    assert card_testing_triggered, "Card testing rule did not fire for attack sequence"

def test_high_deviation_rule(sample_accounts):
    df_acc = sample_accounts
    
    # Create sequence that triggers high deviation
    base_time = datetime(2023, 6, 1, 12, 0, 0)
    txns = []
    # First some normal ones to establish baseline
    for i in range(5):
        txns.append({
            "txn_id": f"txn_{i}",
            "account_id": "acc_normal",
            "timestamp": base_time + timedelta(days=i),
            "amount": 100.0,
            "merchant_id": "merch_1",
            "merchant_category": "retail",
            "payment_instrument_id": "card_1",
            "device_id": "dev_1",
            "ip_address": "1.1.1.1",
            "geo_location": "New York",
            "status": TransactionStatus.success.value,
            "label_fraud": False
        })
        
    # Then a massive deviation with new geo
    txns.append({
            "txn_id": "txn_deviation",
            "account_id": "acc_normal",
            "timestamp": base_time + timedelta(days=6),
            "amount": 5000.0, # massive
            "merchant_id": "merch_1",
            "merchant_category": "retail",
            "payment_instrument_id": "card_1",
            "device_id": "dev_new", # new device
            "ip_address": "9.9.9.9",
            "geo_location": "Tokyo", # new geo
            "status": TransactionStatus.success.value,
            "label_fraud": True
    })
    
    df_txn = pd.DataFrame(txns)
    features_df = extract_features(df_txn, df_acc)
    scored_df = evaluate_rules(features_df)
    
    deviation_txn = scored_df[scored_df['txn_id'] == 'txn_deviation'].iloc[0]
    
    assert 'high_deviation_flag' in deviation_txn['triggered_rules']
    assert deviation_txn['rule_score'] > 0.0
