import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.generate_data import generate_data

@pytest.fixture(scope="module")
def setup_data():
    # Run generation
    generate_data()
    
    accounts_path = os.path.join(config.DATA_DIR, 'accounts.parquet')
    transactions_path = os.path.join(config.DATA_DIR, 'transactions.parquet')
    
    df_acc = pd.read_parquet(accounts_path)
    df_txn = pd.read_parquet(transactions_path)
    
    return df_acc, df_txn

def test_schema_conformance(setup_data):
    df_acc, df_txn = setup_data
    
    expected_acc_cols = {'account_id', 'created_at', 'kyc_verified', 'registration_device_id', 
                         'registration_ip', 'historical_avg_amount', 'label_ring_member'}
    assert expected_acc_cols.issubset(set(df_acc.columns))
    
    expected_txn_cols = {'txn_id', 'account_id', 'timestamp', 'amount', 'merchant_id', 
                         'merchant_category', 'payment_instrument_id', 'device_id', 
                         'ip_address', 'geo_location', 'status', 'label_fraud'}
    assert expected_txn_cols.issubset(set(df_txn.columns))

def test_fraud_rate(setup_data):
    _, df_txn = setup_data
    fraud_rate = df_txn['label_fraud'].mean()
    # The requirement is between 2% and 8%
    assert 0.02 <= fraud_rate <= 0.08, f"Fraud rate {fraud_rate*100:.2f}% is outside the 2-8% range"

def test_rings_present(setup_data):
    df_acc, _ = setup_data
    ring_accounts = df_acc[df_acc['label_ring_member'] == True]
    
    # We requested at least 8 rings, each of size 4-8. 
    # Minimum ring accounts is 8 * 4 = 32
    assert len(ring_accounts) >= 32
    
    # Ring members should have shared IPs or devices
    shared_ips = ring_accounts.groupby('registration_ip').size()
    shared_devices = ring_accounts.groupby('registration_device_id').size()
    
    assert len(shared_ips[shared_ips >= 2]) > 0 or len(shared_devices[shared_devices >= 2]) > 0

def test_ambiguous_set_logic(setup_data):
    df_acc, _ = setup_data
    normal_accounts = df_acc[df_acc['label_ring_member'] == False]
    
    # Ensure there are families/ambiguous accounts sharing devices but marked as normal
    normal_shared_devices = normal_accounts.groupby('registration_device_id').size()
    
    assert len(normal_shared_devices[normal_shared_devices >= 2]) > 0, "No ambiguous (shared device but normal) accounts found"

def test_no_overlap(setup_data):
    df_acc, _ = setup_data
    ring_accounts = df_acc[df_acc['label_ring_member'] == True]
    normal_accounts = df_acc[df_acc['label_ring_member'] == False]
    
    # A quick check to ensure they are mutually exclusive subsets
    assert len(set(ring_accounts['account_id']).intersection(set(normal_accounts['account_id']))) == 0
