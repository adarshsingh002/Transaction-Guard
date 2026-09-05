import pandas as pd
import numpy as np
from faker import Faker
import uuid
import random
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import config and schemas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from schemas import TransactionStatus

fake = Faker()

def init_seeds():
    Faker.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

def random_date(start_date_str, end_date_str):
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_normal_accounts(n_accounts):
    accounts = []
    for _ in range(n_accounts):
        accounts.append({
            "account_id": str(uuid.uuid4()),
            "created_at": random_date(config.START_DATE, config.END_DATE),
            "kyc_verified": random.random() > 0.1,  # 90% verified
            "registration_device_id": str(uuid.uuid4()),
            "registration_ip": fake.ipv4(),
            "historical_avg_amount": round(np.random.lognormal(mean=3, sigma=1), 2) + 10,
            "label_ring_member": False,
            "ring_id": None,
            "is_ambiguous": False
        })
    return accounts

def generate_rings(n_rings, accounts_list):
    # We will replace some normal accounts with ring accounts to keep total N constant, 
    # or just add them. Let's add them and we'll adjust the base count if needed.
    ring_accounts = []
    for i in range(n_rings):
        ring_id = f"syn_ring_{i}"
        ring_size = random.randint(4, 8)
        
        # Determine shared features for this ring
        share_ip = random.choice([True, False])
        share_device = random.choice([True, False])
        if not share_ip and not share_device:
            share_ip = True # share at least something
            
        shared_ip = fake.ipv4() if share_ip else None
        shared_device = str(uuid.uuid4()) if share_device else None
        
        # Registration time cluster
        base_time = random_date(config.START_DATE, config.END_DATE)
        
        for _ in range(ring_size):
            ring_accounts.append({
                "account_id": str(uuid.uuid4()),
                "created_at": base_time + timedelta(minutes=random.randint(0, 60)),
                "kyc_verified": random.random() > 0.8, # Mostly unverified (80% unverified)
                "registration_device_id": shared_device if shared_device else str(uuid.uuid4()),
                "registration_ip": shared_ip if shared_ip else fake.ipv4(),
                "historical_avg_amount": round(np.random.lognormal(mean=4, sigma=0.5), 2),
                "label_ring_member": True,
                "ring_id": ring_id,
                "is_ambiguous": False
            })
    return ring_accounts

def generate_ambiguous_accounts(n_ambiguous):
    ambiguous = []
    # e.g., families sharing a device
    n_families = n_ambiguous // 3
    for _ in range(n_families):
        shared_device = str(uuid.uuid4())
        shared_ip = fake.ipv4()
        family_size = random.randint(2, 4)
        for _ in range(family_size):
            ambiguous.append({
                "account_id": str(uuid.uuid4()),
                "created_at": random_date(config.START_DATE, config.END_DATE),
                "kyc_verified": True,
                "registration_device_id": shared_device,
                "registration_ip": shared_ip,
                "historical_avg_amount": round(np.random.lognormal(mean=3, sigma=1), 2) + 10,
                "label_ring_member": False,
                "ring_id": None,
                "is_ambiguous": True
            })
    
    # Fill the rest with normal accounts
    remaining = n_ambiguous - len(ambiguous)
    if remaining > 0:
        ambiguous.extend(generate_normal_accounts(remaining))
    return ambiguous

def generate_base_transactions(accounts, n_transactions):
    transactions = []
    merchants = [str(uuid.uuid4()) for _ in range(50)]
    categories = ['retail', 'travel', 'digital_goods', 'food', 'services']
    merchant_cats = {m: random.choice(categories) for m in merchants}
    
    # Pre-assign home locations and devices for normal behavior
    account_profiles = {}
    for acc in accounts:
        account_profiles[acc['account_id']] = {
            'device_id': acc['registration_device_id'],
            'ip_address': acc['registration_ip'],
            'geo_location': fake.city(),
            'payment_instruments': [str(uuid.uuid4()) for _ in range(random.randint(1, 3))]
        }

    for _ in range(n_transactions):
        acc = random.choice(accounts)
        profile = account_profiles[acc['account_id']]
        merchant = random.choice(merchants)
        
        # 90% chance to use known device/ip/geo
        if random.random() < 0.9:
            device = profile['device_id']
            ip = profile['ip_address']
            geo = profile['geo_location']
        else:
            device = str(uuid.uuid4())
            ip = fake.ipv4()
            geo = fake.city()

        amount = max(1.0, np.random.normal(acc['historical_avg_amount'], acc['historical_avg_amount']*0.2))
        
        status = np.random.choice([TransactionStatus.success, TransactionStatus.failed, TransactionStatus.refunded], p=[0.95, 0.03, 0.02])
        
        transactions.append({
            "txn_id": str(uuid.uuid4()),
            "account_id": acc['account_id'],
            "timestamp": random_date(config.START_DATE, config.END_DATE),
            "amount": round(amount, 2),
            "merchant_id": merchant,
            "merchant_category": merchant_cats[merchant],
            "payment_instrument_id": random.choice(profile['payment_instruments']),
            "device_id": device,
            "ip_address": ip,
            "geo_location": geo,
            "status": status.value,
            "label_fraud": False
        })
    return transactions

def embed_card_testing(transactions, accounts, num_cases=50):
    # Many small transactions in tight time window on one payment instrument, high failure rate
    for _ in range(num_cases):
        acc = random.choice(accounts)
        payment_instrument = str(uuid.uuid4())
        base_time = random_date(config.START_DATE, config.END_DATE)
        merchant = str(uuid.uuid4())
        device = str(uuid.uuid4())
        ip = fake.ipv4()
        geo = fake.city()
        
        num_attempts = random.randint(10, 30)
        for i in range(num_attempts):
            status = TransactionStatus.failed.value if random.random() < 0.8 else TransactionStatus.success.value
            transactions.append({
                "txn_id": str(uuid.uuid4()),
                "account_id": acc['account_id'],
                "timestamp": base_time + timedelta(seconds=i*15),
                "amount": round(random.uniform(0.5, 5.0), 2),
                "merchant_id": merchant,
                "merchant_category": "digital_goods",
                "payment_instrument_id": payment_instrument,
                "device_id": device,
                "ip_address": ip,
                "geo_location": geo,
                "status": status,
                "label_fraud": True
            })

def embed_amount_deviation(transactions, accounts, num_cases=50):
    # Amount far above historical avg, new geo/device
    for _ in range(num_cases):
        acc = random.choice(accounts)
        transactions.append({
            "txn_id": str(uuid.uuid4()),
            "account_id": acc['account_id'],
            "timestamp": random_date(config.START_DATE, config.END_DATE),
            "amount": round(acc['historical_avg_amount'] * random.uniform(5, 10), 2),
            "merchant_id": str(uuid.uuid4()),
            "merchant_category": "retail",
            "payment_instrument_id": str(uuid.uuid4()),
            "device_id": str(uuid.uuid4()),
            "ip_address": fake.ipv4(),
            "geo_location": fake.city(), # new geo
            "status": TransactionStatus.success.value,
            "label_fraud": True
        })

def embed_odd_hour(transactions, accounts, num_cases=50):
    # Odd hour (e.g., 2 AM - 4 AM) + new merchant category
    for _ in range(num_cases):
        acc = random.choice(accounts)
        # Force time to be between 2 and 4 AM
        ts = random_date(config.START_DATE, config.END_DATE)
        ts = ts.replace(hour=random.randint(2, 4))
        
        transactions.append({
            "txn_id": str(uuid.uuid4()),
            "account_id": acc['account_id'],
            "timestamp": ts,
            "amount": round(random.uniform(50, 500), 2),
            "merchant_id": str(uuid.uuid4()),
            "merchant_category": "travel", # assuming new
            "payment_instrument_id": str(uuid.uuid4()),
            "device_id": acc['registration_device_id'], # might use same device
            "ip_address": acc['registration_ip'],
            "geo_location": fake.city(),
            "status": TransactionStatus.success.value,
            "label_fraud": True
        })

def embed_ring_transactions(transactions, ring_accounts, num_txns_per_ring=20):
    # Synchronized transaction/refund timing
    # Group accounts by ring (roughly, since we just have a list, we can group them by registration IP/Device or time)
    # Actually, we generated rings sequentially, so let's just pick small chunks of ring accounts
    chunk_size = 5
    for i in range(0, len(ring_accounts), chunk_size):
        ring_subset = ring_accounts[i:i+chunk_size]
        if not ring_subset:
            continue
            
        base_time = random_date(config.START_DATE, config.END_DATE)
        merchant = str(uuid.uuid4())
        
        for _ in range(num_txns_per_ring):
            acc = random.choice(ring_subset)
            is_refund = random.random() < 0.3
            status = TransactionStatus.refunded.value if is_refund else TransactionStatus.success.value
            
            transactions.append({
                "txn_id": str(uuid.uuid4()),
                "account_id": acc['account_id'],
                "timestamp": base_time + timedelta(minutes=random.randint(0, 30)),
                "amount": round(random.uniform(100, 1000), 2),
                "merchant_id": merchant,
                "merchant_category": "retail",
                "payment_instrument_id": str(uuid.uuid4()),
                "device_id": acc['registration_device_id'],
                "ip_address": acc['registration_ip'],
                "geo_location": fake.city(),
                "status": status,
                "label_fraud": True # Since it's ring behavior, mark as fraud
            })

def generate_data():
    init_seeds()
    
    n_rings = config.NUM_RINGS
    n_ambiguous = 100
    n_normal_accounts = config.N_ACCOUNTS - n_ambiguous - (n_rings * 6) # approx
    
    print("Generating accounts...")
    normal_accounts = generate_normal_accounts(n_normal_accounts)
    ring_accounts = generate_rings(n_rings, normal_accounts)
    ambiguous_accounts = generate_ambiguous_accounts(n_ambiguous)
    
    all_accounts = normal_accounts + ring_accounts + ambiguous_accounts
    
    print("Generating base transactions...")
    transactions = generate_base_transactions(all_accounts, config.N_TRANSACTIONS)
    
    print("Embedding fraud patterns...")
    embed_card_testing(transactions, all_accounts, num_cases=100)
    embed_amount_deviation(transactions, all_accounts, num_cases=200)
    embed_odd_hour(transactions, all_accounts, num_cases=200)
    
    print("Embedding ring transactions...")
    embed_ring_transactions(transactions, ring_accounts, num_txns_per_ring=30)
    
    # Convert to DataFrames
    df_accounts = pd.DataFrame(all_accounts)
    df_transactions = pd.DataFrame(transactions)
    
    # Enforce schemas
    df_accounts['created_at'] = pd.to_datetime(df_accounts['created_at'])
    df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])
    
    # Save to parquet
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df_accounts.to_parquet(os.path.join(config.DATA_DIR, 'accounts.parquet'), index=False)
    df_transactions.to_parquet(os.path.join(config.DATA_DIR, 'transactions.parquet'), index=False)
    
    fraud_rate = df_transactions['label_fraud'].mean() * 100
    
    print("\n--- Summary ---")
    print(f"Total Accounts: {len(df_accounts)}")
    print(f"Total Transactions: {len(df_transactions)}")
    print(f"Fraud Rate: {fraud_rate:.2f}%")
    print(f"Number of Rings embedded: {n_rings}")
    print(f"Ring Accounts: {len(ring_accounts)}")
    print(f"Ambiguous Set Size: {len(ambiguous_accounts)}")
    print("Data successfully generated and saved to data/")

if __name__ == "__main__":
    generate_data()
