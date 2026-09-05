import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import TransactionStatus

def extract_features(df_transactions: pd.DataFrame, df_accounts: pd.DataFrame) -> pd.DataFrame:
    """Extracts features for layer 1 models and rules."""
    
    df = df_transactions.sort_values(by=['account_id', 'timestamp']).copy()
    
    # Merge accounts to get historical_avg_amount
    df = df.merge(df_accounts[['account_id', 'historical_avg_amount']], on='account_id', how='left')
    
    df = df.set_index('timestamp')
    
    # 1. Velocity features
    df['velocity_1min'] = df.groupby('account_id')['txn_id'].transform(lambda x: x.rolling('1min').count())
    df['velocity_1hr'] = df.groupby('account_id')['txn_id'].transform(lambda x: x.rolling('1h').count())
    df['velocity_24hr'] = df.groupby('account_id')['txn_id'].transform(lambda x: x.rolling('24h').count())
    
    # 2. Amount Z-Score
    # Use shift() so the current amount doesn't inflate the historical std
    running_std = df.groupby('account_id')['amount'].transform(lambda x: x.shift().expanding().std())
    running_std = running_std.fillna(1.0)
    running_std = running_std.replace(0, 1.0)
    df['amount_zscore'] = (df['amount'] - df['historical_avg_amount']) / running_std
    
    # 3. is_new_merchant_category
    def is_new_cat(s):
        seen = set()
        res = []
        for val in s:
            if val not in seen:
                res.append(True)
                seen.add(val)
            else:
                res.append(False)
        return res
    
    df['is_new_merchant_category'] = df.groupby('account_id')['merchant_category'].transform(is_new_cat)
    
    # 4. geo_device_consistency
    def check_consistency(s):
        from collections import defaultdict
        counts = defaultdict(int)
        res = []
        max_tuple = None
        max_count = 0
        for val in s:
            counts[val] += 1
            if counts[val] > max_count:
                max_count = counts[val]
                max_tuple = val
            
            if val == max_tuple:
                res.append(True)
            else:
                res.append(False)
        return res

    df['geo_device_tuple'] = list(zip(df['geo_location'], df['device_id']))
    df['geo_device_consistency'] = df.groupby('account_id')['geo_device_tuple'].transform(check_consistency)
    
    # 5. is_unusual_hour
    def check_unusual_hour(s):
        from collections import defaultdict
        counts = defaultdict(int)
        total = 0
        res = []
        for val in s:
            if total == 0:
                res.append(False) 
            else:
                freq = counts[val] / total
                res.append(freq < 0.05)
            counts[val] += 1
            total += 1
        return res

    df['hour'] = df.index.hour
    df['is_unusual_hour'] = df.groupby('account_id')['hour'].transform(check_unusual_hour)
    
    # 6. failure_retry_ratio
    # Get previous status and timestamp avoiding SettingWithCopyWarning
    prev_status = df.groupby('account_id')['status'].shift(1)
    
    # Extract timestamps for calculation safely
    timestamps = pd.Series(df.index, index=df.index)
    
    # Create a DataFrame to hold account_id and timestamps
    temp_df = pd.DataFrame({'account_id': df['account_id'], 'timestamp': timestamps})
    
    # Shift within groups
    prev_time = temp_df.groupby('account_id')['timestamp'].shift(1)
    
    # Time diff in seconds
    time_diff_sec = (timestamps - prev_time).dt.total_seconds()
    
    df['is_retry'] = (prev_status == TransactionStatus.failed.value) & (time_diff_sec <= 300)
                     
    df['retry_count_24hr'] = df.groupby('account_id')['is_retry'].transform(lambda x: x.rolling('24h').sum())
    
    df['failure_retry_ratio'] = df['retry_count_24hr'] / df['velocity_24hr']
    df['failure_retry_ratio'] = df['failure_retry_ratio'].fillna(0.0)
    
    df = df.reset_index()
    
    df = df.drop(columns=['geo_device_tuple', 'hour', 'is_retry', 'retry_count_24hr', 'historical_avg_amount'])
    
    return df
