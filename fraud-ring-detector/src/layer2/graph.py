"""Layer 2 graph construction."""
import os
import sys
import pandas as pd
import networkx as nx
from itertools import combinations
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Global cache for real-time lookups
_FLAGGED_CLUSTER_CACHE: Dict[str, str] = {}

def get_subnet(ip_address: str) -> str:
    """Helper to convert IPv4 to a /24 subnet string (e.g. 192.168.1.55 -> 192.168.1.0/24)."""
    if not isinstance(ip_address, str) or '.' not in ip_address:
        return ip_address
    parts = ip_address.split('.')
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    return ip_address

def calculate_edge_weight(shared_types: set, days_since_latest: float) -> float:
    """
    Calculates the weight of an edge between two accounts.
    
    Formula:
    Base Weight = Sum of configured weights for each shared identifier type.
    Recency Bonus = max(0, 0.5 * (1 - days_since_latest_shared_event / 30.0))
    Total Weight = Base Weight + Recency Bonus
    
    This ensures that accounts sharing hard identifiers like a device get a strong edge (>= 1.0),
    while accounts only sharing a weak signal like registration proximity get a very weak edge (~0.1).
    Recent shared activity gives a small bump (up to +0.5) to help clustering algorithms prioritize active rings.
    """
    base_weight = 0.0
    if 'device' in shared_types:
        base_weight += config.EDGE_WEIGHT_SHARED_DEVICE
    if 'payment_instrument' in shared_types:
        base_weight += config.EDGE_WEIGHT_SHARED_PAYMENT
    if 'subnet' in shared_types:
        base_weight += config.EDGE_WEIGHT_SHARED_SUBNET
    if 'reg_proximity' in shared_types:
        base_weight += config.EDGE_WEIGHT_REG_PROXIMITY
        
    recency_bonus = max(0.0, 0.5 * (1.0 - (days_since_latest / 30.0)))
    
    return base_weight + recency_bonus

def build_graph(df_accounts: pd.DataFrame, df_transactions: pd.DataFrame) -> nx.Graph:
    """Builds the account-transaction graph for layer 2 analysis."""
    G = nx.Graph()
    
    # 1. Add all accounts as nodes
    for acc_id in df_accounts['account_id'].unique():
        G.add_node(acc_id)
        
    edges = defaultdict(lambda: {'types': set(), 'latest_ts': pd.Timestamp.min})
    
    def add_link(acc1, acc2, link_type, ts):
        if acc1 == acc2:
            return
        if acc1 > acc2: # Consistent ordering
            acc1, acc2 = acc2, acc1
            
        edge = edges[(acc1, acc2)]
        edge['types'].add(link_type)
        if ts > edge['latest_ts']:
            edge['latest_ts'] = ts

    # 2. Extract shared identifiers
    # 2a. Registration Device and IP (Subnet)
    print("Processing registration identifiers...")
    df_acc_temp = df_accounts[['account_id', 'registration_device_id', 'registration_ip', 'created_at']].copy()
    df_acc_temp['subnet'] = df_acc_temp['registration_ip'].apply(get_subnet)
    
    # Group by registration device
    for _, group in df_acc_temp.groupby('registration_device_id'):
        if len(group) > 1:
            accs = group['account_id'].tolist()
            ts_max = group['created_at'].max()
            for a, b in combinations(accs, 2):
                add_link(a, b, 'device', ts_max)
                
    # Group by subnet
    for _, group in df_acc_temp.groupby('subnet'):
        if len(group) > 1:
            accs = group['account_id'].tolist()
            ts_max = group['created_at'].max()
            for a, b in combinations(accs, 2):
                add_link(a, b, 'subnet', ts_max)
                
    # 2b. Transaction Device, IP (Subnet), Payment Instrument
    print("Processing transaction identifiers...")
    if not df_transactions.empty:
        df_txn_temp = df_transactions[['account_id', 'device_id', 'ip_address', 'payment_instrument_id', 'timestamp']].copy()
        df_txn_temp['subnet'] = df_txn_temp['ip_address'].apply(get_subnet)
        
        for _, group in df_txn_temp.groupby('device_id'):
            accs = group['account_id'].unique()
            if len(accs) > 1:
                ts_max = group['timestamp'].max()
                for a, b in combinations(accs, 2):
                    add_link(a, b, 'device', ts_max)
                    
        for _, group in df_txn_temp.groupby('subnet'):
            accs = group['account_id'].unique()
            if len(accs) > 1:
                ts_max = group['timestamp'].max()
                for a, b in combinations(accs, 2):
                    add_link(a, b, 'subnet', ts_max)
                    
        for _, group in df_txn_temp.groupby('payment_instrument_id'):
            accs = group['account_id'].unique()
            if len(accs) > 1:
                ts_max = group['timestamp'].max()
                for a, b in combinations(accs, 2):
                    add_link(a, b, 'payment_instrument', ts_max)
                    
    # 3. Registration Time Proximity (Weak Signal)
    print("Processing registration time proximity...")
    df_acc_sorted = df_acc_temp.sort_values('created_at').reset_index(drop=True)
    
    window_minutes = pd.Timedelta(minutes=config.REG_TIME_WINDOW_MINUTES)
    n = len(df_acc_sorted)
    
    for i in range(n):
        acc1 = df_acc_sorted.iloc[i]
        j = i + 1
        while j < n and (df_acc_sorted.iloc[j]['created_at'] - acc1['created_at']) <= window_minutes:
            acc2 = df_acc_sorted.iloc[j]
            add_link(acc1['account_id'], acc2['account_id'], 'reg_proximity', acc2['created_at'])
            j += 1
            
    # 4. Construct Final Graph Edges
    print("Building final graph edges...")
    # Let's use max timestamp in the data as 'now' to be deterministic for historical data
    max_ts = pd.Timestamp.min
    if not df_transactions.empty:
        max_ts = max(max_ts, df_transactions['timestamp'].max())
    if not df_accounts.empty:
        max_ts = max(max_ts, df_accounts['created_at'].max())
        
    for (u, v), edge_data in edges.items():
        types = edge_data['types']
        latest = edge_data['latest_ts']
        days_since = max(0.0, (max_ts - latest).total_seconds() / 86400.0) if max_ts != pd.Timestamp.min else 0.0
        
        weight = calculate_edge_weight(types, days_since)
        
        G.add_edge(u, v, weight=weight, shared_types=list(types))
        
    return G

def get_connected_components(G: nx.Graph):
    """
    Runs connected_components to get initial clusters.
    Returns:
    - dict mapping account_id -> cluster_id
    - dataframe per-cluster summary: size, list of shared identifier types, list of member account_ids
    """
    components = list(nx.connected_components(G))
    
    mapping = {}
    summary = []
    
    for idx, comp in enumerate(components):
        cluster_id = f"cluster_{idx}"
        member_list = list(comp)
        
        for acc in member_list:
            mapping[acc] = cluster_id
            
        # Collect all shared identifier types present in this cluster's internal edges
        subgraph = G.subgraph(comp)
        all_shared_types = set()
        for u, v, data in subgraph.edges(data=True):
            all_shared_types.update(data.get('shared_types', []))
            
        summary.append({
            'cluster_id': cluster_id,
            'size': len(comp),
            'shared_types': list(all_shared_types),
            'members': member_list
        })
        
    global _FLAGGED_CLUSTER_CACHE
    _FLAGGED_CLUSTER_CACHE = mapping
    
    df_summary = pd.DataFrame(summary)
    return mapping, df_summary

def is_account_in_flagged_cluster(account_id: str) -> Optional[str]:
    """
    Lightweight real-time lookup function backed by memory cache.
    Returns the cluster_id if found, else None.
    """
    return _FLAGGED_CLUSTER_CACHE.get(account_id, None)

def update_cluster_cache(mapping: dict):
    """Utility to safely update cache for tests or fusion layer."""
    global _FLAGGED_CLUSTER_CACHE
    _FLAGGED_CLUSTER_CACHE = mapping
