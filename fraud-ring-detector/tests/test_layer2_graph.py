import os
import sys
import pandas as pd
import pytest
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.layer2.graph import build_graph, get_connected_components, get_subnet, calculate_edge_weight, is_account_in_flagged_cluster

@pytest.fixture(scope="module")
def data():
    accounts_path = os.path.join(config.DATA_DIR, 'accounts.parquet')
    transactions_path = os.path.join(config.DATA_DIR, 'transactions.parquet')
    
    if not os.path.exists(accounts_path) or not os.path.exists(transactions_path):
        pytest.skip("Data files not found. Run generate_data.py first.")
        
    df_acc = pd.read_parquet(accounts_path)
    df_txn = pd.read_parquet(transactions_path)
    return df_acc, df_txn

def test_subnet_bucketing():
    assert get_subnet("192.168.1.55") == "192.168.1.0/24"
    assert get_subnet("10.0.0.1") == "10.0.0.0/24"
    assert get_subnet("invalid_ip") == "invalid_ip"

def test_edge_weight():
    # Only weak signal
    w1 = calculate_edge_weight({'reg_proximity'}, days_since_latest=0.0)
    assert w1 == 0.1 + 0.5 # Base + Recency

    # Hard signal + weak signal
    w2 = calculate_edge_weight({'device', 'reg_proximity'}, days_since_latest=30.0)
    assert w2 == 1.1 + 0.0 # Base + No Recency

def test_graph_and_embedded_rings(data):
    df_acc, df_txn = data
    
    ring_accounts = df_acc[df_acc['label_ring_member'] == True]['account_id'].tolist()
    assert len(ring_accounts) > 0, "No embedded rings found in dataset"
    
    print(f"\nBuilding graph with {len(df_acc)} accounts and {len(df_txn)} transactions...")
    G = build_graph(df_acc, df_txn)
    
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == len(df_acc)
    
    print("Extracting connected components...")
    mapping, df_summary = get_connected_components(G)
    
    # Check if cache works
    sample_acc = df_acc['account_id'].iloc[0]
    assert is_account_in_flagged_cluster(sample_acc) == mapping[sample_acc]
    
    # Analyze recovery of embedded rings
    # We want to ensure that all generated ring members end up in a cluster of size >= 4
    # (since our generate_data creates rings of size 4-8).
    
    sizes_for_ring_members = []
    
    for acc in ring_accounts:
        cluster_id = mapping[acc]
        cluster_size = df_summary[df_summary['cluster_id'] == cluster_id]['size'].values[0]
        sizes_for_ring_members.append(cluster_size)
        
        # An embedded ring member MUST be in a component with its peers. 
        # (It could merge into a larger component via shared IPs across rings)
        assert cluster_size >= 4, f"Ring account {acc} is isolated or in too small a cluster (size {cluster_size})"
        
    s = pd.Series(sizes_for_ring_members)
    print("\nCluster sizes containing ring members (shows if rings merged):")
    print(s.value_counts().sort_index())
    
    # Ambiguous set logic check: Ensure there are some clusters of size 2-3 (like families)
    # that exist and contain NO labeled ring members (perfectly benign clusters).
    
    benign_clusters = df_summary[~df_summary['members'].apply(lambda x: any(acc in ring_accounts for acc in x))]
    small_benign = benign_clusters[(benign_clusters['size'] >= 2) & (benign_clusters['size'] <= 3)]
    
    assert len(small_benign) > 0, "Failed to recover the ambiguous small benign clusters (families)"
    print(f"\nSuccessfully recovered {len(small_benign)} small, completely benign clusters (ambiguous set).")
