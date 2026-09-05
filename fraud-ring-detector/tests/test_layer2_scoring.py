"""Tests for layer 2 clustering and scoring."""
import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.layer2.graph import build_graph, get_connected_components
from src.layer2.clustering import refine_clusters
from src.layer2.scoring import score_clusters

@pytest.fixture(scope="module")
def data():
    accounts_path = os.path.join(config.DATA_DIR, 'accounts.parquet')
    transactions_path = os.path.join(config.DATA_DIR, 'transactions.parquet')
    
    if not os.path.exists(accounts_path) or not os.path.exists(transactions_path):
        pytest.skip("Data files not found. Run generate_data.py first.")
        
    df_acc = pd.read_parquet(accounts_path)
    df_txn = pd.read_parquet(transactions_path)
    return df_acc, df_txn

def test_layer2_pipeline(data):
    df_acc, df_txn = data
    
    ring_accounts = set(df_acc[df_acc['label_ring_member'] == True]['account_id'])
    
    print("\n--- Layer 2 Pipeline ---")
    print("1. Building graph...")
    G = build_graph(df_acc, df_txn)
    
    print("2. Extracting initial components...")
    initial_mapping, initial_summary = get_connected_components(G)
    
    print("3. Refining clusters with Louvain...")
    refined_mapping, refined_summary = refine_clusters(G, initial_mapping, initial_summary)
    
    print("4. Scoring clusters...")
    scored_clusters = score_clusters(refined_summary, df_acc, df_txn)
    
    assert len(scored_clusters) == len(refined_summary)
    
    fraud_scores = []
    benign_scores = []
    
    for cluster in scored_clusters:
        members = cluster['member_accounts']
        score = cluster['ring_score']
        
        fraud_overlap = sum(1 for m in members if m in ring_accounts)
        
        if fraud_overlap > len(members) * 0.5:
            fraud_scores.append(score)
        else:
            if len(members) > 1: # Only care about actual clusters
                benign_scores.append(score)
                
    avg_fraud = sum(fraud_scores) / len(fraud_scores) if fraud_scores else 0
    avg_benign = sum(benign_scores) / len(benign_scores) if benign_scores else 0
    
    print(f"\nAverage score of recovered Fraud Rings: {avg_fraud:.4f} (N={len(fraud_scores)})")
    print(f"Average score of Benign Ambiguous sets: {avg_benign:.4f} (N={len(benign_scores)})")
    
    assert avg_fraud > avg_benign, "Fraud rings do not score noticeably higher than benign clusters!"
    assert (avg_fraud - avg_benign) > 0.1, f"Score gap is too small: {avg_fraud - avg_benign:.4f}"
    
    print(f"Score Gap: {avg_fraud - avg_benign:.4f} -> SUCCESS!")
