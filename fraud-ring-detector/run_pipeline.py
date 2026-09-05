"""Evaluation script for the full Fraud Ring Detector Pipeline."""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import subprocess
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.layer1.features import extract_features
from src.layer1.rules import evaluate_rules
from src.layer1.model import train, predict
from src.layer2.graph import build_graph, get_connected_components
from src.layer2.clustering import refine_clusters
from src.layer2.scoring import score_clusters
from src.fusion import fuse_scores
from src.audit import log_transactions

def run_data_generation():
    print("Regenerating synthetic data...")
    subprocess.run([sys.executable, "src/generate_data.py"], check=True)

def main():
    parser = argparse.ArgumentParser(description="Run full fraud ring detector pipeline.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate synthetic data before running.")
    args = parser.parse_args()
    
    if args.regenerate:
        run_data_generation()

    print("Loading data...")
    accounts_path = os.path.join(config.DATA_DIR, 'accounts.parquet')
    transactions_path = os.path.join(config.DATA_DIR, 'transactions.parquet')
    
    if not os.path.exists(accounts_path) or not os.path.exists(transactions_path):
        print("Data not found. Generating data...")
        run_data_generation()
        
    df_acc = pd.read_parquet(accounts_path)
    df_txn = pd.read_parquet(transactions_path)
    
    print("--- Layer 1 Pipeline ---")
    print("Extracting features...")
    df_features = extract_features(df_txn, df_acc)
    print("Evaluating rules...")
    df_features = evaluate_rules(df_features)
    
    print("Running ML model...")
    _, X_test, y_test = train(df_features)
    test_indices = X_test.index
    df_test_features = df_features.loc[test_indices]
    
    l1_outputs = predict(df_test_features)
    
    print("--- Layer 2 Pipeline ---")
    print("Building graph...")
    G = build_graph(df_acc, df_txn)
    
    print("Extracting initial components...")
    initial_mapping, initial_summary = get_connected_components(G)
    
    print("Refining clusters with Louvain...")
    refined_mapping, refined_summary = refine_clusters(G, initial_mapping, initial_summary)
    
    print("Scoring clusters...")
    scored_clusters = score_clusters(refined_summary, df_acc, df_txn)
    
    # Save scored clusters for dashboard
    clusters_path = os.path.join(config.DATA_DIR, "scored_clusters.json")
    with open(clusters_path, "w") as f:
        json.dump(scored_clusters, f)
        
    print("--- Fusion Pipeline ---")
    df_test_txn = df_txn.loc[test_indices]
    fusion_results = fuse_scores(l1_outputs, df_test_txn, scored_clusters)
    
    # Clear audit log before writing new run
    audit_path = os.path.join(config.DATA_DIR, "audit_log.jsonl")
    if os.path.exists(audit_path):
        os.remove(audit_path)
        
    print("Writing audit logs...")
    log_transactions(fusion_results, l1_outputs, scored_clusters)
    
    y_true = y_test.values
    
    layer1_scores = np.array([r['layer1_score'] for r in fusion_results])
    final_scores = np.array([r['final_score'] for r in fusion_results])
    
    block_threshold = config.THRESHOLD_BLOCK
    review_threshold = config.THRESHOLD_REVIEW
    
    # Impact of Layer 2 at REVIEW tier
    caught_l1_review = sum((layer1_scores >= review_threshold) & (y_true == 1))
    caught_l2_only_review = sum((layer1_scores < review_threshold) & (final_scores >= review_threshold) & (y_true == 1))
    total_fraud = sum(y_true == 1)
    
    l2_fp_review = sum((layer1_scores < review_threshold) & (final_scores >= review_threshold) & (y_true == 0))
    
    metrics = {
        "total_fraud_test_set": int(total_fraud),
        "caught_layer1": int(caught_l1_review),
        "caught_layer2_only": int(caught_l2_only_review),
        "false_positives_layer2": int(l2_fp_review)
    }
    
    metrics_path = os.path.join(config.DATA_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
        
    print("Evaluation Complete. Dashboard metrics saved to data/evaluation_metrics.json.")

if __name__ == "__main__":
    main()
