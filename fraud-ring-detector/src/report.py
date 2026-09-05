"""Generates the final evaluation report for the Fraud Ring Detector."""
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    print("Loading data...")
    accounts_path = os.path.join(config.DATA_DIR, 'accounts.parquet')
    transactions_path = os.path.join(config.DATA_DIR, 'transactions.parquet')
    audit_path = os.path.join(config.DATA_DIR, 'audit_log.jsonl')
    clusters_path = os.path.join(config.DATA_DIR, 'scored_clusters.json')
    
    df_acc = pd.read_parquet(accounts_path)
    df_txn = pd.read_parquet(transactions_path)
    
    audit_logs = []
    with open(audit_path, 'r') as f:
        for line in f:
            if line.strip():
                audit_logs.append(json.loads(line))
                
    df_audit = pd.DataFrame(audit_logs)
    
    with open(clusters_path, 'r') as f:
        scored_clusters = json.load(f)
        
    # Prepare data for Layer 1 Evaluation
    # Join df_audit with df_txn to get true labels for test set
    df_eval = pd.merge(df_audit, df_txn[['txn_id', 'label_fraud', 'account_id']], on='txn_id', how='left')
    
    y_true = df_eval['label_fraud'].astype(int).values
    layer1_scores = df_eval['layer1_score'].values
    
    print("Calculating Layer 1 Metrics...")
    
    metrics_block = {
        'precision': precision_score(y_true, layer1_scores >= config.THRESHOLD_BLOCK, zero_division=0),
        'recall': recall_score(y_true, layer1_scores >= config.THRESHOLD_BLOCK, zero_division=0),
        'f1': f1_score(y_true, layer1_scores >= config.THRESHOLD_BLOCK, zero_division=0)
    }
    
    metrics_review = {
        'precision': precision_score(y_true, layer1_scores >= config.THRESHOLD_REVIEW, zero_division=0),
        'recall': recall_score(y_true, layer1_scores >= config.THRESHOLD_REVIEW, zero_division=0),
        'f1': f1_score(y_true, layer1_scores >= config.THRESHOLD_REVIEW, zero_division=0)
    }
    
    roc_auc = roc_auc_score(y_true, layer1_scores)
    
    # Layer 2 Ring Identification
    print("Calculating Layer 2 Ring Identification...")
    true_rings = df_acc[df_acc['label_ring_member'] == True].groupby('ring_id')['account_id'].apply(set).to_dict()
    
    # A cluster is flagged if its ring_score >= config.THRESHOLD_REVIEW
    flagged_clusters = {c['cluster_id']: set(c['member_accounts']) for c in scored_clusters if c['ring_score'] >= config.THRESHOLD_REVIEW}
    
    account_to_flagged_cluster = {}
    for cid, members in flagged_clusters.items():
        for m in members:
            account_to_flagged_cluster[m] = cid
            
    fully_identified = []
    partially_identified = []
    missed = []
    
    for ring_id, true_members in true_rings.items():
        found_in_clusters = set()
        members_found = 0
        for m in true_members:
            if m in account_to_flagged_cluster:
                found_in_clusters.add(account_to_flagged_cluster[m])
                members_found += 1
                
        if members_found == len(true_members) and len(found_in_clusters) == 1:
            fully_identified.append(ring_id)
        elif members_found > 0:
            partially_identified.append(ring_id)
        else:
            missed.append(ring_id)
            
    # Ambiguous Friction Cost
    print("Calculating Ambiguous Set Friction...")
    ambiguous_accounts = set(df_acc[df_acc['is_ambiguous'] == True]['account_id'])
    
    df_ambiguous_eval = df_eval[df_eval['account_id'].isin(ambiguous_accounts)]
    df_ambiguous_legit = df_ambiguous_eval[df_ambiguous_eval['label_fraud'] == False]
    
    total_ambiguous_legit = len(df_ambiguous_legit)
    
    if total_ambiguous_legit > 0:
        ambiguous_flagged = len(df_ambiguous_legit[df_ambiguous_legit['final_score'] >= config.THRESHOLD_REVIEW])
        ambiguous_friction = (ambiguous_flagged / total_ambiguous_legit) * 100
    else:
        ambiguous_flagged = 0
        ambiguous_friction = 0.0
        
    # Extract Audit Log Examples
    print("Extracting Audit Log Examples...")
    
    # 1. Clean Layer 1 block (High L1 score, not in a ring)
    # Check if empty before iloc
    l1_blocks = df_eval[(df_eval['layer1_score'] >= config.THRESHOLD_BLOCK) & (df_eval['ring_score'].isna())]
    l1_block = l1_blocks.iloc[0] if not l1_blocks.empty else None
    
    # 2. Layer 2 only escalation (layer 1 clean < 0.5, ring score high >= 0.5)
    l2_escalations = df_eval[(df_eval['layer1_score'] < config.THRESHOLD_REVIEW) & (df_eval['ring_score'] >= config.THRESHOLD_REVIEW)]
    l2_escalation = l2_escalations.iloc[0] if not l2_escalations.empty else None
    
    # 3. False Positive from ambiguous set
    fp_ambiguous_list = df_ambiguous_legit[df_ambiguous_legit['final_score'] >= config.THRESHOLD_REVIEW]
    fp_ambiguous = fp_ambiguous_list.iloc[0] if not fp_ambiguous_list.empty else None
    
    def format_example(row):
        if row is None or row.empty if hasattr(row, 'empty') else False: 
            return "*(No example found in this test split)*"
        try:
            entry = {
                "txn_id": row['txn_id'],
                "timestamp": row['timestamp'],
                "layer1_score": row['layer1_score'],
                "layer1_triggers": row['layer1_triggers'],
                "ring_membership": row['ring_membership'],
                "ring_score": row['ring_score'] if not pd.isna(row['ring_score']) else None,
                "final_score": row['final_score'],
                "action_taken": row['action_taken'],
                "explanation": row['explanation']
            }
            return f"```json\n{json.dumps(entry, indent=2)}\n```"
        except Exception as e:
            return f"*(Failed to parse example: {str(e)})*"

    report_content = f"""# Fraud Ring Detector: Evaluation Report

## 1. Layer 1: Transaction-Level Analytics (XGBoost + Rules)

Evaluation on held-out test set for individual transaction scoring.

| Metric | Block Tier (>= {config.THRESHOLD_BLOCK}) | Review Tier (>= {config.THRESHOLD_REVIEW}) |
|---|---|---|
| **Precision** | {metrics_block['precision']:.4f} | {metrics_review['precision']:.4f} |
| **Recall** | {metrics_block['recall']:.4f} | {metrics_review['recall']:.4f} |
| **F1 Score** | {metrics_block['f1']:.4f} | {metrics_review['f1']:.4f} |

**Overall ROC-AUC**: {roc_auc:.4f}

---

## 2. Layer 2: Network Analytics (Graph Clustering)

Recovery rate of the explicitly embedded Stage 1 synthetic fraud rings (Ground truth: {len(true_rings)} total rings).
*A cluster is considered "identified" if its ring score triggered at least a Review escalation (>= {config.THRESHOLD_REVIEW}).*

- **Fully Identified** ({len(fully_identified)}): All accounts tightly grouped into a single flagged cluster.
  - `{', '.join(fully_identified) if fully_identified else 'None'}`
- **Partially Identified** ({len(partially_identified)}): Fragmented across clusters or only partially captured.
  - `{', '.join(partially_identified) if partially_identified else 'None'}`
- **Missed Entirely** ({len(missed)}): Evaded network detection thresholds completely.
  - `{', '.join(missed) if missed else 'None'}`

---

## 3. Ambiguous Set Friction Cost

To ensure the clustering algorithms don't aggressively penalize innocent device-sharing configurations, we evaluated the False Positive Rate specifically against the isolated "Ambiguous Set" (households, travelers).

- **Total Legitimate Transactions in Ambiguous Set (Test Split)**: {total_ambiguous_legit}
- **Incorrectly Flagged/Blocked**: {ambiguous_flagged}
- **Estimated Friction Cost**: **{ambiguous_friction:.2f}%** of legitimate household/traveler transactions were delayed or blocked.

---

## 4. Audit Log Exemplars

### A. Clean Layer-1 Block
Transaction flagged heavily by transaction features/rules, independent of ring topology.
{format_example(l1_block)}

### B. Layer-2 Escalation
Transaction looked extremely clean in isolation (low Layer-1 score), but was blocked/flagged purely due to network analytics proving the account belongs to a fraud ring.
{format_example(l2_escalation)}

### C. Ambiguous False Positive
Legitimate transaction from a household/shared device that incorrectly triggered a block/flag.
{format_example(fp_ambiguous)}

"""
    
    report_path = os.path.join(config.DATA_DIR, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()
