"""Audit logging functionality for the fraud ring detector."""
import os
import sys
import json
from datetime import datetime, timezone
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from schemas import FusionOutput, Layer1Output, ClusterOutput, AuditLogEntry

AUDIT_LOG_FILE = os.path.join(config.DATA_DIR, "audit_log.jsonl")

def generate_layer1_explanation(action: str, l1_out: Layer1Output) -> str:
    """Generates an explanation for a Layer 1 driven decision."""
    triggers = l1_out.get('triggered_rules', [])
    top_features = [f"{tf['feature']} ({tf['contribution']})" for tf in l1_out.get('top_features', [])[:2]]
    
    action_text = "blocked" if action == "block" else ("flagged for review" if action == "flag_for_review" else "logged")
    
    if triggers:
        return f"Transaction {action_text}: Triggered hard rules ({', '.join(triggers)}). Top contributing features were {', '.join(top_features)}."
    else:
        return f"Transaction {action_text}: Suspicious behavior detected by ML model. Top contributing features were {', '.join(top_features)}."

def generate_layer2_explanation(action: str, cluster: ClusterOutput) -> str:
    """Generates an explanation for a Layer 2 driven decision."""
    n_accounts = len(cluster.get('member_accounts', []))
    signals = cluster.get('shared_signals', [])
    flags = cluster.get('flags', [])
    
    action_text = "blocked" if action == "block" else ("flagged for review" if action == "flag_for_review" else "logged")
    
    signal_str = ', '.join(signals) if signals else "weak proximity signals"
    flag_str = ', '.join(flags) if flags else "general suspicious synchrony"
    
    return f"Transaction {action_text}: account belongs to {cluster['cluster_id']} ({n_accounts} accounts sharing {signal_str}; {flag_str})."

def generate_explanation(action: str, fusion_out: FusionOutput, l1_out: Layer1Output, cluster: Optional[ClusterOutput]) -> str:
    """Composes the final explanation based on what triggered the action."""
    layer1_score = fusion_out['layer1_score']
    ring_score = fusion_out['ring_score'] or 0.0
    
    # If the ring score was the primary driver for the action, use the layer 2 explanation.
    # Otherwise, use the layer 1 transactional explanation.
    if ring_score >= layer1_score and cluster is not None:
        return generate_layer2_explanation(action, cluster)
    else:
        return generate_layer1_explanation(action, l1_out)

def log_transactions(fusion_outputs: List[FusionOutput], layer1_outputs: List[Layer1Output], scored_clusters: List[ClusterOutput]):
    """Generates explanations and appends to the audit log."""
    
    l1_map = {out['txn_id']: out for out in layer1_outputs}
    cluster_map = {c['cluster_id']: c for c in scored_clusters}
    
    entries = []
    
    for f_out in fusion_outputs:
        txn_id = f_out['txn_id']
        l1_out = l1_map.get(txn_id)
        if not l1_out:
            continue
            
        cluster_id = f_out['ring_membership']
        cluster = cluster_map.get(cluster_id) if cluster_id else None
        
        explanation = generate_explanation(f_out['action'], f_out, l1_out, cluster)
        
        entry = AuditLogEntry(
            txn_id=txn_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            layer1_score=float(f_out['layer1_score']),
            layer1_triggers=l1_out.get('triggered_rules', []),
            ring_membership=f_out['ring_membership'],
            ring_score=float(f_out['ring_score']) if f_out['ring_score'] is not None else None,
            final_score=float(f_out['final_score']),
            action_taken=f_out['action'],
            explanation=explanation
        )
        entries.append(entry)
        
    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
    with open(AUDIT_LOG_FILE, 'a') as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    return entries

def get_audit_trail(txn_id: str) -> Optional[dict]:
    """Retrieves the audit log entry for a specific transaction."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return None
        
    with open(AUDIT_LOG_FILE, 'r') as f:
        for line in f:
            if not line.strip(): continue
            entry = json.loads(line)
            if entry.get('txn_id') == txn_id:
                return entry
    return None

def get_cluster_history(cluster_id: str) -> List[dict]:
    """Retrieves all audit log entries associated with a specific cluster."""
    history = []
    if not os.path.exists(AUDIT_LOG_FILE):
        return history
        
    with open(AUDIT_LOG_FILE, 'r') as f:
        for line in f:
            if not line.strip(): continue
            entry = json.loads(line)
            if entry.get('ring_membership') == cluster_id:
                history.append(entry)
                
    return history
