"""Fusion Engine combining Layer 1 and Layer 2 outputs."""
import os
import sys
import pandas as pd
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from schemas import Layer1Output, ClusterOutput, FusionOutput
from src.layer2.graph import is_account_in_flagged_cluster

def fuse_scores(layer1_outputs: List[Layer1Output], df_transactions: pd.DataFrame, scored_clusters: List[ClusterOutput]) -> List[FusionOutput]:
    """
    Fuses the Layer 1 transaction-level scores with Layer 2 cluster-level scores.
    """
    # Create lookup map for transaction -> account_id
    txn_to_acc = df_transactions.set_index('txn_id')['account_id'].to_dict()
    
    # Create lookup map for cluster -> ring_score
    cluster_score_map = {c['cluster_id']: c['ring_score'] for c in scored_clusters}
    
    fusion_outputs = []
    
    for l1 in layer1_outputs:
        txn_id = l1['txn_id']
        layer1_score = l1['layer1_score']
        
        account_id = txn_to_acc.get(txn_id)
        
        ring_score = None
        cluster_id = None
        
        if account_id:
            cluster_id = is_account_in_flagged_cluster(account_id)
            if cluster_id:
                ring_score = cluster_score_map.get(cluster_id, None)
                
        # We use max() instead of average here because if an account is confirmed
        # to be part of a highly suspicious fraud ring (high ring_score), even a 
        # completely benign-looking transaction (low layer1_score) from that account 
        # must still be aggressively blocked. Averages would dangerously dilute the signal
        # and allow rings to trickle transactions under the radar.
        final_score = max(layer1_score, ring_score if ring_score is not None else 0.0)
        
        if final_score >= config.THRESHOLD_BLOCK:
            action = "block"
        elif final_score >= config.THRESHOLD_REVIEW:
            action = "flag_for_review"
        else:
            action = "log_only"
            
        fusion_outputs.append(FusionOutput(
            txn_id=txn_id,
            layer1_score=layer1_score,
            ring_membership=cluster_id,
            ring_score=ring_score,
            final_score=final_score,
            action=action
        ))
        
    return fusion_outputs
