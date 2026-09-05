"""Layer 2 cluster scoring engine."""
import os
import sys
import numpy as np
import pandas as pd
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from schemas import ClusterOutput

def calculate_age_homogeneity(df_members: pd.DataFrame) -> float:
    """Calculates age homogeneity (0-1). 1.0 means highly synchronized registration."""
    if len(df_members) <= 1:
        return 0.0
    
    # Calculate std dev in hours
    std_hours = df_members['created_at'].std().total_seconds() / 3600.0
    
    # If std dev is small (e.g., < 2 hours), homogeneity is high.
    # We use a decay function so 0 hours = 1.0, 24 hours ~ 0.13, > 100 hours ~ 0
    score = np.exp(-std_hours / 12.0)
    return float(score)

def calculate_behavioral_synchrony(df_txns_members: pd.DataFrame, members: List[str]) -> float:
    """Calculates behavioral synchrony (0-1) based on hourly transaction binning correlation."""
    if df_txns_members.empty or len(members) <= 1:
        return 0.0
        
    df = df_txns_members[['account_id', 'timestamp']].copy()
    df['hour_bin'] = df['timestamp'].dt.floor('h') # Using 'h' alias to avoid FutureWarning
    
    # Pivot to get accounts as columns and hour_bins as index, filling missing with 0
    pivot = df.pivot_table(index='hour_bin', columns='account_id', aggfunc='size', fill_value=0)
    
    # Ensure all members are present even if they have 0 transactions
    for acc in members:
        if acc not in pivot.columns:
            pivot[acc] = 0
            
    # Calculate pairwise correlation matrix
    corr_matrix = pivot.corr(method='pearson')
    
    # Extract the upper triangle of the correlation matrix, excluding the diagonal
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    correlations = corr_matrix.values[upper_tri_indices]
    
    # Filter out NaNs (happens if an account has zero variance/zero txns)
    correlations = correlations[~np.isnan(correlations)]
    
    if len(correlations) == 0:
        return 0.0
        
    # Average correlation (cap below at 0)
    avg_corr = np.clip(np.mean(correlations), 0, 1)
    return float(avg_corr)

def calculate_identifier_density(shared_types: List[str]) -> float:
    """Scales the distinct shared identifier types to 0-1."""
    max_expected = 4.0 # device, payment, subnet, reg_proximity
    density = min(1.0, len(shared_types) / max_expected)
    return float(density)

def calculate_kyc_completeness(df_members: pd.DataFrame) -> float:
    """Returns 0-1 score where 1.0 means lack of KYC."""
    if len(df_members) == 0:
        return 0.0
    kyc_ratio = df_members['kyc_verified'].mean()
    # Score is lack of KYC
    return 1.0 - float(kyc_ratio)

def calculate_net_financial_behavior(df_txns_members: pd.DataFrame, df_txns_all: pd.DataFrame) -> float:
    """
    Scores (0-1) based on concentration of bad financial behavior (refunds/failed/disputed)
    compared to the global baseline.
    """
    if df_txns_members.empty:
        return 0.0
        
    def get_bad_rate(df):
        if df.empty:
            return 0.0
        is_bad = df['status'].astype(str).str.contains('fail|refund|disput', case=False, na=False)
        return float(is_bad.mean())
        
    cluster_bad_rate = get_bad_rate(df_txns_members)
    global_bad_rate = get_bad_rate(df_txns_all)
    
    if global_bad_rate == 0:
        global_bad_rate = 0.01 # prevent div zero
        
    ratio = cluster_bad_rate / global_bad_rate
    
    # Map ratio to 0-1 (ratio of 1 = 0.0 score, ratio of 5+ = 1.0 score)
    score = np.clip((ratio - 1.0) / 4.0, 0, 1)
    return float(score)

def score_clusters(df_summary: pd.DataFrame, df_accounts: pd.DataFrame, df_transactions: pd.DataFrame) -> List[ClusterOutput]:
    """
    Scores refined clusters and flags highly suspicious behavior.
    
    NOTE: This is a robust rule-based scoring algorithm. Once sufficient labeled rings are collected
    (e.g., > 100 verified fraud rings), this function can be easily swapped out to execute a trained 
    RandomForest or XGBoost model utilizing the exact same features extracted above as input vectors.
    """
    outputs = []
    
    for _, row in df_summary.iterrows():
        cluster_id = row['cluster_id']
        members = row['members']
        shared_types = row['shared_types']
        
        df_members = df_accounts[df_accounts['account_id'].isin(members)]
        df_txns_members = df_transactions[df_transactions['account_id'].isin(members)]
        
        # Calculate sub-scores
        score_age = calculate_age_homogeneity(df_members)
        score_sync = calculate_behavioral_synchrony(df_txns_members, members)
        score_id = calculate_identifier_density(shared_types)
        score_kyc = calculate_kyc_completeness(df_members)
        score_fin = calculate_net_financial_behavior(df_txns_members, df_transactions)
        
        # Weighted sum
        w = config.CLUSTER_SCORE_WEIGHTS
        ring_score = (
            score_age * w['age_homogeneity'] +
            score_sync * w['synchrony'] +
            score_id * w['identifier_density'] +
            score_kyc * w['kyc_lack'] +
            score_fin * w['financial_behavior']
        )
        
        flags = []
        if score_age > 0.8:
            flags.append("synchronized_registration")
        if score_sync > 0.7:
            flags.append("high_behavioral_synchrony")
        if score_id > 0.7:
            flags.append("dense_identifier_sharing")
        if score_kyc > 0.8:
            flags.append("low_kyc_verification")
        if score_fin > 0.8:
            flags.append("concentrated_refund_activity")
            
        outputs.append(ClusterOutput(
            cluster_id=cluster_id,
            member_accounts=members,
            ring_score=round(ring_score, 4),
            shared_signals=shared_types,
            flags=flags
        ))
        
    return outputs
