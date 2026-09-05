"""Streamlit dashboard for Fraud Ring Detector."""
import os
import sys
import json
import pandas as pd
import streamlit as st
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title="Fraud Ring Detector", layout="wide")

def load_data():
    audit_path = os.path.join(config.DATA_DIR, "audit_log.jsonl")
    clusters_path = os.path.join(config.DATA_DIR, "scored_clusters.json")
    metrics_path = os.path.join(config.DATA_DIR, "evaluation_metrics.json")
    transactions_path = os.path.join(config.DATA_DIR, "transactions.parquet")
    
    audit_logs = []
    if os.path.exists(audit_path):
        with open(audit_path, 'r') as f:
            for line in f:
                if line.strip():
                    audit_logs.append(json.loads(line))
                    
    clusters = []
    if os.path.exists(clusters_path):
        with open(clusters_path, 'r') as f:
            clusters = json.load(f)
            
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    df_txn = pd.DataFrame()
    if os.path.exists(transactions_path):
        df_txn = pd.read_parquet(transactions_path)
            
    return audit_logs, clusters, metrics, df_txn

audit_logs, clusters, metrics, df_txn = load_data()

st.sidebar.title("Controls")
if st.sidebar.button("Re-run Pipeline"):
    with st.spinner("Running pipeline..."):
        # Run it and print output
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run([sys.executable, "run_pipeline.py"], cwd=script_dir, capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Pipeline completed!")
            st.rerun()
        else:
            st.sidebar.error("Pipeline failed!")
            st.sidebar.text(result.stderr)

tab1, tab2, tab3 = st.tabs(["Summary Dashboard", "Transaction Lookup", "Cluster View"])

with tab1:
    st.header("Summary Dashboard")
    if not metrics:
        st.warning("No metrics found. Run the pipeline first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Fraud Cases", metrics.get("total_fraud_test_set", 0))
        col2.metric("Caught by Layer 1", metrics.get("caught_layer1", 0))
        col3.metric("Caught solely by Layer 2", metrics.get("caught_layer2_only", 0))
        
        st.metric("False Positives from Layer 2", metrics.get("false_positives_layer2", 0), delta_color="inverse")
        
        df_audit = pd.DataFrame(audit_logs)
        if not df_audit.empty:
            st.subheader("Action Counts (Test Set)")
            action_counts = df_audit['action_taken'].value_counts()
            st.bar_chart(action_counts)

with tab2:
    st.header("Transaction Lookup")
    txn_id_input = st.text_input("Enter txn_id to investigate:")
    if txn_id_input:
        entry = next((log for log in audit_logs if log['txn_id'] == txn_id_input), None)
        if entry:
            st.subheader("Decision Audit Trail")
            
            st.markdown(f"**Action Taken**: `{entry['action_taken'].upper()}`")
            st.markdown(f"**Explanation**: {entry['explanation']}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Score", f"{entry['final_score']:.3f}")
            col2.metric("Layer 1 Score", f"{entry['layer1_score']:.3f}")
            col3.metric("Layer 2 Ring Score", f"{entry['ring_score']:.3f}" if entry['ring_score'] is not None else "N/A")
            
            st.json(entry)
        else:
            st.error("Transaction not found in audit log.")

with tab3:
    st.header("Cluster View")
    if not clusters:
        st.warning("No clusters found.")
    else:
        df_clusters = pd.DataFrame(clusters)
        df_clusters = df_clusters.sort_values(by="ring_score", ascending=False)
        
        # Display sorted dataframe
        st.dataframe(df_clusters[['cluster_id', 'ring_score', 'shared_signals', 'flags', 'member_accounts']])
        
        selected_cluster = st.selectbox("Select a cluster to visualize", df_clusters['cluster_id'])
        
        if selected_cluster:
            cluster_info = df_clusters[df_clusters['cluster_id'] == selected_cluster].iloc[0]
            st.subheader(f"Timeline: {selected_cluster}")
            st.write(f"**Ring Score**: {cluster_info['ring_score']:.3f}")
            st.write(f"**Flags**: {', '.join(cluster_info['flags'])}")
            
            members = cluster_info['member_accounts']
            
            if not df_txn.empty:
                df_cluster_txns = df_txn[df_txn['account_id'].isin(members)].copy()
                if not df_cluster_txns.empty:
                    df_cluster_txns['hour'] = df_cluster_txns['timestamp'].dt.floor('h')
                    # Group by hour and account
                    timeline = df_cluster_txns.groupby(['hour', 'account_id']).size().reset_index(name='count')
                    
                    st.vega_lite_chart(timeline, {
                        "mark": "circle",
                        "encoding": {
                            "x": {"field": "hour", "type": "temporal", "title": "Time"},
                            "y": {"field": "account_id", "type": "nominal", "title": "Account ID"},
                            "size": {"field": "count", "type": "quantitative"},
                            "color": {"field": "account_id", "type": "nominal"}
                        }
                    }, use_container_width=True)
                else:
                    st.info("No transactions found for these members.")
