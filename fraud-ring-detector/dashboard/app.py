"""Streamlit dashboard for Fraud Ring Detector."""
import os
import sys
import json
import pandas as pd
import streamlit as st
import subprocess
import threading
import time
import random
from datetime import datetime
import plotly.express as px
from streamlit_option_menu import option_menu

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title="Fraud Ring Detector", layout="wide", page_icon="🛡️", initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────────
# CSS: Complete Modern Theme
# ──────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* ── Global Font Reset ── */
    html, body, [class*="css"], p, span, div, h1, h2, h3, h4, h5, h6,
    label, input, button, textarea, select,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* ── Color Variables ── */
    :root {
        --danger: #ef4444;
        --warning: #f59e0b;
        --safe: #22c55e;
        --accent: #6366f1;
        --bg-primary: #0f1117;
        --bg-card: rgba(22, 24, 35, 0.85);
        --bg-card-hover: rgba(30, 33, 48, 0.9);
        --bg-inset: rgba(15, 17, 23, 0.6);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-hover: rgba(255, 255, 255, 0.12);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }
    
    /* ── Hide Sidebar ── */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    
    /* ── Global Layout ── */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* ── Header Bar ── */
    .header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .header-logo {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text-primary);
    }
    .header-logo span { color: var(--accent); }
    
    /* ── Metric Cards ── */
    div[data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: var(--text-secondary) !important;
    }
    
    /* ── Glass Cards ── */
    .card {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        border: 1px solid var(--border-subtle);
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    .card:hover {
        border-color: var(--border-hover);
        background: var(--bg-card-hover);
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0;
    }
    .card-txn {
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--text-secondary);
        font-family: 'SF Mono', 'Fira Code', monospace !important;
        word-break: break-all;
    }
    
    /* ── Reason Inset ── */
    .reason-inset {
        background: var(--bg-inset);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 0.8rem;
        border-left: 3px solid var(--accent);
    }
    .reason-label {
        display: block;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }
    .reason-text {
        font-size: 0.88rem;
        line-height: 1.5;
        color: var(--text-secondary);
    }
    
    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.7rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        flex-shrink: 0;
    }
    .badge-block   { background: rgba(239,68,68,0.12); color: var(--danger);  border: 1px solid rgba(239,68,68,0.25); }
    .badge-review  { background: rgba(245,158,11,0.12); color: var(--warning); border: 1px solid rgba(245,158,11,0.25); }
    .badge-log     { background: rgba(34,197,94,0.12);  color: var(--safe);    border: 1px solid rgba(34,197,94,0.25); }
    
    /* ── Cluster Card ── */
    .cluster-panel {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--border-subtle);
        border-left: 4px solid gray;
    }
    .cluster-panel h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .cluster-meta {
        display: flex;
        gap: 2rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    .cluster-meta strong { color: var(--text-muted); font-weight: 500; }
    .cluster-meta span { color: var(--text-primary); font-weight: 600; }
    
    /* ── Section Titles ── */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
    }
    .section-subtitle {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────
# State & Helpers
# ──────────────────────────────────────────────────
AUDIT_PATH = os.path.join(config.DATA_DIR, "audit_log.jsonl")

for key, default in [('last_audit_line_count', 0), ('parsed_audit_logs', []),
                     ('historical_pool', []), ('sim_active', False)]:
    if key not in st.session_state:
        st.session_state[key] = default

def read_audit_updates():
    if not os.path.exists(AUDIT_PATH):
        return []
    new_logs = []
    with open(AUDIT_PATH, 'r') as f:
        lines = f.readlines()
        if not st.session_state.historical_pool:
            st.session_state.historical_pool = [json.loads(l) for l in lines if l.strip()]
        if len(lines) > st.session_state.last_audit_line_count:
            for line in lines[st.session_state.last_audit_line_count:]:
                if line.strip():
                    new_logs.append(json.loads(line))
            st.session_state.last_audit_line_count = len(lines)
    if new_logs:
        st.session_state.parsed_audit_logs.extend(new_logs)
    return new_logs

def load_static_data():
    paths = {
        'clusters': os.path.join(config.DATA_DIR, "scored_clusters.json"),
        'metrics':  os.path.join(config.DATA_DIR, "evaluation_metrics.json"),
        'txn':      os.path.join(config.DATA_DIR, "transactions.parquet"),
    }
    clusters = json.load(open(paths['clusters'])) if os.path.exists(paths['clusters']) else []
    metrics  = json.load(open(paths['metrics']))  if os.path.exists(paths['metrics'])  else {}
    df_txn   = pd.read_parquet(paths['txn'])       if os.path.exists(paths['txn'])      else pd.DataFrame()
    return clusters, metrics, df_txn

clusters, metrics, df_txn = load_static_data()

# ──────────────────────────────────────────────────
# Simulator Thread
# ──────────────────────────────────────────────────
def traffic_simulator(speed_seconds):
    while st.session_state.get('sim_active', False):
        if st.session_state.historical_pool:
            tmpl = random.choice(st.session_state.historical_pool)
            new_entry = dict(tmpl)
            new_entry['timestamp'] = datetime.utcnow().isoformat() + "Z"
            with open(AUDIT_PATH, 'a') as f:
                f.write(json.dumps(new_entry) + "\n")
        time.sleep(speed_seconds)

# ──────────────────────────────────────────────────
# Header & Navigation
# ──────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div class="header-logo"><span>🛡️</span> FraudRing</div>
</div>
""", unsafe_allow_html=True)

nav_selection = option_menu(
    menu_title=None,
    options=["Live Monitor", "Summary", "Lookup", "Clusters"],
    icons=["activity", "bar-chart-fill", "search", "diagram-3-fill"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "4px 6px",
            "background-color": "var(--bg-card)",
            "border-radius": "12px",
            "border": "1px solid var(--border-subtle)",
            "margin-bottom": "1.5rem",
        },
        "icon": {"color": "var(--text-muted)", "font-size": "16px"},
        "nav-link": {
            "font-family": "'Inter', sans-serif",
            "font-size": "0.85rem",
            "font-weight": "500",
            "text-align": "center",
            "color": "var(--text-secondary)",
            "border-radius": "8px",
            "padding": "10px 20px",
            "margin": "0 2px",
            "--hover-color": "rgba(255,255,255,0.04)",
        },
        "nav-link-selected": {
            "background-color": "var(--accent)",
            "color": "#ffffff",
            "font-weight": "600",
        },
    }
)

# Settings popover (top-right)
with st.popover("⚙️ Settings"):
    st.markdown("**Simulation**")
    sim_toggle = st.toggle("Simulate Live Traffic", value=st.session_state.sim_active)
    sim_speed = st.slider("Speed (sec)", 1, 5, 2)
    if sim_toggle != st.session_state.sim_active:
        st.session_state.sim_active = sim_toggle
        if sim_toggle:
            threading.Thread(target=traffic_simulator, args=(sim_speed,), daemon=True).start()
    st.divider()
    st.markdown(f"""
    **Pipeline Config**
    ```
    THRESHOLD_BLOCK  = {config.THRESHOLD_BLOCK}
    THRESHOLD_REVIEW = {config.THRESHOLD_REVIEW}
    RULE_WEIGHT      = {config.LAYER1_RULE_WEIGHT}
    ```
    """)
    if st.button("Re-run Full Pipeline"):
        with st.spinner("Running pipeline..."):
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            res = subprocess.run([sys.executable, "run_pipeline.py"], cwd=script_dir, capture_output=True, text=True)
            if res.returncode == 0:
                st.session_state.last_audit_line_count = 0
                st.session_state.parsed_audit_logs = []
                st.session_state.historical_pool = []
                st.rerun()

# ──────────────────────────────────────────────────
# Helper: render badge HTML
# ──────────────────────────────────────────────────
def get_badge_html(action):
    if action == 'block':
        return "<span class='badge badge-block'>Blocked</span>"
    elif action == 'flag_for_review':
        return "<span class='badge badge-review'>Review</span>"
    return "<span class='badge badge-log'>Passed</span>"

# ══════════════════════════════════════════════════
# VIEW: Live Monitor
# ══════════════════════════════════════════════════
if nav_selection == "Live Monitor":

    @st.fragment(run_every=2)
    def live_activity_feed():
        _ = read_audit_updates()
        all_logs = st.session_state.parsed_audit_logs
        if not all_logs:
            st.info("No audit logs yet. Toggle **Simulate Live Traffic** in ⚙️ Settings.")
            return

        df_live = pd.DataFrame(all_logs)
        counts = df_live['action_taken'].value_counts()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Processed",  f"{len(df_live):,}")
        c2.metric("Flagged",    f"{counts.get('flag_for_review', 0):,}")
        c3.metric("Blocked",    f"{counts.get('block', 0):,}")
        c4.metric("Logged",     f"{counts.get('log_only', 0):,}")

        st.markdown('<p class="section-subtitle" style="margin-top:1.5rem;">Recent Transactions</p>', unsafe_allow_html=True)

        for log in all_logs[-5:][::-1]:
            badge = get_badge_html(log['action_taken'])
            st.markdown(f"""
            <div class="card">
                <div class="card-header">
                    <span class="card-txn">{log['txn_id']}</span>
                    {badge}
                </div>
                <div class="reason-inset">
                    <span class="reason-label">Why this classification</span>
                    <span class="reason-text">{log['explanation']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    @st.fragment(run_every=3)
    def live_charts():
        all_logs = st.session_state.parsed_audit_logs
        if not all_logs:
            return
        df_live = pd.DataFrame(all_logs)

        fig_hist = px.histogram(df_live, x="layer1_score", nbins=50, color_discrete_sequence=["#6366f1"])
        fig_hist.update_layout(
            title=dict(text="Score Distribution", font=dict(size=14, family="Inter")),
            margin=dict(l=10, r=10, t=40, b=10), height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8", size=11),
            xaxis_title=None, yaxis_title=None,
        )
        fig_hist.update_xaxes(showgrid=False, zeroline=False, color="#64748b")
        fig_hist.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, color="#64748b")
        st.plotly_chart(fig_hist, use_container_width=True)

        df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
        df_g = df_live.groupby([pd.Grouper(key='timestamp', freq='1min'), 'action_taken']).size().reset_index(name='count')
        cmap = {'block': '#ef4444', 'flag_for_review': '#f59e0b', 'log_only': '#22c55e'}
        fig_bar = px.bar(df_g, x="timestamp", y="count", color="action_taken", color_discrete_map=cmap)
        fig_bar.update_layout(
            title=dict(text="Action Volume (1m buckets)", font=dict(size=14, family="Inter")),
            margin=dict(l=10, r=10, t=40, b=10), height=280, barmode='stack',
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8", size=11),
            xaxis_title=None, yaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1,
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        )
        fig_bar.update_xaxes(showgrid=False, zeroline=False, color="#64748b")
        fig_bar.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, color="#64748b")
        st.plotly_chart(fig_bar, use_container_width=True)

    col_feed, col_charts = st.columns([1, 1.2], gap="large")
    with col_feed:
        st.markdown('<p class="section-title">Activity Feed</p>', unsafe_allow_html=True)
        live_activity_feed()
    with col_charts:
        st.markdown('<p class="section-title">Analytics</p>', unsafe_allow_html=True)
        live_charts()

# ══════════════════════════════════════════════════
# VIEW: Summary Dashboard
# ══════════════════════════════════════════════════
elif nav_selection == "Summary":
    st.markdown('<p class="section-title">Summary Dashboard</p>', unsafe_allow_html=True)
    if not metrics:
        st.warning("No metrics found. Run the pipeline first via ⚙️ Settings.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraud Cases",    f"{metrics.get('total_fraud_test_set', 0):,}")
        c2.metric("Caught (L1)",    f"{metrics.get('caught_layer1', 0):,}")
        c3.metric("Caught (L2)",    f"{metrics.get('caught_layer2_only', 0):,}")
        c4.metric("False Pos. (L2)", f"{metrics.get('false_positives_layer2', 0):,}")

        if os.path.exists(AUDIT_PATH):
            df_audit = pd.read_json(AUDIT_PATH, lines=True)
            if not df_audit.empty:
                st.markdown('<p class="section-subtitle" style="margin-top:2rem;">Action Distribution</p>', unsafe_allow_html=True)
                action_counts = df_audit['action_taken'].value_counts().reset_index()
                action_counts.columns = ['Action', 'Count']
                cmap = {'block': '#ef4444', 'flag_for_review': '#f59e0b', 'log_only': '#22c55e'}
                fig = px.bar(action_counts, x='Action', y='Count', color='Action',
                             color_discrete_map=cmap)
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10), height=350,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#94a3b8", size=12),
                    showlegend=False, xaxis_title=None, yaxis_title=None,
                )
                fig.update_xaxes(showgrid=False, color="#64748b")
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#64748b")
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════
# VIEW: Transaction Lookup
# ══════════════════════════════════════════════════
elif nav_selection == "Lookup":
    st.markdown('<p class="section-title">Transaction Lookup</p>', unsafe_allow_html=True)
    txn_id_input = st.text_input("Enter a transaction ID:", placeholder="e.g. 1055559a-cbfe-4314-...")

    if txn_id_input and os.path.exists(AUDIT_PATH):
        found = False
        with open(AUDIT_PATH, 'r') as f:
            for line in f:
                if txn_id_input in line:
                    entry = json.loads(line)
                    if entry['txn_id'] == txn_id_input:
                        found = True
                        badge = get_badge_html(entry['action_taken'])

                        st.markdown(f"""
                        <div class="card" style="margin-top:1rem;">
                            <div class="card-header" style="margin-bottom: 0.5rem;">
                                <span style="font-size:1.1rem; font-weight:600; color:var(--text-primary);">
                                    Investigation Result
                                </span>
                                {badge}
                            </div>
                            <div class="reason-inset">
                                <span class="reason-label">Why this classification</span>
                                <span class="reason-text">{entry['explanation']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Final Score",  f"{entry['final_score']:.3f}")
                        c2.metric("Layer 1 Score", f"{entry['layer1_score']:.3f}")
                        c3.metric("Ring Score",    f"{entry['ring_score']:.3f}" if entry['ring_score'] is not None else "N/A")

                        with st.expander("Raw Audit Entry"):
                            st.json(entry)
                        break
        if not found:
            st.error("Transaction not found in audit log.")

# ══════════════════════════════════════════════════
# VIEW: Cluster View
# ══════════════════════════════════════════════════
elif nav_selection == "Clusters":
    st.markdown('<p class="section-title">Network Analytics</p>', unsafe_allow_html=True)
    if not clusters:
        st.warning("No clusters found.")
    else:
        df_clusters = pd.DataFrame(clusters).sort_values(by="ring_score", ascending=False)
        selected_cluster = st.selectbox("Select cluster", df_clusters['cluster_id'])

        if selected_cluster:
            ci = df_clusters[df_clusters['cluster_id'] == selected_cluster].iloc[0]
            score = ci['ring_score']
            bc = "#64748b"
            if score >= config.THRESHOLD_BLOCK:   bc = "var(--danger)"
            elif score >= config.THRESHOLD_REVIEW: bc = "var(--warning)"

            st.markdown(f"""
            <div class="cluster-panel" style="border-left-color:{bc};">
                <h3>{selected_cluster}</h3>
                <div class="cluster-meta">
                    <div><strong>Score:</strong> <span>{score:.3f}</span></div>
                    <div><strong>Members:</strong> <span>{len(ci['member_accounts'])}</span></div>
                    <div><strong>Flags:</strong> <span>{', '.join(ci['flags'])}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            members = ci['member_accounts']
            if not df_txn.empty:
                df_ct = df_txn[df_txn['account_id'].isin(members)].copy()
                if not df_ct.empty:
                    df_ct['hour'] = df_ct['timestamp'].dt.floor('h')
                    timeline = df_ct.groupby(['hour', 'account_id']).size().reset_index(name='count')

                    st.markdown('<p class="section-subtitle" style="margin-top:1.5rem;">Behavioral Synchrony Timeline</p>', unsafe_allow_html=True)
                    st.vega_lite_chart(timeline, {
                        "mark": {"type": "circle", "opacity": 0.85},
                        "encoding": {
                            "x": {"field": "hour", "type": "temporal", "title": "Time"},
                            "y": {"field": "account_id", "type": "nominal", "title": "Account"},
                            "size": {"field": "count", "type": "quantitative"},
                            "color": {"field": "account_id", "type": "nominal", "legend": None}
                        },
                        "config": {
                            "background": "transparent",
                            "axis": {"labelColor": "#94a3b8", "titleColor": "#64748b",
                                     "gridColor": "rgba(255,255,255,0.04)", "labelFont": "Inter",
                                     "titleFont": "Inter"}
                        }
                    }, use_container_width=True)
                else:
                    st.info("No transactions found for these members.")
