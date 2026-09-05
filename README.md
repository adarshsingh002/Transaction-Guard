# 🛡️ Fraud Ring Detector

A two-layer hybrid fraud detection system that catches both individual fraudulent transactions and coordinated abuse rings through network graph analysis — with fully explainable, auditable decisions.

![Architecture](docs/architecture_diagram.jpg)

> **For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)**

---

## The Problem

Traditional fraud detection systems analyze transactions in isolation. While they catch obvious attacks (stolen cards, suspicious amounts), they completely miss **organized fraud rings** — groups of fake accounts that individually behave normally but are secretly coordinated by the same bad actor.

This project solves that by combining **transaction-level ML** with **network-level graph analysis** to detect both solo fraud and hidden rings.

---

## Features

### 🔍 Layer 1 — Transaction-Level Analytics

| Feature | Description |
|---|---|
| **Rolling Window Features** | Velocity tracking (1min / 1hr / 24hr), amount z-score vs. account history, geo/device consistency, unusual hour detection, failure-retry ratio |
| **Heuristic Rules Engine** | Named, independently testable rules for card testing attacks, amount deviation + new device combos, and odd-hour + new merchant category patterns |
| **XGBoost ML Model** | Gradient-boosted tree classifier trained on engineered features AND rule flags together, outputting calibrated probability scores (0–1) |
| **Hybrid Blending** | Hard-override rules force scores ≥ 0.9 regardless of ML output; otherwise configurable weighted average |

### 🕸️ Layer 2 — Network Analytics

| Feature | Description |
|---|---|
| **Account Graph** | Undirected NetworkX graph where edges connect accounts sharing device IDs, /24 IP subnets, payment instruments, or suspicious registration timing |
| **Weighted Edges** | Edge weight = sum of identifier type weights + recency bonus. Hard identifiers (device) weigh more than weak signals (registration proximity) |
| **Louvain Community Detection** | Splits large connected components into tight sub-clusters (fraud rings) by maximizing modularity |
| **Ring Scoring** | Each cluster scored on account age homogeneity, behavioral synchrony (hourly activity correlation), and shared identifier density |

### ⚡ Fusion Engine

| Feature | Description |
|---|---|
| **Max-based Fusion** | `final_score = max(L1_score, ring_score)` — ensures ring members are flagged even if their individual transactions look clean |
| **Confidence Tiers** | Score ≥ 0.5 → flag for review; Score < 0.5 → log only |
| **Defensive Design** | The system flags and logs — it never autonomously blocks transactions |

### 📋 Audit Trail

| Feature | Description |
|---|---|
| **JSONL Audit Log** | Every decision persisted to `data/audit_log.jsonl` with full score breakdowns |
| **Dynamic Explanations** | Plain-language reasons generated from actual triggered rules, top ML features, or cluster membership — not static templates |
| **Query Helpers** | `get_audit_trail(txn_id)` and `get_cluster_history(cluster_id)` for programmatic lookups |

### 📊 Real-Time Dashboard

| Feature | Description |
|---|---|
| **Live Activity Feed** | `@st.fragment(run_every=2)` polls the audit log for new entries and displays them as styled cards with classification reasons |
| **Live Analytics** | Score distribution histogram and rolling 1-minute action volume chart, both updating without page flicker |
| **Transaction Lookup** | Enter any `txn_id` to see its full pipeline trail: scores, triggered rules, ring membership, and explanation |
| **Cluster View** | Detected clusters sorted by ring score with behavioral synchrony timeline charts showing coordinated activity |
| **Traffic Simulator** | Background thread replays historical transactions into the audit log for demo purposes |

### 🧪 Synthetic Data Generation

| Feature | Description |
|---|---|
| **Realistic Baseline** | 2,000 accounts and 50,000+ transactions with consistent device/geo profiles and realistic amount distributions per merchant category |
| **Embedded Fraud Patterns** | Card testing (rapid small txns, high failure rate), amount deviation (10x+ normal with new device), odd-hour + new category combos |
| **Embedded Abuse Rings** | 8 synthetic rings with shared devices/IPs, synchronized registration, and coordinated transaction timing |
| **Ambiguous Stress Test** | Deliberately tricky cases — families sharing routers, frequent travelers, one-off big purchases — to honestly measure false-positive friction |

### 📈 Evaluation & Reporting

| Feature | Description |
|---|---|
| **Layer 1 Metrics** | Precision, recall, F1, and ROC-AUC at both confidence tiers on held-out test set |
| **Ring Recovery** | Tracks how many of the 8 embedded rings were fully identified, partially captured, or missed entirely |
| **False Positive Analysis** | Friction cost measured specifically against the ambiguous set, not just the general population |
| **Markdown Report** | Auto-generated `data/evaluation_report.md` with metrics, ring recovery breakdown, and example audit entries |

---

## Project Structure

```
fraud-ring-detector/
├── config.py                     # All configurable parameters
├── schemas.py                    # TypedDict schemas for all data structures
├── run_pipeline.py               # End-to-end pipeline runner
│
├── src/
│   ├── generate_data.py          # Synthetic data generator
│   ├── fusion.py                 # Score fusion engine
│   ├── audit.py                  # Audit logging + explanation generator
│   ├── report.py                 # Evaluation report generator
│   │
│   ├── layer1/
│   │   ├── features.py           # Rolling window feature extraction
│   │   ├── rules.py              # Heuristic rules engine
│   │   └── model.py              # XGBoost training + prediction
│   │
│   └── layer2/
│       ├── graph.py              # NetworkX graph construction
│       ├── clustering.py         # Louvain community detection
│       └── scoring.py            # Cluster risk scoring
│
├── dashboard/
│   └── app.py                    # Streamlit real-time dashboard
│
├── tests/
│   ├── test_generate_data.py     # Data generation tests
│   ├── test_layer1.py            # Feature + rules + model tests
│   ├── test_layer2_graph.py      # Graph construction tests
│   └── test_layer2_scoring.py    # Clustering + scoring tests
│
├── data/                         # Generated at runtime
│   ├── accounts.parquet
│   ├── transactions.parquet
│   ├── audit_log.jsonl
│   ├── scored_clusters.json
│   ├── evaluation_metrics.json
│   ├── evaluation_report.md
│   └── models/
│       └── layer1_model.joblib
│
└── docs/
    └── architecture_diagram.jpg
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/adarshsingh002/Transaction-Guard.git
cd fraud-ring-detector
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Generate data + train model + build graph + score clusters + write audit log
PYTHONPATH=. python run_pipeline.py --regenerate
```

### Generate Evaluation Report

```bash
PYTHONPATH=. python src/report.py
```

### Launch Dashboard

```bash
PYTHONPATH=. streamlit run dashboard/app.py
```

### Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Configuration

All tunable parameters are centralized in [`config.py`](config.py):

| Parameter | Default | Purpose |
|---|---|---|
| `N_ACCOUNTS` | 2000 | Number of synthetic accounts |
| `N_TRANSACTIONS` | 50000 | Number of synthetic transactions |
| `TEST_SIZE` | 0.2 | Train/test split ratio |
| `LAYER1_RULE_WEIGHT` | 0.4 | Weight of rules in ML+rules blend |
| `THRESHOLD_REVIEW` | 0.5 | Score threshold for flagging |
| `THRESHOLD_BLOCK` | 0.85 | Score threshold for blocking |
| `LOUVAIN_MIN_COMPONENT_SIZE` | 6 | Min component size for Louvain |
| `REG_TIME_WINDOW_MINUTES` | 360 | Registration proximity window |

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | XGBoost |
| Graph Analysis | NetworkX |
| Community Detection | python-louvain |
| Data Processing | Pandas, NumPy |
| Synthetic Data | Faker |
| Dashboard | Streamlit, Plotly |
| Testing | pytest |

---

## License

This project is for educational and demonstration purposes.
