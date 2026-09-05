# Fraud Ring Detector: Evaluation Report

## 1. Layer 1: Transaction-Level Analytics (XGBoost + Rules)

Evaluation on held-out test set for individual transaction scoring.

| Metric | Block Tier (>= 0.85) | Review Tier (>= 0.5) |
|---|---|---|
| **Precision** | 0.8021 | 0.9478 |
| **Recall** | 0.1392 | 0.9530 |
| **F1 Score** | 0.2373 | 0.9504 |

**Overall ROC-AUC**: 0.9927

---

## 2. Layer 2: Network Analytics (Graph Clustering)

Recovery rate of the explicitly embedded Stage 1 synthetic fraud rings (Ground truth: 8 total rings).
*A cluster is considered "identified" if its ring score triggered at least a Review escalation (>= 0.5).*

- **Fully Identified** (2): All accounts tightly grouped into a single flagged cluster.
  - `syn_ring_0, syn_ring_7`
- **Partially Identified** (0): Fragmented across clusters or only partially captured.
  - `None`
- **Missed Entirely** (6): Evaded network detection thresholds completely.
  - `syn_ring_1, syn_ring_2, syn_ring_3, syn_ring_4, syn_ring_5, syn_ring_6`

---

## 3. Ambiguous Set Friction Cost

To ensure the clustering algorithms don't aggressively penalize innocent device-sharing configurations, we evaluated the False Positive Rate specifically against the isolated "Ambiguous Set" (households, travelers).

- **Total Legitimate Transactions in Ambiguous Set (Test Split)**: 490
- **Incorrectly Flagged/Blocked**: 3
- **Estimated Friction Cost**: **0.61%** of legitimate household/traveler transactions were delayed or blocked.

---

## 4. Audit Log Exemplars

### A. Clean Layer-1 Block
Transaction flagged heavily by transaction features/rules, independent of ring topology.
```json
{
  "txn_id": "29bcf7e9-3029-4736-9bc9-2ba102ad46b8",
  "timestamp": "2026-09-05T17:09:27.426123+00:00",
  "layer1_score": 0.9,
  "layer1_triggers": [
    "high_deviation_flag",
    "odd_hour_new_category_flag"
  ],
  "ring_membership": null,
  "ring_score": null,
  "final_score": 0.9,
  "action_taken": "block",
  "explanation": "Transaction blocked: Triggered hard rules (high_deviation_flag, odd_hour_new_category_flag). Top contributing features were high_deviation_flag (high), velocity_1hr (low)."
}
```

### B. Layer-2 Escalation
Transaction looked extremely clean in isolation (low Layer-1 score), but was blocked/flagged purely due to network analytics proving the account belongs to a fraud ring.
```json
{
  "txn_id": "33e02dcc-9e03-4d8b-94e6-a41dbdda10d7",
  "timestamp": "2026-09-05T17:09:27.433465+00:00",
  "layer1_score": 0.0,
  "layer1_triggers": [],
  "ring_membership": "refined_cluster_180",
  "ring_score": 0.5029,
  "final_score": 0.5029,
  "action_taken": "flag_for_review",
  "explanation": "Transaction flagged for review: account belongs to refined_cluster_180 (7 accounts sharing reg_proximity, subnet, device; synchronized_registration, dense_identifier_sharing)."
}
```

### C. Ambiguous False Positive
Legitimate transaction from a household/shared device that incorrectly triggered a block/flag.
```json
{
  "txn_id": "1055559a-cbfe-4314-9e9e-d52aa9b4b2ae",
  "timestamp": "2026-09-05T17:09:27.425344+00:00",
  "layer1_score": 0.5689,
  "layer1_triggers": [],
  "ring_membership": null,
  "ring_score": null,
  "final_score": 0.5689,
  "action_taken": "flag_for_review",
  "explanation": "Transaction flagged for review: Suspicious behavior detected by ML model. Top contributing features were velocity_1hr (high), geo_device_consistency (low)."
}
```

