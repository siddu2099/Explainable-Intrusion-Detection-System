# Explainable Network Traffic Analytics & Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/ML-XGBoost%20%7C%20scikit--learn-orange.svg)](https://xgboost.readthedocs.io/)
[![XAI](https://img.shields.io/badge/Explainability-SHAP%20%7C%20ARM-red.svg)](https://shap.readthedocs.io/)

A production-grade, explainable hybrid network intrusion detection pipeline combining gradient-boosted trees (**XGBoost**), unsupervised anomaly detection (**Isolation Forest**), and deterministic rule filtering (**DDoS Rule Override**) trained and evaluated on the **CIC-IDS2017** and **CIC-IDS2018** benchmark network-traffic datasets.

---

## Highlights

* **Hybrid Intrusion-Detection Pipeline**: Combines multiclass XGBoost classification, Isolation Forest anomaly detection for zero-day outliers, calibrated confidence filtering, and deterministic DDoS rule post-processing.
* **Engineered & Selected 7 Network-Flow Features**: Curated low-drift physical flow counters and high-impact engineered indicators (`burst_ratio`, `byte_per_packet`) with `StandardScaler` normalization.
* **Realistic IDS Training (No SMOTE)**: Avoids unphysical synthetic packet interpolation by applying balanced class weighting with a 2.0× penalty boost for high-consequence DDoS events.
* **Zero-Leakage GroupShuffleSplit**: Partitions flows across minute-level `time_bucket` groups, strictly preventing temporal context leakage between train and test distributions with verified audit logs (`outputs/experiment_audit.json`).
* **Multi-Method Explainability**: Employs **SHAP TreeExplainer** for per-class feature attribution, global gain analysis, and **Association Rule Mining (ARM)** to extract human-interpretable attack condition rules.

---

## System Architecture

```
                       Raw Network Traffic (CIC-IDS2017 & CIC-IDS2018)
                                              │
                                              ▼
                             Feature Alignment & Extraction
                    [5 Raw Flow Counters + 2 Engineered Ratios = 7 Features]
                                              │
                                              ▼
                                 GroupShuffleSplit (80 / 20)
                       [Grouped by minute-level time_bucket: 0% Leakage]
                                              │
                                              ▼
                                       StandardScaler
                            [Fitted on Train partition only]
                                              │
                                              ▼
                                   XGBoost Multiclass Model
                         [Class-Weighted Loss + 2.0x DDoS Boost]
                                              │
                                              ▼
                                      Inference Pipeline
                                              │
                ┌─────────────────────────────┼─────────────────────────────┐
                ▼                             ▼                             ▼
       [Deterministic DDoS]           [Isolation Forest]           [Confidence Check]
    burst_ratio > 10.0 &              Distributional outlier?      Max class prob < threshold?
    uni_fwd_packets >= 4              (Contamination = 5%)         (Global: 0.65, DDoS: 0.55)
         │                                   │                               │
         ▼ Yes                               ▼ Yes                           ▼ Yes
       "DDOS"                            "ANOMALY"                   "LOW_CONFIDENCE"
    (Rule Override)                   (Zero-day Alert)                (Analyst Triage)
         │ No                                │ No                            │ No
         └───────────────────────────────────┴───────────────────────────────┘
                                              │
                                              ▼
                                    "BENIGN" / "DDOS" /
                             "FTP-BRUTEFORCE" / "SSH-BRUTEFORCE"
```

---

## Curated 7-Feature Set

Seven robust, protocol-grounded features were engineered and selected to maximize attack discriminability while minimizing cross-dataset feature drift:

| Feature | Type | Definition / Formula | Security Rationale |
| :--- | :--- | :--- | :--- |
| `uni_flow_duration` | Raw | Flow lifetime ($\mu s$) | Distinguishes ephemeral probes from persistent brute-force or exfiltration sessions. |
| `uni_fwd_packets` | Raw | Total forward packets | Measures outbound volume; high in floods and scanning sweeps. |
| `uni_bwd_packets` | Raw | Total backward packets | Server response count; collapses toward zero during overwhelming DDoS floods. |
| `uni_fwd_bytes` | Raw | Total forward payload (bytes) | Separates payload-heavy protocols (FTP data blobs) from lightweight auth strings (SSH). |
| `uni_bwd_bytes` | Raw | Total backward payload (bytes) | Quantifies downstream traffic and server reply magnitude. |
| `burst_ratio` | Engineered | $\frac{\text{uni\_fwd\_packets}}{\text{uni\_bwd\_packets} + \epsilon}$ | **DDoS Signature**: Under flood conditions, target servers cannot reply ($\text{burst\_ratio} \gg 10$). Normal traffic remains near symmetric ($\approx 1$). |
| `byte_per_packet` | Engineered | $\frac{\text{uni\_fwd\_bytes}}{\text{uni\_fwd\_packets} + \epsilon}$ | **Brute-Force Separator**: FTP authentication transfers large directory/command buffers, while SSH brute-force exchanges tiny encrypted handshake packets. |

---

## Empirical Evaluation

Evaluated on held-out test traffic ($n = 27,654$) across four operational network classes:

### Classification Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **BENIGN** | **1.00** | **0.98** | **0.99** | 16,344 |
| **DDOS** | **1.00** | **1.00** | **1.00** | 2,555 |
| **FTP-BRUTEFORCE** | **0.92** | **0.97** | **0.95** | 4,855 |
| **SSH-BRUTEFORCE** | **0.96** | **0.97** | **0.96** | 3,900 |
| **Macro Average** | **0.97** | **0.98** | **0.97** | 27,654 |
| **Overall Accuracy** | — | — | **97.81%** | 27,654 |

### Baseline Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8841 | 0.8710 | 0.8522 | 0.8601 |
| **Random Forest** | 0.9740 | 0.9612 | 0.9720 | 0.9665 |
| **XGBoost (Proposed v5 Pipeline)** | **0.9781** | **0.9677** | **0.9799** | **0.9736** |

### Feature Ablation Analysis

| Configuration | Features ($n$) | Accuracy | Macro F1 | Macro Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Full Pipeline (7 Features)** | **7** | **0.9781** | **0.9736** | **0.9799** |
| Without `burst_ratio` | 6 | 0.9612 | 0.9534 | 0.9580 |
| Without `byte_per_packet` | 6 | 0.9680 | 0.9611 | 0.9654 |
| Raw Features Only (No Engineered) | 5 | 0.9540 | 0.9421 | 0.9488 |

---

## Explainability & Interpretability

1. **SHAP TreeExplainer**: Quantifies exact marginal attribution per feature across individual flow classifications, validating that `burst_ratio` drives DDoS classification while `byte_per_packet` and `uni_flow_duration` disambiguate FTP vs. SSH brute-force attempts.
2. **Association Rule Mining (ARM)**: Employs Apriori and association rules on discretized flow indicators, discovering high-confidence ($\text{lift} > 8.0$) operational rules associating medium flow durations and low packet sizes with SSH attacks.
3. **Split Isolation Verification**: Enforces zero group overlap across train and test partitions (`outputs/experiment_audit.json`), logging a verified $0.0000\%$ leakage score.

---

## Repository Structure

```
├── main.py                     # Primary pipeline execution CLI entry point
├── projectv5.py                # Backward-compatibility wrapper (runs main.py)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── src/                        # Modular package source code
│   ├── __init__.py
│   ├── config.py               # Paths, features, thresholds, and hyperparameters
│   ├── data_loader.py          # Data ingestion, preprocessing, GroupShuffleSplit
│   ├── pipeline.py             # StandardScaler, XGBoost, DDoS rule, IsolationForest
│   ├── explainability.py       # SHAP TreeExplainer, ARM, and visualization plots
│   ├── feature_stability.py    # Cross-dataset rank invariance & stability analyzer
│   └── utils.py                # Split verification audit and runtime helpers
│
├── tests/                      # Automated unit and integration test suite
│   ├── __init__.py
│   └── test_pipeline.py        # Leakage audit, feature engineering, and rule tests
│
├── data/                       # Dataset directory (CIC-IDS-2017 & CIC-IDS-2018)
│   ├── CIC-IDS-2017/           # Friday DDoS and port scan captures
│   └── CIC-IDS-2018/           # Brute-force and DoS captures
│
├── models/                     # Serialized production model artifacts
│   └── ids_cic_dual_model_v5.pkl
│
├── reports/                    # Technical research reports & runtime audit logs
│   ├── REPORT.md               # Comprehensive methodology and findings report
│   ├── experiment_audit.json   # GroupShuffleSplit zero-leakage audit verification
│   ├── baseline_scores.json    # Logistic Regression vs RF vs XGBoost benchmarks
│   └── feature_stability_report.md
│
└── figures/                    # High-resolution publication visualizations
    ├── confusion_matrices.png          # Raw counts & per-class normalized recall
    ├── global_feature_importance.png   # XGBoost feature gain rankings
    ├── anomaly_score_distribution.png  # Isolation Forest distribution
    ├── pipeline_breakdown.png          # Tiered prediction routing breakdown
    ├── shap_summary.png                # SHAP per-class bar summary
    ├── association_rules.png           # Support vs. Confidence rule scatter
    ├── stability_heatmap.png           # Random Forest & XGBoost stability consensus
    ├── feature_rank_invariance_heatmap.png
    ├── feature_stability_scores.png
    └── domain_importance_comparison.png
```

---

## Quickstart

### 1. Prerequisites & Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/network-traffic-analytics-ids.git
cd network-traffic-analytics-ids
pip install -r requirements.txt
```

### 2. Dataset Setup

Ensure the CIC-IDS datasets are placed in the `data/` directory:
- `data/CIC-IDS-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `data/CIC-IDS-2018/02-14-2018.csv`

### 3. Run the Complete Pipeline

Execute the end-to-end pipeline:

```bash
python main.py
```

Or run via the backward-compatible wrapper:

```bash
python projectv5.py
```

### 4. Run Automated Tests

Execute the unit and integration test suite:

```bash
pytest -v tests/
```

### 5. Run Feature Stability Research Analysis

To evaluate feature rank invariance, Spearman rank correlation, and cross-domain shift:

```bash
python -m src.feature_stability
```

### 6. Inspect Outputs & Artifacts

The execution generates:
* **Trained Model Artifact**: `models/ids_cic_dual_model_v5.pkl` (contains trained XGBoost, Isolation Forest, Scaler, Encoder, and metadata)
* **Leakage Audit Record**: `reports/experiment_audit.json`
* **Benchmark Scores**: `reports/baseline_scores.json`
* **Visualizations**: All high-resolution PNG charts in `figures/`.

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.
