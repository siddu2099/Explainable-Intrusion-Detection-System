# Hybrid Intrusion Detection System Using Machine Learning and Anomaly Detection

## Abstract

With the rapid growth of internet-connected systems, cyberattacks such as Distributed Denial of Service (DDoS), FTP-BruteForce, and SSH-BruteForce attacks have become increasingly common. Traditional signature-based Intrusion Detection Systems (IDS) struggle to detect unknown or evolving threats. This project presents a Hybrid Intrusion Detection System that combines machine learning, anomaly detection, and rule-based detection techniques for robust attack classification.

The proposed system uses CIC-IDS2017 and CIC-IDS2018 datasets and integrates XGBoost, IsolationForest, confidence-threshold filtering, and deterministic DDoS rule overrides into a unified detection pipeline. The system performs multiclass classification of BENIGN, DDOS, FTP-BRUTEFORCE, and SSH-BRUTEFORCE traffic while also identifying unknown anomalies and low-confidence predictions.

Feature engineering techniques such as burst ratio and byte-per-packet analysis were used to improve class separability. The final model achieved approximately 98% accuracy and 97.5% macro F1-score without using SMOTE, relying instead on balanced class weighting for realistic IDS behavior. Explainability and robustness were further validated using SHAP analysis, ablation studies, confusion matrices, feature stability analysis, and association rule mining.

---

# Chapter 1: Introduction

## 1.1 Background

Modern networks face increasingly sophisticated cyber threats capable of disrupting services, stealing information, and damaging infrastructure. Intrusion Detection Systems (IDS) are essential cybersecurity tools used to monitor network traffic and detect malicious activities.

Traditional IDS solutions are primarily:

* Signature-based
* Rule-based
* Static in nature

Although effective for known attacks, they struggle against:

* Zero-day attacks
* Evolving traffic patterns
* Large-scale network environments

Machine learning approaches improve IDS adaptability by learning attack behavior directly from data.

---

## 1.2 Problem Statement

Existing IDS systems suffer from:

* Poor generalization across datasets
* High false positives
* Weak minority attack detection
* Inability to identify unknown attacks
* Overdependence on synthetic oversampling methods such as SMOTE

The project aims to design a realistic hybrid IDS capable of:

* Detecting multiple attack categories
* Preserving DDoS traffic characteristics
* Identifying anomalies
* Providing explainable predictions

---

## 1.3 Objectives

The main objectives of the project are:

* Build a multiclass IDS using machine learning
* Detect DDoS, FTP-BruteForce, and SSH-BruteForce attacks
* Integrate anomaly detection using IsolationForest
* Develop deterministic DDoS override rules
* Reduce feature drift and redundant features
* Improve model interpretability using SHAP
* Evaluate performance using multiple metrics

---

# Chapter 2: Literature Survey

## 2.1 Traditional IDS

Traditional IDS systems use:

* Signature matching
* Rule-based analysis
* Port monitoring

Limitations:

* Cannot detect unknown attacks
* Require frequent rule updates
* Poor scalability

---

## 2.2 Machine Learning-Based IDS

Machine learning IDS systems classify traffic using:

* Random Forest
* Logistic Regression
* XGBoost
* Neural Networks

Advantages:

* Adaptive learning
* Better attack detection
* Automatic feature learning

Challenges:

* Dataset imbalance
* Overfitting
* Explainability issues

---

## 2.3 Hybrid IDS

Hybrid IDS combines:

* Signature detection
* Statistical analysis
* Machine learning
* Anomaly detection

Hybrid systems improve:

* Accuracy
* Robustness
* Zero-day attack detection

The proposed system follows this hybrid architecture.

---

# Chapter 3: Dataset Description

## 3.1 CIC-IDS2017

Contains:

* BENIGN traffic
* DDoS attack traffic

Features include:

* Flow duration
* Packet counts
* Packet lengths
* Byte statistics

---

## 3.2 CIC-IDS2018

Contains:

* BENIGN traffic
* FTP-BruteForce attacks
* SSH-BruteForce attacks

---

## 3.3 Final Dataset Distribution

Combined dataset size:

* 127,432 samples

Classes:

* BENIGN
* DDOS
* FTP-BRUTEFORCE
* SSH-BRUTEFORCE

---

# Chapter 4: Methodology

## 4.1 System Architecture

The final IDS pipeline:

```text
7 Features
   ↓
StandardScaler
   ↓
XGBoost Classifier
   ↓
DDoS Rule Override
   ↓
Confidence Threshold
   ↓
IsolationForest
   ↓
Final Prediction
```

---

## 4.2 Feature Engineering

Final selected features:

| Feature           | Purpose                |
| ----------------- | ---------------------- |
| uni_flow_duration | Flow lifetime analysis |
| uni_fwd_packets   | Flood/scan volume      |
| uni_bwd_packets   | Response asymmetry     |
| uni_fwd_bytes     | Payload size analysis  |
| uni_bwd_bytes     | Traffic imbalance      |
| burst_ratio       | DDoS detection         |
| byte_per_packet   | FTP vs SSH separation  |

### Engineered Features

#### Burst Ratio

```text
burst_ratio = fwd_packets / bwd_packets
```

Used to identify DDoS traffic where forward traffic greatly exceeds backward traffic.

#### Byte Per Packet

```text
byte_per_packet = fwd_bytes / fwd_packets
```

Used to distinguish FTP traffic from SSH traffic.

---

## 4.3 Data Preprocessing

Steps:

* Missing value handling
* Infinite value removal
* Feature clipping
* StandardScaler normalization

IQR normalization was removed because it compressed extreme DDoS values and reduced recall.

---

## 4.4 XGBoost Classifier

XGBoost was selected because:

* Handles tabular data efficiently
* Supports multiclass classification
* Provides feature importance
* Performs well on imbalanced datasets

### Hyperparameters

* Trees: 300
* Max depth: 6
* Learning rate: 0.05
* Subsample: 0.8

---

## 4.5 Class Weighting

SMOTE was removed to preserve realistic network behavior.

Balanced class weights were used instead.

DDoS class weight was boosted 2× to prioritize attack recall.

---

## 4.6 DDoS Rule Override

A deterministic rule was added:

```text
burst_ratio > 10
AND
uni_fwd_packets > 50
```

This ensures physically obvious DDoS flows are never ignored.

---

## 4.7 IsolationForest

IsolationForest was integrated for anomaly detection.

Purpose:

* Detect unknown attacks
* Identify outlier traffic
* Flag zero-day behavior

Training:

* 30% mixed training sample
* Contamination = 0.05

---

## 4.8 Confidence Thresholding

Predictions below confidence thresholds were labeled:

* LOW_CONFIDENCE

Thresholds:

* Global = 0.65
* DDOS = 0.55

---

# Chapter 5: Results and Analysis

## 5.1 Model Performance

| Metric          | Value  |
| --------------- | ------ |
| Accuracy        | 98.00% |
| Macro Precision | 96.96% |
| Macro Recall    | 98.21% |
| Macro F1-Score  | 97.52% |

---

## 5.2 Classification Results

| Class          | Precision | Recall | F1   |
| -------------- | --------- | ------ | ---- |
| BENIGN         | 1.00      | 0.98   | 0.99 |
| DDOS           | 1.00      | 1.00   | 1.00 |
| FTP-BRUTEFORCE | 0.90      | 0.99   | 0.94 |
| SSH-BRUTEFORCE | 0.99      | 0.96   | 0.97 |

---

## 5.3 Confusion Matrix Analysis

Observations:

* DDoS classification achieved near-perfect detection.
* Minor confusion occurred between FTP and SSH attacks.
* BENIGN traffic had very low false positives.

---

## 5.4 SHAP Explainability

SHAP analysis revealed:

* Flow duration strongly influenced SSH detection.
* Byte-per-packet influenced FTP detection.
* Packet count features influenced DDoS classification.

---

## 5.5 Ablation Study

Removing engineered features reduced performance:

* Removing burst_ratio reduced DDoS recall.
* Removing byte_per_packet reduced FTP/SSH separation quality.

This validates the importance of feature engineering.

---

# Chapter 6: Advantages and Limitations

## Advantages

* High accuracy
* Strong DDoS detection
* Hybrid detection architecture
* Explainable predictions
* Anomaly detection support
* Realistic training without SMOTE

---

## Limitations

* Limited number of datasets
* Random train-test split may introduce leakage
* No real-time packet capture integration
* Limited attack categories

---

# Chapter 7: Future Enhancements

Future improvements may include:

* Real-time packet capture using Scapy
* Deployment using Flask or FastAPI
* Deep learning integration
* Additional datasets (UNSW-NB15, BoT-IoT)
* Temporal validation
* Cloud deployment

---

# Chapter 8: Conclusion

This project successfully developed a Hybrid Intrusion Detection System capable of detecting multiple cyberattacks using machine learning and anomaly detection techniques. By integrating XGBoost, rule-based DDoS detection, IsolationForest, confidence filtering, SHAP explainability, and association rule mining, the system achieved strong classification performance while maintaining realistic IDS behavior.

The proposed architecture demonstrates that combining deterministic security rules with machine learning significantly improves robustness and attack detection capability. The project provides a scalable foundation for future cybersecurity research and real-world IDS deployment.





**Hybrid Intrusion Detection System**
*Python, XGBoost, IsolationForest, scikit-learn, SHAP, Pandas*

* Developed a hybrid machine learning-based Intrusion Detection System (IDS) using CIC-IDS2017 and CIC-IDS2018 datasets to detect BENIGN, DDoS, FTP-BruteForce, and SSH-BruteForce traffic.
* Built a multi-stage detection pipeline integrating XGBoost classification, rule-based DDoS detection, confidence-threshold filtering, and IsolationForest anomaly detection for zero-day attack identification.
* Engineered and optimized 7 network traffic features including burst ratio, byte-per-packet, packet counts, and flow statistics to improve attack separation and reduce feature drift.
* Achieved 98% classification accuracy and 97.5% macro F1-score without SMOTE by using balanced class weighting and scalable preprocessing with StandardScaler.
* Performed SHAP-based explainability analysis, feature stability evaluation, ablation studies, confusion matrix analysis, and association rule mining to validate model robustness and interpretability.
* Exported a production-ready IDS pipeline (`.pkl`) supporting real-time classification into attack, anomaly, and low-confidence categories.
