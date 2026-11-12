"""
Hybrid Intrusion Detection Pipeline: Training, Inference, and Evaluation
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from src.config import (
    STABLE_FEATURES, DDOS_BURST_THRESHOLD, DDOS_PACKETS_THRESHOLD,
    DDOS_WEIGHT_MULTIPLIER, CONFIDENCE_THRESHOLD, DDOS_CONFIDENCE_THRESHOLD,
    ISO_SAMPLE_FRAC, ISO_CONTAMINATION, RANDOM_STATE
)


def fit_scaler(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame, features: list = STABLE_FEATURES):
    """
    Fit StandardScaler strictly on the training partition and transform both splits.
    Zero test/validation distribution leakage.
    """
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_raw[features]), columns=features)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test_raw[features]), columns=features)
    return scaler, X_train_sc, X_test_sc


def compute_sample_weights(y_train_encoded: np.ndarray, class_names: list, ddos_multiplier: float = DDOS_WEIGHT_MULTIPLIER):
    """
    Calculate balanced class weights to compensate for severe attack imbalance
    without relying on synthetic artifact generation (SMOTE).
    """
    classes = np.unique(y_train_encoded)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_encoded)
    weight_dict = dict(enumerate(weights))

    if 'DDOS' in class_names:
        ddos_idx = list(class_names).index('DDOS')
        weight_dict[ddos_idx] *= ddos_multiplier

    sample_weights = np.array([weight_dict[y] for y in y_train_encoded])
    return weight_dict, sample_weights


def train_xgboost(X_train: pd.DataFrame, y_train_encoded: np.ndarray, sample_weights: np.ndarray, num_classes: int):
    """Train the primary multiclass XGBoost classifier."""
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    return model


def train_isolation_forest(X_train: pd.DataFrame, sample_frac: float = ISO_SAMPLE_FRAC, contamination: float = ISO_CONTAMINATION):
    """
    Train an Isolation Forest anomaly detector on a mixed sample of training traffic
    to establish broad distributional boundaries for detecting zero-day anomalies.
    """
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    sample_size = int(sample_frac * len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    iso_model.fit(X_train.iloc[sample_idx])
    return iso_model


def ddos_rule_override(X_raw_row: pd.Series, burst_thresh: float = DDOS_BURST_THRESHOLD, pkt_thresh: float = DDOS_PACKETS_THRESHOLD) -> bool:
    """
    Deterministic rule override for obvious physical DDoS signatures:
    Severely elevated forward-to-backward packet ratio with sustained volume.
    """
    return bool(
        X_raw_row['burst_ratio'] > burst_thresh and
        X_raw_row['uni_fwd_packets'] >= pkt_thresh
    )


def run_inference_pipeline(
    X_test_sc: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    xgb_model: XGBClassifier,
    iso_model: IsolationForest,
    encoder: LabelEncoder,
    class_thresholds: dict = None
):
    """
    Multi-stage decision routing pipeline:
      Stage 1: Deterministic DDoS Rule Override
      Stage 2: Isolation Forest Anomaly Detection (zero-day alerts)
      Stage 3: XGBoost Classifier + Per-Class Confidence Thresholding
    """
    if class_thresholds is None:
        class_thresholds = {cls: CONFIDENCE_THRESHOLD for cls in encoder.classes_}
        if 'DDOS' in class_thresholds:
            class_thresholds['DDOS'] = DDOS_CONFIDENCE_THRESHOLD

    # XGBoost base predictions
    y_pred = xgb_model.predict(X_test_sc)
    y_proba = xgb_model.predict_proba(X_test_sc)

    # Isolation Forest anomaly check
    iso_preds = iso_model.predict(X_test_sc)
    iso_scores = iso_model.score_samples(X_test_sc)

    # Rule checks on unscaled features
    rule_overrides = X_test_raw.reset_index(drop=True).apply(ddos_rule_override, axis=1)

    y_final = []
    stage_counts = {'DDOS_RULE': 0, 'ANOMALY': 0, 'LOW_CONFIDENCE': 0, 'CLASSIFIED': 0}

    for i in range(len(X_test_sc)):
        if rule_overrides.iloc[i]:
            y_final.append("DDOS")
            stage_counts['DDOS_RULE'] += 1
            continue

        if iso_preds[i] == -1:
            y_final.append("ANOMALY")
            stage_counts['ANOMALY'] += 1
            continue

        pred_idx = y_pred[i]
        pred_class = encoder.inverse_transform([pred_idx])[0]
        prob = y_proba[i][pred_idx]
        thr = class_thresholds.get(pred_class, CONFIDENCE_THRESHOLD)

        if prob < thr:
            y_final.append("LOW_CONFIDENCE")
            stage_counts['LOW_CONFIDENCE'] += 1
        else:
            y_final.append(pred_class)
            stage_counts['CLASSIFIED'] += 1

    return np.array(y_final), stage_counts, y_pred, y_proba, rule_overrides, iso_scores


def evaluate_model(y_true_encoded: np.ndarray, y_pred_encoded: np.ndarray, class_names: list, title: str = ""):
    """Compute and print standard multi-class classification metrics."""
    acc = accuracy_score(y_true_encoded, y_pred_encoded)
    prec = precision_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
    rec = recall_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
    f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)

    print(f"\n{title}:")
    print(f"  Accuracy       : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro Precision: {prec:.4f}")
    print(f"  Macro Recall   : {rec:.4f}")
    print(f"  Macro F1-Score : {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true_encoded, y_pred_encoded, target_names=class_names, zero_division=0))
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def run_baseline_comparison(X_train: pd.DataFrame, y_train_enc: np.ndarray, X_test: pd.DataFrame, y_test_enc: np.ndarray):
    """Benchmark against Logistic Regression and Random Forest."""
    baselines = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    }
    results = []
    for name, clf in baselines.items():
        clf.fit(X_train, y_train_enc)
        preds = clf.predict(X_test)
        results.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test_enc, preds), 4),
            'Precision': round(precision_score(y_test_enc, preds, average='macro', zero_division=0), 4),
            'Recall': round(recall_score(y_test_enc, preds, average='macro', zero_division=0), 4),
            'F1-Score': round(f1_score(y_test_enc, preds, average='macro', zero_division=0), 4),
        })
    return pd.DataFrame(results)


def run_ablation_study(X_train: pd.DataFrame, y_train_enc: np.ndarray, sample_weights: np.ndarray,
                       X_test: pd.DataFrame, y_test_enc: np.ndarray, num_classes: int):
    """Quantify the impact of removing individual engineered features."""
    configs = [
        ('Full (7 features)', STABLE_FEATURES),
        ('Without burst_ratio', [f for f in STABLE_FEATURES if f != 'burst_ratio']),
        ('Without byte_per_packet', [f for f in STABLE_FEATURES if f != 'byte_per_packet']),
        ('Raw 5 Features Only', ['uni_flow_duration', 'uni_fwd_packets', 'uni_bwd_packets', 'uni_fwd_bytes', 'uni_bwd_bytes']),
    ]
    results = []
    for label, feats in configs:
        fa = [f for f in feats if f in X_train.columns]
        clf = XGBClassifier(
            objective='multi:softprob', num_class=num_classes, n_estimators=300,
            max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_lambda=1.0, reg_alpha=0.5, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric='mlogloss'
        )
        clf.fit(X_train[fa], y_train_enc, sample_weight=sample_weights)
        preds = clf.predict(X_test[fa])
        results.append({
            'Config': label,
            'n_features': len(fa),
            'Accuracy': round(accuracy_score(y_test_enc, preds), 4),
            'Macro F1': round(f1_score(y_test_enc, preds, average='macro', zero_division=0), 4),
            'Macro Recall': round(recall_score(y_test_enc, preds, average='macro', zero_division=0), 4),
            'Macro Precision': round(precision_score(y_test_enc, preds, average='macro', zero_division=0), 4),
        })
    return pd.DataFrame(results)


def save_artifact(
    filepath: str,
    xgb_model: XGBClassifier,
    encoder: LabelEncoder,
    scaler: StandardScaler,
    iso_model: IsolationForest,
    features: list,
    class_thresholds: dict
):
    """Serialize the trained pipeline components into a deployable artifact bundle."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    artifact = {
        'model': xgb_model,
        'encoder': encoder,
        'features': features,
        'scaler': scaler,
        'anomaly_detector': iso_model,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'per_class_thresholds': class_thresholds,
        'ddos_burst_threshold': DDOS_BURST_THRESHOLD,
        'ddos_packets_threshold': DDOS_PACKETS_THRESHOLD,
        'ddos_weight_multiplier': DDOS_WEIGHT_MULTIPLIER,
        'meta': {
            'xgboost_version': xgb.__version__,
            'sklearn_version': sklearn.__version__,
            'feature_names': features,
            'n_features': len(features),
            'n_classes': len(encoder.classes_),
            'class_names': list(encoder.classes_),
            'prediction_labels': list(encoder.classes_) + ['ANOMALY', 'LOW_CONFIDENCE'],
            'split_strategy': 'GroupShuffleSplit on time_bucket',
        }
    }
    joblib.dump(artifact, filepath)
    print(f"Saved production artifact bundle to '{filepath}'")
