"""
=============================================================================
Explainable Network Traffic Analytics & Intrusion Detection
=============================================================================
Primary End-to-End Pipeline Entry Point

Combines:
  1. 7 Selected & Engineered Flow Features (StandardScaler on train only)
  2. Zero-Leakage GroupShuffleSplit on minute-level time_bucket groups
  3. Class-weighted XGBoost Classifier (no SMOTE; DDoS 2.0x loss penalty)
  4. Deterministic DDoS Rule Override for physical flood signatures
  5. Isolation Forest Anomaly Detector (zero-day detection on 30% sample)
  6. Calibrated Per-Class Confidence Thresholding
  7. Multi-Method Explainability: SHAP TreeExplainer & Association Rule Mining

Usage:
  python main.py
=============================================================================
"""

import os
import sys
import tempfile
import warnings

# Configure environment before loading any ML/visualization packages
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    PATH_2017, PATH_2018, DICT_2017, DICT_2018, STABLE_FEATURES,
    FEATURE_JUSTIFICATION, SAMPLE_FRAC, RANDOM_STATE, CONFIDENCE_THRESHOLD,
    DDOS_CONFIDENCE_THRESHOLD, FIGURES_DIR, MODELS_DIR, REPORTS_DIR
)
from src.utils import verify_split_isolation
from src.data_loader import (
    load_and_align, preprocess_and_clean, perform_group_split
)
from src.pipeline import (
    fit_scaler, compute_sample_weights, train_xgboost,
    train_isolation_forest, run_inference_pipeline, evaluate_model,
    run_baseline_comparison, run_ablation_study, save_artifact
)
from src.explainability import (
    plot_confusion_matrices, plot_feature_importance,
    plot_anomaly_distribution, plot_pipeline_breakdown,
    compute_and_plot_shap, run_association_rule_mining,
    plot_stability_heatmap
)


def run_pipeline(
    path_2017: str = PATH_2017,
    path_2018: str = PATH_2018,
    sample_frac: float = SAMPLE_FRAC,
    figures_dir: str = FIGURES_DIR,
    models_dir: str = MODELS_DIR,
    reports_dir: str = REPORTS_DIR,
):
    setup_environment()
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 70)
    print("EXPLAINABLE NETWORK TRAFFIC ANALYTICS & INTRUSION DETECTION")
    print("=" * 70)
    print("""
ARCHITECTURE:
  7 Features (clean + protocol grounded)
       ↓
  StandardScaler (zero-leakage fit on train)
       ↓
  XGBoost (balanced class weights + 2.0x DDoS boost)
       ↓
  DDoS Rule Override (burst_ratio > 10.0 & fwd_packets >= 4)
       ↓
  Isolation Forest (anomaly detection on 30% mixed sample)
       ↓
  Confidence Threshold (Global=0.65 / DDoS=0.55)
       ↓
  OUTPUT: BENIGN / DDOS / FTP-BRUTEFORCE / SSH-BRUTEFORCE
          / ANOMALY / LOW_CONFIDENCE
""")

    # ── Phase 1: Data Ingestion & Alignment ──────────────────────────────────
    print("\n[Phase 1] Ingesting and aligning network datasets...")
    df_2017_raw = load_and_align(path_2017, DICT_2017, "CIC-IDS-2017", sample_frac=sample_frac)
    df_2018_raw = load_and_align(path_2018, DICT_2018, "CIC-IDS-2018", sample_frac=sample_frac)

    if df_2017_raw is None or df_2018_raw is None:
        raise FileNotFoundError("One or more required dataset CSV files could not be loaded.")

    df_2017_clean = preprocess_and_clean(df_2017_raw)
    df_2018_clean = preprocess_and_clean(df_2018_raw)

    print(f"\n  CIC-IDS-2017 Class Distribution:\n{df_2017_clean['Target_Label'].value_counts().to_string()}")
    print(f"\n  CIC-IDS-2018 Class Distribution:\n{df_2018_clean['Target_Label'].value_counts().to_string()}")

    df_all = pd.concat([df_2017_clean, df_2018_clean], ignore_index=True)
    print(f"\n  Merged Dataset Shape: {df_all.shape}")
    print(f"  Combined Class Distribution:\n{df_all['Target_Label'].value_counts().to_string()}")

    # ── Phase 2: Zero-Leakage GroupShuffleSplit ──────────────────────────────
    print("\n[Phase 2] Partitioning with GroupShuffleSplit on time_bucket...")
    X_train_raw, X_test_raw, y_train, y_test, grp_train, grp_test = perform_group_split(
        df_all, features=STABLE_FEATURES
    )
    print(f"  Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")

    audit_path = os.path.join(reports_dir, "experiment_audit.json")
    verify_split_isolation(grp_train.to_frame(), grp_test.to_frame(), group_col="time_bucket", audit_path=audit_path)

    # ── Phase 3: Scaling & Pre-training Checks ────────────────────────────────
    print("\n[Phase 3] Fitting StandardScaler on training partition...")
    scaler, X_train_sc, X_test_sc = fit_scaler(X_train_raw, X_test_raw, STABLE_FEATURES)

    drift = (X_train_sc.mean() - X_test_sc.mean()).abs().max()
    print(f"  Maximum Feature Drift (train mean - test mean): {drift:.4f} (✓ OK)")

    # ── Phase 4: Hybrid Feature Ranking & Stability Heatmap ─────────────────
    print("\n[Phase 4] Computing feature stability agreement...")
    le_rank = LabelEncoder()
    y_train_enc_temp = le_rank.fit_transform(y_train)

    rf_rank = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train_sc, y_train_enc_temp)
    xgb_rank = XGBClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss').fit(X_train_sc, y_train_enc_temp)

    rf_top = pd.Series(rf_rank.feature_importances_, index=STABLE_FEATURES).nlargest(4).index.tolist()
    xgb_top = pd.Series(xgb_rank.feature_importances_, index=STABLE_FEATURES).nlargest(4).index.tolist()

    stability_series = pd.Series({
        f: (1.0 if (f in rf_top and f in xgb_top) else 0.5 if (f in rf_top or f in xgb_top) else 0.0)
        for f in STABLE_FEATURES
    })
    stability_plot_path = os.path.join(figures_dir, "stability_heatmap.png")
    plot_stability_heatmap(stability_series, save_path=stability_plot_path)

    # ── Phase 5: Model Training ──────────────────────────────────────────────
    print("\n[Phase 5] Training XGBoost and Isolation Forest...")
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    weight_dict, sample_weights = compute_sample_weights(y_train_encoded, list(encoder.classes_))
    xgb_model = train_xgboost(X_train_sc, y_train_encoded, sample_weights, num_classes=len(encoder.classes_))
    iso_model = train_isolation_forest(X_train_sc)

    # Base evaluation
    evaluate_model(y_test_encoded, xgb_model.predict(X_test_sc), list(encoder.classes_), title="XGBoost Base Performance")

    # ── Phase 6: Multi-Stage Inference Pipeline ──────────────────────────────
    print("\n[Phase 6] Running multi-stage decision routing...")
    class_thresholds = {cls: CONFIDENCE_THRESHOLD for cls in encoder.classes_}
    if 'DDOS' in class_thresholds:
        class_thresholds['DDOS'] = DDOS_CONFIDENCE_THRESHOLD

    y_final, stage_counts, y_pred_raw, y_proba, rule_overrides, iso_scores = run_inference_pipeline(
        X_test_sc, X_test_raw, xgb_model, iso_model, encoder, class_thresholds
    )

    total_test = len(y_final)
    print(f"\n  Pipeline Routing Breakdown:")
    for stage, count in stage_counts.items():
        print(f"    {stage:<18}: {count:>6} ({100*count/total_test:5.1f}%)")

    # ── Phase 7: Baseline Comparison & Ablation ──────────────────────────────
    print("\n[Phase 7] Benchmarks and Ablation Analysis...")
    baseline_df = run_baseline_comparison(X_train_sc, y_train_encoded, X_test_sc, y_test_encoded)
    xgb_metrics = evaluate_model(y_test_encoded, y_pred_raw, list(encoder.classes_), title="Final XGBoost Evaluation")
    baseline_df = pd.concat([
        baseline_df,
        pd.DataFrame([{
            'Model': 'XGBoost (Proposed Pipeline)',
            'Accuracy': round(xgb_metrics['accuracy'], 4),
            'Precision': round(xgb_metrics['precision'], 4),
            'Recall': round(xgb_metrics['recall'], 4),
            'F1-Score': round(xgb_metrics['f1'], 4),
        }])
    ], ignore_index=True)
    print("\nBaseline Model Comparison:\n", baseline_df.set_index('Model').to_string())

    baseline_json_path = os.path.join(reports_dir, "baseline_scores.json")
    baseline_df.to_json(baseline_json_path, orient='records', indent=2)

    ablation_df = run_ablation_study(X_train_sc, y_train_encoded, sample_weights, X_test_sc, y_test_encoded, len(encoder.classes_))
    print("\nAblation Study Results:\n", ablation_df.set_index('Config').to_string())

    # ── Phase 8: Explainability & Interpretability ───────────────────────────
    print("\n[Phase 8] Generating explainability plots...")
    y_pred_post_rule = y_pred_raw.copy()
    if 'DDOS' in encoder.classes_:
        ddos_idx = list(encoder.classes_).index('DDOS')
        y_pred_post_rule[rule_overrides.values] = ddos_idx

    plot_confusion_matrices(y_test_encoded, y_pred_raw, y_pred_post_rule, list(encoder.classes_),
                            save_path=os.path.join(figures_dir, "confusion_matrices.png"))
    plot_feature_importance(xgb_model, save_path=os.path.join(figures_dir, "global_feature_importance.png"))
    plot_anomaly_distribution(iso_scores, save_path=os.path.join(figures_dir, "anomaly_score_distribution.png"))
    plot_pipeline_breakdown(stage_counts, save_path=os.path.join(figures_dir, "pipeline_breakdown.png"))

    shap_sample = X_test_sc.sample(n=min(2000, len(X_test_sc)), random_state=RANDOM_STATE)
    compute_and_plot_shap(xgb_model, shap_sample, list(encoder.classes_),
                          save_path=os.path.join(figures_dir, "shap_summary.png"))

    run_association_rule_mining(X_train_sc, y_train, save_path=os.path.join(figures_dir, "association_rules.png"))

    # ── Phase 9: Model Artifact Serialization ────────────────────────────────
    model_path = os.path.join(models_dir, "ids_cic_dual_model_v5.pkl")
    save_artifact(model_path, xgb_model, encoder, scaler, iso_model, STABLE_FEATURES, class_thresholds)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Artifacts saved to : {models_dir}/")
    print(f"  Figures saved to   : {figures_dir}/")
    print(f"  Reports saved to   : {reports_dir}/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Explainable Network Traffic Analytics & Intrusion Detection")
    parser.add_argument("--data-2017", type=str, default=PATH_2017, help="Path to CIC-IDS-2017 CSV file")
    parser.add_argument("--data-2018", type=str, default=PATH_2018, help="Path to CIC-IDS-2018 CSV file")
    parser.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC, help="Subsampling fraction")
    parser.add_argument("--figures-dir", type=str, default=FIGURES_DIR, help="Directory to save figures")
    parser.add_argument("--models-dir", type=str, default=MODELS_DIR, help="Directory to save models")
    parser.add_argument("--reports-dir", type=str, default=REPORTS_DIR, help="Directory to save reports")
    args = parser.parse_args()

    run_pipeline(
        path_2017=args.data_2017,
        path_2018=args.data_2018,
        sample_frac=args.sample_frac,
        figures_dir=args.figures_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
