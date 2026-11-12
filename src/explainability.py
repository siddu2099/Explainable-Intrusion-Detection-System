"""
Explainability, Interpretability, and Publication Visualizations
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules


def plot_confusion_matrices(y_true_enc: np.ndarray, y_pred_enc: np.ndarray, y_pred_post_rule: np.ndarray,
                            class_names: list, save_path: str = "figures/confusion_matrices.png"):
    """Plot raw count confusion matrix alongside normalized recall matrix."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts before rule override
    cm_raw = confusion_matrix(y_true_enc, y_pred_enc)
    sns.heatmap(cm_raw, annot=True, fmt='g', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Confusion Matrix — Raw Counts\n(before DDoS rule override)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Normalized recall after DDoS rule override
    cm_rule = confusion_matrix(y_true_enc, y_pred_post_rule)
    cm_norm = np.divide(cm_rule, cm_rule.sum(axis=1, keepdims=True),
                        out=np.zeros_like(cm_rule, dtype=float),
                        where=cm_rule.sum(axis=1, keepdims=True) != 0)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    axes[1].set_title('Normalized Confusion Matrix — Per-Class Recall\n(diagonal = recall; after DDoS rule applied)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved '{save_path}'")


def plot_feature_importance(model, save_path: str = "figures/global_feature_importance.png"):
    """Plot XGBoost global feature gain importance."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    xgb.plot_importance(model, importance_type='gain', ax=ax,
                        title="Global Feature Importance (Gain)\n(7 features, StandardScaler, no SMOTE)",
                        color='#0047AB')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved '{save_path}'")


def plot_anomaly_distribution(scores: np.ndarray, save_path: str = "figures/anomaly_score_distribution.png"):
    """Plot Isolation Forest anomaly score histogram with percentile marker."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(9, 4))
    plt.hist(scores, bins=60, color='#B22222', edgecolor='white', alpha=0.8)
    p5 = np.percentile(scores, 5)
    plt.axvline(p5, color='black', linestyle='--', label=f'5th percentile = {p5:.3f}')
    plt.title('Isolation Forest Anomaly Score Distribution\n(Trained on 30% mixed sample)')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved '{save_path}'")


def plot_pipeline_breakdown(stage_counts: dict, save_path: str = "figures/pipeline_breakdown.png"):
    """Plot pie chart of decision routing stages."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(7, 7))
    colors = ['#CC0000', '#B22222', '#FF8C00', '#0047AB']
    plt.pie(list(stage_counts.values()), labels=list(stage_counts.keys()),
            autopct='%1.1f%%', colors=colors, startangle=140,
            wedgeprops=dict(edgecolor='white', linewidth=1.5))
    plt.title('Full Pipeline Prediction Breakdown\n(DDoS Rule → IsolationForest → XGB+Threshold)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved '{save_path}'")


def compute_and_plot_shap(model, X_sample: pd.DataFrame, class_names: list, save_path: str = "figures/shap_summary.png"):
    """Compute SHAP TreeExplainer attributions and save per-class bar summary plot."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_sample)

    sv = shap_vals.values if hasattr(shap_vals, 'values') else shap_vals
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        shap_plot = [sv[:, :, i] for i in range(sv.shape[2])]
    elif isinstance(sv, list):
        shap_plot = sv
    else:
        shap_plot = shap_vals

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_plot, X_sample, plot_type="bar", class_names=class_names, show=False)
    plt.title("SHAP Feature Importance — Per-Class Breakdown\n(7 features | no SMOTE | StandardScaler)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved '{save_path}'")


def run_association_rule_mining(X_train: pd.DataFrame, y_train: pd.Series, save_path: str = "figures/association_rules.png"):
    """Mine association rules linking discretized traffic patterns to attack classes."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    attack_df = X_train.copy()
    attack_df['Target_Label'] = y_train.reset_index(drop=True).values
    attack_df = attack_df[attack_df['Target_Label'] != 'BENIGN']

    if len(attack_df) < 50:
        print("Insufficient attack traffic for Association Rule Mining.")
        return

    for col, lbl in [
        ('uni_flow_duration', 'flow_duration_bin'),
        ('uni_fwd_bytes',     'fwd_bytes_bin'),
        ('burst_ratio',       'burst_ratio_bin'),
        ('byte_per_packet',   'byte_per_packet_bin'),
    ]:
        if col in attack_df.columns:
            attack_df[lbl] = pd.qcut(attack_df[col].rank(method='first'), q=3, labels=['low', 'medium', 'high'])

    bin_cols = [c for c in ['flow_duration_bin', 'fwd_bytes_bin', 'burst_ratio_bin', 'byte_per_packet_bin', 'Target_Label'] if c in attack_df.columns]
    attack_encoded = pd.get_dummies(attack_df[bin_cols])

    frequent_itemsets = apriori(attack_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    rules = rules[rules['consequents'].astype(str).str.contains('Target_Label')].sort_values('lift', ascending=False)

    if not rules.empty:
        print("\nTop Association Rules (Traffic Indicators → Attack Class):")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string(index=False))

        plt.figure(figsize=(9, 5))
        sns.scatterplot(data=rules, x='support', y='confidence', size='lift', hue='lift', palette='viridis', sizes=(20, 200))
        plt.title("Association Rules: Support vs Confidence")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor='white')
        plt.close()
        print(f"Saved '{save_path}'")


def plot_stability_heatmap(stability_series: pd.Series, save_path: str = "figures/stability_heatmap.png"):
    """Plot agreement heatmap between Random Forest and XGBoost rankings."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(9, 3))
    sns.heatmap(stability_series.to_frame(name='RF+XGB Agreement').T,
                annot=True, cmap='Blues', linewidths=0.5, fmt='.1f', vmin=0, vmax=1)
    plt.title("Feature Stability (7 features)\n1.0 = RF+XGB agree | 0.5 = one model | 0.0 = neither")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved '{save_path}'")
