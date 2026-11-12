"""
Unit and Integration Tests for Intrusion Detection Pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import STABLE_FEATURES, DDOS_BURST_THRESHOLD, DDOS_PACKETS_THRESHOLD
from src.data_loader import map_multiclass, preprocess_and_clean
from src.pipeline import ddos_rule_override
from src.utils import verify_split_isolation


def test_split_isolation_clean():
    """Verify that verify_split_isolation succeeds when train and test groups are disjoint."""
    train_df = pd.DataFrame({'time_bucket': [100, 100, 101, 102], 'val': [1, 2, 3, 4]})
    test_df = pd.DataFrame({'time_bucket': [200, 201, 202], 'val': [5, 6, 7]})
    audit = verify_split_isolation(train_df, test_df, group_col='time_bucket', audit_path=None)
    assert audit['isolation_verified'] is True
    assert audit['overlap_count'] == 0
    assert audit['leakage_percentage'] == 0.0


def test_split_isolation_leaky_raises_error():
    """Verify that verify_split_isolation raises ValueError when groups overlap."""
    train_df = pd.DataFrame({'time_bucket': [100, 101, 102], 'val': [1, 2, 3]})
    test_df = pd.DataFrame({'time_bucket': [100, 200, 201], 'val': [4, 5, 6]})
    with pytest.raises(ValueError) as excinfo:
        verify_split_isolation(train_df, test_df, group_col='time_bucket', audit_path=None)
    assert "DATA LEAKAGE DETECTED" in str(excinfo.value)


def test_label_mapping():
    """Verify multiclass label standardization."""
    assert map_multiclass('BENIGN') == 'BENIGN'
    assert map_multiclass('0') == 'BENIGN'
    assert map_multiclass('DDoS') == 'DDOS'
    assert map_multiclass('DoS Hulk') == 'DDOS'
    assert map_multiclass('FTP-Patator') == 'FTP-BRUTEFORCE'
    assert map_multiclass('SSH-Patator') == 'SSH-BRUTEFORCE'
    assert map_multiclass('RandomAttack') == 'UNKNOWN'


def test_feature_engineering():
    """Verify calculation of burst_ratio and byte_per_packet."""
    df = pd.DataFrame({
        'uni_flow_duration': [1000],
        'uni_fwd_packets': [10],
        'uni_bwd_packets': [2],
        'uni_fwd_bytes': [1000],
        'uni_bwd_bytes': [200],
        'uni_label': ['BENIGN']
    })
    cleaned = preprocess_and_clean(df)
    assert 'burst_ratio' in cleaned.columns
    assert 'byte_per_packet' in cleaned.columns
    # burst_ratio = 10 / (2 + 1e-6) ≈ 5.0
    assert np.isclose(cleaned['burst_ratio'].iloc[0], 5.0, atol=1e-3)
    # byte_per_packet = 1000 / (10 + 1e-6) ≈ 100.0
    assert np.isclose(cleaned['byte_per_packet'].iloc[0], 100.0, atol=1e-3)


def test_ddos_rule_override():
    """Verify deterministic DDoS detection rule conditions."""
    # DDoS attack flow: burst_ratio >> 10 and packets >= 4
    ddos_flow = pd.Series({'burst_ratio': 50.0, 'uni_fwd_packets': 5})
    assert ddos_rule_override(ddos_flow, burst_thresh=10.0, pkt_thresh=4.0) is True

    # Normal symmetric flow
    normal_flow = pd.Series({'burst_ratio': 1.0, 'uni_fwd_packets': 5})
    assert ddos_rule_override(normal_flow, burst_thresh=10.0, pkt_thresh=4.0) is False

    # Single-packet probe with high burst ratio
    probe_flow = pd.Series({'burst_ratio': 100.0, 'uni_fwd_packets': 1})
    assert ddos_rule_override(probe_flow, burst_thresh=10.0, pkt_thresh=4.0) is False


def test_model_artifact_load():
    """Verify serialized model artifact contains all expected components."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ids_cic_dual_model_v5.pkl")
    if not os.path.exists(model_path):
        pytest.skip("Model artifact not yet generated.")

    artifact = joblib.load(model_path)
    assert 'model' in artifact
    assert 'encoder' in artifact
    assert 'scaler' in artifact
    assert 'anomaly_detector' in artifact
    assert 'features' in artifact
    assert artifact['features'] == STABLE_FEATURES
