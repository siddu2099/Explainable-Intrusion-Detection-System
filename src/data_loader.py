"""
Data Loading, Feature Engineering, and GroupShuffleSplit
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.config import (
    DICT_2017, DICT_2018, STABLE_FEATURES, SAMPLE_FRAC,
    RANDOM_STATE, TEST_SIZE, EPS
)

_TS_CANDIDATES = [
    'timestamp', 'flow_start_time', 'flow start', 'flow_start',
    'start_time', 'starttime'
]


def extract_time_bucket(df_raw: pd.DataFrame) -> pd.Series:
    """
    Parse the timestamp column (if present) into minute-level integer Unix timestamps
    for use as GroupShuffleSplit groups. Falls back to synthetic index-based buckets
    if no timestamp column is available.
    """
    raw_cols_lower = {c.strip().lower(): c for c in df_raw.columns}
    ts_col_key = None
    for candidate in _TS_CANDIDATES:
        if candidate in raw_cols_lower:
            ts_col_key = raw_cols_lower[candidate]
            break

    if ts_col_key is not None:
        try:
            ts_series = pd.to_datetime(
                df_raw[ts_col_key],
                format='mixed',
                dayfirst=False,
                errors='coerce'
            )
            n_nulls = ts_series.isna().sum()
            if n_nulls > 0:
                print(f"     WARNING: {n_nulls} unparseable timestamps — propagating via ffill/bfill.")
                ts_series = ts_series.ffill().bfill()

            ts_minute = (ts_series.astype('int64') // (60 * 10**9)).astype('int32')
            print(f"     time_bucket: parsed from '{ts_col_key}' — {ts_minute.nunique()} unique minute-buckets.")
            return ts_minute
        except Exception as e:
            print(f"     WARNING: Timestamp parse failed ({e}). Falling back to synthetic buckets.")

    # Fallback to synthetic index-based equal-width bins
    n = len(df_raw)
    synthetic = pd.Series(
        (np.arange(n) // max(1, n // 500)).astype('int32'),
        index=df_raw.index,
        name='time_bucket'
    )
    print(f"     time_bucket: synthetic — {synthetic.nunique()} buckets.")
    return synthetic


def load_and_align(
    file_path: str,
    rename_dict: dict,
    dataset_name: str,
    sample_frac: float = SAMPLE_FRAC,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Load a raw dataset, extract time_bucket before column truncation, rename
    features to unified names, and subsample for balanced training.
    """
    print(f"\n  -> Loading {dataset_name} from '{file_path}'...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"     ERROR loading {file_path}: {e}")
        return None

    # Step 1: Extract time_bucket before column truncation
    time_bucket_series = extract_time_bucket(df)

    # Step 2: Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    df['time_bucket'] = time_bucket_series.values

    # Step 3: Align feature column names
    available = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=available)
    keep = [v for v in rename_dict.values() if v in df.columns] + ['time_bucket']
    missing = [v for v in rename_dict.values() if v not in df.columns and v != 'uni_label']
    if missing:
        raise ValueError(f"[{dataset_name}] Missing required features: {missing}")

    df = df[keep].sample(frac=sample_frac, random_state=random_state)
    print(f"     Loaded shape: {df.shape} (includes time_bucket group key)")
    return df


def map_multiclass(label: str) -> str:
    """Standardize multiclass attack labels."""
    label = str(label).strip().upper()
    if label in ['BENIGN', 'NORMAL', '0', '0.0']:
        return 'BENIGN'
    elif 'DDOS' in label or 'DOS' in label:
        return 'DDOS'
    elif 'FTP' in label:
        return 'FTP-BRUTEFORCE'
    elif 'SSH' in label:
        return 'SSH-BRUTEFORCE'
    else:
        return 'UNKNOWN'


def preprocess_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numerical features, clip out-of-bounds values, compute engineered features,
    and map multiclass labels.
    """
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in df.columns:
        if col not in ('uni_label', 'time_bucket'):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Physical clipping
    df['uni_flow_duration'] = np.clip(df['uni_flow_duration'], 0, None)
    df['uni_fwd_bytes']     = np.clip(df['uni_fwd_bytes'],     0, None)
    df['uni_bwd_bytes']     = np.clip(df['uni_bwd_bytes'],     0, None)
    df['uni_fwd_packets']   = np.clip(df['uni_fwd_packets'],   0, None)
    df['uni_bwd_packets']   = np.clip(df['uni_bwd_packets'],   0, None)

    # Feature Engineering
    # burst_ratio: FWD/BWD packet asymmetry; >> 10 during floods, ≈ 1 in normal traffic
    df['burst_ratio'] = df['uni_fwd_packets'] / (df['uni_bwd_packets'] + EPS)

    # byte_per_packet: payload density; separates large FTP blobs from tiny SSH packets
    df['byte_per_packet'] = df['uni_fwd_bytes'] / (df['uni_fwd_packets'] + EPS)

    # Label Mapping
    df['Target_Label'] = df['uni_label'].apply(map_multiclass)
    df = df.drop(columns=['uni_label'])
    df = df[df['Target_Label'] != 'UNKNOWN']
    return df


def perform_group_split(
    df_all: pd.DataFrame,
    features: list = STABLE_FEATURES,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
):
    """
    Execute GroupShuffleSplit partitioned on time_bucket to guarantee
    zero temporal/contextual leakage between train and test distributions.
    """
    X_all = df_all[[f for f in features if f in df_all.columns]]
    y_all = df_all['Target_Label']
    groups = df_all['time_bucket']

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups))

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train     = y_all.iloc[train_idx].reset_index(drop=True)
    y_test      = y_all.iloc[test_idx].reset_index(drop=True)

    groups_train = groups.iloc[train_idx].reset_index(drop=True).rename('time_bucket')
    groups_test  = groups.iloc[test_idx].reset_index(drop=True).rename('time_bucket')

    return X_train_raw, X_test_raw, y_train, y_test, groups_train, groups_test
