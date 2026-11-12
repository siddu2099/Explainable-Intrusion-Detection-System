"""
Utility Functions and Audit Verification
"""

import os
import json
import datetime
import tempfile
import warnings
import pandas as pd


def setup_environment():
    """
    Configure environment variables and matplotlib backend for reliable,
    non-blocking headless operation across all platforms.
    """
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)


def verify_split_isolation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "time_bucket",
    audit_path: str = "reports/experiment_audit.json",
    split_strategy: str = "GroupShuffleSplit",
    random_seed: int = 42,
) -> dict:
    """
    Verify that no group from `group_col` appears in both train_df and test_df.
    Raises ValueError immediately if any overlap is detected (leakage > 0).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition.
    test_df : pd.DataFrame
        Test partition.
    group_col : str
        Column name used as grouping key (e.g. 'time_bucket').
    audit_path : str
        Output file path for the audit record.
    split_strategy : str
        Identifier of the splitting strategy used.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    audit : dict
        Audit dictionary written to audit_path.
    """
    if group_col not in train_df.columns:
        raise KeyError(f"Column '{group_col}' not found in train_df. Available: {list(train_df.columns)}")
    if group_col not in test_df.columns:
        raise KeyError(f"Column '{group_col}' not found in test_df. Available: {list(test_df.columns)}")

    train_groups = set(train_df[group_col].unique())
    test_groups = set(test_df[group_col].unique())
    overlap = train_groups & test_groups

    total_unique = len(train_groups | test_groups)
    n_train = len(train_groups)
    n_test = len(test_groups)
    n_overlap = len(overlap)

    leakage_pct = (n_overlap / total_unique * 100) if total_unique > 0 else 0.0

    audit = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'random_seed': random_seed,
        'split_strategy': split_strategy,
        'grouping_variable': group_col,
        'train_rows': int(len(train_df)),
        'test_rows': int(len(test_df)),
        'train_groups': int(n_train),
        'test_groups': int(n_test),
        'total_unique_groups': int(total_unique),
        'overlap_count': int(n_overlap),
        'leakage_percentage': round(leakage_pct, 6),
        'isolation_verified': leakage_pct == 0.0,
        'overlapping_groups_sample': sorted(str(g) for g in list(overlap)[:20]) if n_overlap > 0 else [],
    }

    if audit_path:
        os.makedirs(os.path.dirname(os.path.abspath(audit_path)), exist_ok=True)
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(audit, f, indent=2)

    print("─" * 60)
    print("SPLIT ISOLATION VERIFICATION REPORT")
    print("─" * 60)
    print(f"  Group column       : {group_col}")
    print(f"  Total rows (train) : {len(train_df)}")
    print(f"  Total rows (test)  : {len(test_df)}")
    print(f"  Unique groups train: {n_train}")
    print(f"  Unique groups test : {n_test}")
    print(f"  Total unique groups: {total_unique}")
    print(f"  Overlapping groups : {n_overlap}")
    print(f"  Leakage percentage : {leakage_pct:.4f}%")
    print(f"  Audit saved to     : {audit_path}")

    if n_overlap > 0:
        raise ValueError(
            f"[verify_split_isolation] DATA LEAKAGE DETECTED: {n_overlap} group(s) "
            f"appear in both train and test partitions ({leakage_pct:.4f}% leakage)."
        )

    print(f"  ✓ ISOLATION VERIFIED — 0 groups overlap between train and test.")
    print("─" * 60)
    return audit
