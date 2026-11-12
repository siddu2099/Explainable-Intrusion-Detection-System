"""
Global Configuration and Constants
"""

import os

# ── Directory Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Default Dataset File Paths
PATH_2017 = os.path.join(DATA_DIR, "CIC-IDS-2017", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
PATH_2018 = os.path.join(DATA_DIR, "CIC-IDS-2018", "02-14-2018.csv")

# ── Column Alignment Mappings ───────────────────────────────────────────────
DICT_2017 = {
    'flow duration':               'uni_flow_duration',
    'total fwd packets':           'uni_fwd_packets',
    'total backward packets':      'uni_bwd_packets',
    'total length of fwd packets': 'uni_fwd_bytes',
    'total length of bwd packets': 'uni_bwd_bytes',
    'label':                       'uni_label',
}

DICT_2018 = {
    'flow duration':   'uni_flow_duration',
    'tot fwd pkts':    'uni_fwd_packets',
    'tot bwd pkts':    'uni_bwd_packets',
    'totlen fwd pkts': 'uni_fwd_bytes',
    'totlen bwd pkts': 'uni_bwd_bytes',
    'label':           'uni_label',
}

# ── Feature Set ─────────────────────────────────────────────────────────────
# 7 Selected and Engineered Features
STABLE_FEATURES = [
    'uni_flow_duration',   # Flow lifetime — persistence/exfiltration indicator
    'uni_fwd_packets',     # Forward packet count — flood/scan volume
    'uni_bwd_packets',     # Backward packet count — near-zero during DDoS floods
    'uni_fwd_bytes',       # Forward payload bytes — large=FTP, small=SSH
    'uni_bwd_bytes',       # Backward response bytes — asymmetry signals attack
    'burst_ratio',         # Engineered: FWD/BWD ratio — DDoS >> 10, normal ≈ 1
    'byte_per_packet',     # Engineered: bytes/pkt — FTP >> SSH
]

FEATURE_JUSTIFICATION = {
    'uni_flow_duration' : "Raw — flow lifetime; persistence/exfil indicator",
    'uni_fwd_packets'   : "Raw — FWD packet count; high = flood/scan volume",
    'uni_bwd_packets'   : "Raw — BWD count; near-zero in DDoS (server overwhelmed)",
    'uni_fwd_bytes'     : "Raw — FWD payload; large=FTP file transfer, small=SSH auth",
    'uni_bwd_bytes'     : "Raw — BWD payload; asymmetry with FWD = attack signal",
    'burst_ratio'       : "Engineered — FWD/BWD ratio; DDoS >> 10, normal ≈ 1, FTP ≈ 2-5",
    'byte_per_packet'   : "Engineered — bytes/pkt; FTP sends large blobs, SSH tiny auth pkts",
}

# ── Hyperparameters & Thresholds ───────────────────────────────────────────
SAMPLE_FRAC = 0.10
RANDOM_STATE = 42
TEST_SIZE = 0.20
EPS = 1e-6

# DDoS Deterministic Rule Thresholds
DDOS_BURST_THRESHOLD = 10.0
DDOS_PACKETS_THRESHOLD = 4.0
DDOS_WEIGHT_MULTIPLIER = 2.0

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.65
DDOS_CONFIDENCE_THRESHOLD = 0.55

# Isolation Forest Settings
ISO_SAMPLE_FRAC = 0.30
ISO_CONTAMINATION = 0.05
