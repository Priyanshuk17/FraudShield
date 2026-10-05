"""Central configuration (supports requirement DS-1 / modular design)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
SAMPLE_DIR = ROOT / "sample_data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
SCALER_PATH = MODEL_DIR / "amount_scaler.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
METRICS_PATH = REPORT_DIR / "metrics.json"

TARGET = "Class"
DROP_COLUMNS = ["Time"]            # Time is seconds-since-first-txn: not a usable input
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = PCA_FEATURES + ["Amount"]   # order expected by the model

RANDOM_STATE = 42
TEST_SIZE = 0.2

# RES-2: risk bands applied on fraud probability
RISK_LOW_MAX = 0.30
RISK_HIGH_MIN = 0.70
DECISION_THRESHOLD = 0.50

# Validation limits (UI-2 / ERR-1)
MAX_AMOUNT = 1_000_000.0
PCA_ABS_LIMIT = 100.0
