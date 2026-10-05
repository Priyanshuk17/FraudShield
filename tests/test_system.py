"""Test cases (IDs follow <MODULE>-TC<n> convention used in the PRD)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
import pytest
from src import config as C
from src.validation import validate_transaction, validate_dataframe
from src.risk import risk_level, label


def valid_txn():
    d = {f: 0.0 for f in C.PCA_FEATURES}; d["Amount"] = 100.0
    return d


# ---- UI-2 / ERR-1: validation
def test_UI2_TC1_valid_input_accepted():
    clean, errs = validate_transaction(valid_txn())
    assert errs == [] and len(clean) == 29

def test_UI2_TC2_missing_value_rejected():
    d = valid_txn(); del d["V5"]
    assert any("V5" in e for e in validate_transaction(d)[1])

def test_UI2_TC3_non_numeric_rejected():
    d = valid_txn(); d["Amount"] = "abc"
    assert validate_transaction(d)[1]

def test_ERR1_TC1_negative_amount_rejected():
    d = valid_txn(); d["Amount"] = -5
    assert validate_transaction(d)[1]

def test_ERR1_TC2_nan_and_inf_rejected():
    d = valid_txn(); d["V1"] = float("nan"); d["V2"] = float("inf")
    assert len(validate_transaction(d)[1]) == 2

def test_ERR1_TC3_out_of_range_rejected():
    d = valid_txn(); d["V3"] = 1e6
    assert validate_transaction(d)[1]

def test_ERR1_TC4_csv_missing_columns():
    assert validate_dataframe(pd.DataFrame({"Amount": [1]}))

# ---- RES-2: risk logic
@pytest.mark.parametrize("p,r", [(0.0, "Low Risk"), (0.29, "Low Risk"), (0.30, "Medium Risk"),
                                 (0.69, "Medium Risk"), (0.70, "High Risk"), (1.0, "High Risk")])
def test_RES2_TC1_risk_bands(p, r):
    assert risk_level(p) == r

def test_RES1_TC1_label():
    assert label(0.5) == "FRAUDULENT" and label(0.49) == "LEGITIMATE"

# ---- ML-1/2/3/4: need trained model
needs_model = pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="model not trained")

@needs_model
def test_ML2_TC1_prediction_runs_and_ML3_probability_valid():
    from src.predict import FraudPredictor
    r = FraudPredictor().predict_one(valid_txn())
    assert r["ok"] and r["prediction"] in ("FRAUDULENT", "LEGITIMATE")
    assert 0.0 <= r["fraud_probability"] <= 1.0
    assert abs(r["fraud_probability"] + r["legit_probability"] - 1) < 1e-9

@needs_model
def test_ML2_TC2_invalid_input_blocks_prediction():
    from src.predict import FraudPredictor
    d = valid_txn(); d["Amount"] = -1
    r = FraudPredictor().predict_one(d)
    assert r["ok"] is False and r["errors"]

@needs_model
def test_ML2_TC3_known_samples_classified():
    from src.predict import FraudPredictor
    s = pd.read_csv(C.SAMPLE_DIR / "sample_transactions.csv")
    out = FraudPredictor().predict_batch(s)
    fraud_recall = (out[s[C.TARGET] == 1]["prediction"] == "FRAUDULENT").mean()
    legit_ok = (out[s[C.TARGET] == 0]["prediction"] == "LEGITIMATE").mean()
    assert fraud_recall >= 0.6 and legit_ok >= 0.9

@needs_model
def test_ML4_TC1_metrics_file_has_required_metrics():
    import json
    m = json.loads(C.METRICS_PATH.read_text())
    r = m["test_results"][m["selected_model"]]
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]:
        assert k in r
    assert r["roc_auc"] > 0.9

@needs_model
def test_PRE1_TC1_scaler_applied_consistently():
    from src.preprocessing import load_scaler, apply_scaler
    sc = load_scaler(); df = pd.DataFrame([valid_txn()])
    a = apply_scaler(df, sc); b = apply_scaler(df, sc)
    assert a.equals(b) and list(a.columns) == C.FEATURES
