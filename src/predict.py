"""ML-2 / ML-3 / RES-1 / RES-2: prediction service used by the UI."""
import json
import joblib
import pandas as pd
from . import config as C
from .preprocessing import apply_scaler, load_scaler
from .validation import validate_transaction
from .risk import risk_level, label


class FraudPredictor:
    def __init__(self):
        if not C.MODEL_PATH.exists():
            raise FileNotFoundError("Trained model not found. Run: python -m src.train")
        self.model = joblib.load(C.MODEL_PATH)
        self.scaler = load_scaler()
        self.metadata = json.loads(C.METADATA_PATH.read_text()) if C.METADATA_PATH.exists() else {}

    def predict_proba_df(self, df: pd.DataFrame):
        return self.model.predict_proba(apply_scaler(df, self.scaler))[:, 1]

    def predict_one(self, values: dict) -> dict:
        clean, errors = validate_transaction(values)
        if errors:
            return {"ok": False, "errors": errors}
        p = float(self.predict_proba_df(pd.DataFrame([clean]))[0])
        return {"ok": True, "prediction": label(p), "fraud_probability": p,
                "legit_probability": 1 - p, "risk": risk_level(p)}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_proba_df(df)
        out = df.copy()
        out["fraud_probability"] = p
        out["prediction"] = [label(x) for x in p]
        out["risk"] = [risk_level(x) for x in p]
        return out
