"""RES-2: risk classification logic."""
from . import config as C


def risk_level(prob: float) -> str:
    if prob < C.RISK_LOW_MAX:
        return "Low Risk"
    if prob < C.RISK_HIGH_MIN:
        return "Medium Risk"
    return "High Risk"


def label(prob: float) -> str:
    return "FRAUDULENT" if prob >= C.DECISION_THRESHOLD else "LEGITIMATE"
