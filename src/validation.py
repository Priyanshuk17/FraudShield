"""UI-2 / ERR-1: validate user-entered transaction values."""
import math
from . import config as C


def validate_transaction(values: dict):
    """Return (clean_values, errors). Prediction must not run if errors is non-empty."""
    errors, clean = [], {}
    for name in C.FEATURES:
        raw = values.get(name)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors.append(f"'{name}' is required.")
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{name}' must be a number (got {raw!r}).")
            continue
        if math.isnan(v) or math.isinf(v):
            errors.append(f"'{name}' must be a finite number.")
            continue
        if name == "Amount":
            if v < 0:
                errors.append("'Amount' cannot be negative.")
                continue
            if v > C.MAX_AMOUNT:
                errors.append(f"'Amount' exceeds the supported maximum ({C.MAX_AMOUNT:,.0f}).")
                continue
        elif abs(v) > C.PCA_ABS_LIMIT:
            errors.append(f"'{name}' is outside the expected range (+/-{C.PCA_ABS_LIMIT:g}).")
            continue
        clean[name] = v
    return clean, errors


def validate_dataframe(df):
    """Validate an uploaded CSV. Returns list of error strings."""
    missing = [c for c in C.FEATURES if c not in df.columns]
    if missing:
        return [f"Missing required columns: {', '.join(missing)}"]
    errs = []
    sub = df[C.FEATURES]
    if sub.isna().any().any():
        errs.append("File contains missing/non-numeric values in required columns.")
    elif (sub["Amount"] < 0).any():
        errs.append("File contains negative Amount values.")
    return errs
