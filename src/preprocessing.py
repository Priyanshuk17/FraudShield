"""PRE-1: Data preprocessing, shared by training and prediction."""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from . import config as C


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, missing rows and the unusable Time column."""
    df = df.drop(columns=[c for c in C.DROP_COLUMNS if c in df.columns])
    return df.drop_duplicates().dropna().reset_index(drop=True)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit the Amount scaler on TRAINING data only (avoids data leakage)."""
    scaler = StandardScaler()
    scaler.fit(X_train[["Amount"]])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    X = X[C.FEATURES].copy()
    X["Amount"] = scaler.transform(X[["Amount"]]).ravel()
    return X


def save_scaler(scaler, path=C.SCALER_PATH):
    joblib.dump(scaler, path)


def load_scaler(path=C.SCALER_PATH):
    return joblib.load(path)
