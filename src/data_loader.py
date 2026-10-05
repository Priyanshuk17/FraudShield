"""DS-1: Dataset management - load, check and split the dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config as C


def load_dataset(path=C.DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place 'creditcard.csv' in the data/ folder.")
    df = pd.read_csv(path)
    missing = set(C.FEATURES + [C.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    n_fraud = int((df[C.TARGET] == 1).sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "fraud": n_fraud,
        "legitimate": int(len(df) - n_fraud),
        "fraud_percent": round(100 * n_fraud / len(df), 4),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def split_data(df: pd.DataFrame):
    """Stratified split so the rare fraud class appears in both subsets."""
    X, y = df[C.FEATURES], df[C.TARGET]
    return train_test_split(X, y, test_size=C.TEST_SIZE,
                            stratify=y, random_state=C.RANDOM_STATE)
