from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw telco churn dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the minimal reusable cleaning steps used across notebooks:
    - strip whitespace from object columns
    - convert TotalCharges to numeric
    - drop rows where TotalCharges cannot be parsed
    """
    df = df.copy()

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()

    return df


def load_and_clean_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Convenience wrapper to load and clean the dataset.
    """
    df = load_raw_data(path)
    return clean_data(df)