from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


MODEL_PATH = Path("artifacts/models/best_model.joblib")


def load_model(path: str | Path = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_dataframe(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate class predictions and probabilities from a prepared feature dataframe.
    """
    model = load_model()

    predictions = model.predict(df_features)
    probabilities = model.predict_proba(df_features)[:, 1]

    output = df_features.copy()
    output["prediction"] = predictions
    output["churn_probability"] = probabilities

    return output