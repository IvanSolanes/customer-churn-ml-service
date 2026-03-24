"""
Inference module — loads the Production model from the MLflow Model Registry.

The model URI follows the MLflow convention:
    models:/<model-name>/<stage>

This means load_model() always loads whatever model is currently in the
Production stage, regardless of which training run produced it. Promoting
a new version in the registry is all that's needed to update inference
without changing any code.
"""
from __future__ import annotations

import mlflow.sklearn
import pandas as pd

from src.config import (
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)


def load_model():
    """
    Load the champion model from the MLflow Model Registry.

    The URI models:/<name>@<alias> always resolves to whichever version
    has been assigned the alias — no code change needed to roll out a
    new model version.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model '{MLFLOW_MODEL_NAME}' "
            f"with alias '@{MLFLOW_MODEL_ALIAS}' "
            f"from '{MLFLOW_TRACKING_URI}'.\n"
            "Run the training pipeline first:\n"
            "  python -m src.training.train_model"
        ) from exc


def predict_dataframe(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate class predictions and churn probabilities
    from a prepared feature DataFrame.
    """
    model = load_model()

    predictions   = model.predict(df_features)
    probabilities = model.predict_proba(df_features)[:, 1]

    output = df_features.copy()
    output["prediction"]        = predictions
    output["churn_probability"] = probabilities

    return output
