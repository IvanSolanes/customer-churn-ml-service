"""
Central configuration for MLflow tracking and model registry.

All modules import from here so MLflow settings are never hardcoded
in multiple places.

The tracking URI defaults to a local SQLite database, which supports
both experiment tracking and the Model Registry out of the box.
Set the MLFLOW_TRACKING_URI environment variable to point at a remote
MLflow server when moving to a shared or production environment.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# MLflow settings
# ---------------------------------------------------------------------------

# Local SQLite backend — supports both Tracking and Model Registry.
# Override with: export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)

# All training runs are grouped under this experiment name.
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "churn-prediction",
)

# Registered model name in the Model Registry.
MLFLOW_MODEL_NAME: str = os.getenv(
    "MLFLOW_MODEL_NAME",
    "churn-predictor",
)

# Alias to load at inference time.
# The training pipeline assigns this alias to the latest trained version.
# Override with: export MLFLOW_MODEL_ALIAS=challenger
MLFLOW_MODEL_ALIAS: str = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)
