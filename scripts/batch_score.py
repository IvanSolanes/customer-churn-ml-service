"""
Batch scoring script — loads the Production model from the MLflow Registry.

Usage
-----
    python scripts/batch_score.py \
        --input  data/raw/Telco-Customer-Churn.csv \
        --output data/scored/scored_customers.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from src.config import (
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model():
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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"]).copy()
    dropped = n_before - len(df)
    if dropped:
        print(f"  Dropped {dropped} row(s) with unparseable TotalCharges.")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["customerID", "Churn"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    X["is_month_to_month"]     = (X["Contract"] == "Month-to-month").astype(int)
    X["is_new_customer"]       = (X["tenure"] <= 12).astype(int)
    X["uses_electronic_check"] = (X["PaymentMethod"] == "Electronic check").astype(int)
    X["has_fiber_optic"]       = (X["InternetService"] == "Fiber optic").astype(int)
    X["service_count"]         = X[SERVICE_COLS].applymap(
        lambda v: int(str(v).strip() == "Yes")
    ).sum(axis=1)
    X["fiber_month_to_month"]  = X["has_fiber_optic"] * X["is_month_to_month"]
    return X


def risk_tier(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.60:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score(input_path: Path, output_path: Path) -> None:
    print(f"Loading input  : {input_path}")
    raw = pd.read_csv(input_path)
    print(f"  {len(raw):,} rows loaded.")

    df = clean(raw)
    X  = build_features(df)

    print(f"Loading model  : {MLFLOW_MODEL_NAME} @{MLFLOW_MODEL_ALIAS}")
    print(f"Tracking URI   : {MLFLOW_TRACKING_URI}")
    model = load_model()

    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    output = df[["customerID"]].copy() if "customerID" in df.columns else pd.DataFrame()
    output["churn_prediction"]  = predictions
    output["churn_probability"] = probabilities.round(4)
    output["risk_tier"]         = [risk_tier(p) for p in probabilities]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    n_churn = int(predictions.sum())
    print(f"\nScoring complete.")
    print(f"  Customers scored  : {len(output):,}")
    print(f"  Predicted churn   : {n_churn:,} ({n_churn / len(output):.1%})")
    print(f"  Risk distribution : {output['risk_tier'].value_counts().to_dict()}")
    print(f"  Output saved to   : {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch churn scoring script.")
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "scored" / "scored_customers.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score(args.input, args.output)
