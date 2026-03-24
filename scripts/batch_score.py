"""
Batch scoring script.

Reads a CSV of customer features, runs inference using the saved model artifact,
and writes predictions to a new CSV.

Usage
-----
    python scripts/batch_score.py \
        --input  data/raw/Telco-Customer-Churn.csv \
        --output data/scored/scored_customers.csv

The input CSV must contain the same columns as the Telco training dataset.
The Churn column is ignored if present, so the script works on both
labeled and unlabeled data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "best_model.joblib"

SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run the training pipeline first:\n"
            "  python -m src.training.train_model"
        )
    return joblib.load(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the minimal cleaning from src/data/load_data.py."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"]).copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} row(s) with unparseable TotalCharges.")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate feature engineering from src/features/build_features.py."""
    drop_cols = [c for c in ["customerID", "Churn"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()

    X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
    X["is_new_customer"] = (X["tenure"] <= 12).astype(int)
    X["uses_electronic_check"] = (X["PaymentMethod"] == "Electronic check").astype(int)
    X["has_fiber_optic"] = (X["InternetService"] == "Fiber optic").astype(int)
    X["service_count"] = X[SERVICE_COLS].applymap(lambda v: int(str(v).strip() == "Yes")).sum(axis=1)
    X["fiber_month_to_month"] = X["has_fiber_optic"] * X["is_month_to_month"]

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
    print(f"Loading input: {input_path}")
    raw = pd.read_csv(input_path)
    print(f"  {len(raw):,} rows loaded.")

    df = clean(raw)
    X = build_features(df)

    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Attach results to original (cleaned) rows so IDs are preserved
    output = df[["customerID"]].copy() if "customerID" in df.columns else df.iloc[:, :0].copy()
    output["churn_prediction"] = predictions
    output["churn_probability"] = probabilities.round(4)
    output["risk_tier"] = [risk_tier(p) for p in probabilities]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    n_churn = int(predictions.sum())
    print(f"\nScoring complete.")
    print(f"  Customers scored : {len(output):,}")
    print(f"  Predicted churn  : {n_churn:,} ({n_churn / len(output):.1%})")
    print(f"  Risk tiers       : {output['risk_tier'].value_counts().to_dict()}")
    print(f"  Output saved to  : {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch churn scoring script.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",
        help="Path to input CSV (default: data/raw/Telco-Customer-Churn.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "scored" / "scored_customers.csv",
        help="Path for output CSV (default: data/scored/scored_customers.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score(args.input, args.output)
