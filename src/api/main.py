"""
Customer Churn Prediction API

Endpoints:
  GET  /              → redirect to /docs
  GET  /health        → service health check
  POST /predict       → single-customer churn prediction
  POST /predict/batch → batch prediction (up to 5,000 customers)

The model is loaded from the MLflow Model Registry (Production stage).
"""
from __future__ import annotations

from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, model_validator

from src.config import (
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Churn ML Service",
    description=(
        "Predicts the probability that a telecom customer will churn. "
        "The model is a calibrated sklearn pipeline trained on the Telco "
        "Customer Churn dataset and served from the MLflow Model Registry."
    ),
    version="0.1.0",
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Model loading (lazy singleton from MLflow Registry)
# ---------------------------------------------------------------------------

_model = None


def get_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
        try:
            _model = mlflow.sklearn.load_model(model_uri)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model '{MLFLOW_MODEL_NAME}' "
                f"with alias '@{MLFLOW_MODEL_ALIAS}' "
                f"from '{MLFLOW_TRACKING_URI}'.\n"
                "Run the training pipeline first:\n"
                "  python -m src.training.train_model"
            ) from exc
    return _model


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """Raw customer attributes, matching Telco dataset column names exactly."""

    gender: Literal["Male", "Female"] = Field(..., example="Male")
    SeniorCitizen: Literal[0, 1] = Field(..., example=0)
    Partner: Literal["Yes", "No"] = Field(..., example="Yes")
    Dependents: Literal["Yes", "No"] = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=72, example=12)
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., example="Month-to-month"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(..., example="Yes")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=844.20)
    PhoneService: Literal["Yes", "No"] = Field(..., example="Yes")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., example="No")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., example="Fiber optic"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., example="No"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., example="No"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., example="No"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., example="No"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., example="Yes"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., example="Yes"
    )

    @model_validator(mode="after")
    def total_charges_ge_monthly(self) -> "CustomerFeatures":
        if self.TotalCharges < self.MonthlyCharges and self.tenure > 0:
            raise ValueError(
                "TotalCharges cannot be less than MonthlyCharges "
                "for a customer with tenure > 0."
            )
        return self


class PredictionResponse(BaseModel):
    churn_prediction: Literal[0, 1] = Field(
        ..., description="1 = predicted to churn, 0 = predicted to stay"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated probability of churn"
    )
    risk_tier: Literal["Low", "Medium", "High"] = Field(
        ..., description="Business-friendly risk bucket"
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    n_customers: int
    n_predicted_churn: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def _is_active_service(value: object) -> int:
    return int(str(value).strip() == "Yes")


def _build_features_for_inference(customers: list[CustomerFeatures]) -> pd.DataFrame:
    rows = [c.model_dump() for c in customers]
    X = pd.DataFrame(rows)
    X["is_month_to_month"]     = (X["Contract"] == "Month-to-month").astype(int)
    X["is_new_customer"]       = (X["tenure"] <= 12).astype(int)
    X["uses_electronic_check"] = (X["PaymentMethod"] == "Electronic check").astype(int)
    X["has_fiber_optic"]       = (X["InternetService"] == "Fiber optic").astype(int)
    X["service_count"]         = X[SERVICE_COLS].map(_is_active_service).sum(axis=1)
    X["fiber_month_to_month"]  = X["has_fiber_optic"] * X["is_month_to_month"]
    return X


def _risk_tier(probability: float) -> Literal["Low", "Medium", "High"]:
    if probability < 0.35:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


def _run_inference(customers: list[CustomerFeatures]) -> list[PredictionResponse]:
    model = get_model()
    X = _build_features_for_inference(customers)
    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return [
        PredictionResponse(
            churn_prediction=int(pred),
            churn_probability=round(float(prob), 4),
            risk_tier=_risk_tier(float(prob)),
        )
        for pred, prob in zip(predictions, probabilities)
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Ops"])
def health() -> dict:
    """
    Returns service status and whether a Production model exists
    in the MLflow Model Registry.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        mv = client.get_model_version_by_alias(
            name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_MODEL_ALIAS,
        )
        model_ready   = True
        model_version = mv.version
    except Exception:
        model_ready   = False
        model_version = None

    return {
        "status":        "ok" if model_ready else "degraded",
        "model_name":    MLFLOW_MODEL_NAME,
        "model_alias":   MLFLOW_MODEL_ALIAS,
        "model_version": model_version,
        "model_uri":     model_uri,
        "tracking_uri":  MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """
    Predict churn for a single customer.

    Returns the binary prediction, a calibrated churn probability,
    and a business-friendly risk tier (Low / Medium / High).
    """
    try:
        return _run_inference([customer])[0]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(customers: list[CustomerFeatures]) -> BatchPredictionResponse:
    """
    Predict churn for a list of customers in a single request.

    Useful for scoring cohorts or targeting retention campaigns.
    Maximum batch size: 5,000 customers.
    """
    if not customers:
        raise HTTPException(status_code=422, detail="Customer list cannot be empty.")
    if len(customers) > 5000:
        raise HTTPException(
            status_code=422,
            detail=(
                "Batch size exceeds the limit of 5,000. "
                "Use the batch scoring script for larger inputs."
            ),
        )
    try:
        results = _run_inference(customers)
        return BatchPredictionResponse(
            predictions=results,
            n_customers=len(results),
            n_predicted_churn=sum(r.churn_prediction for r in results),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
