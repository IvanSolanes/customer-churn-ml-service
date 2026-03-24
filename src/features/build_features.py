from __future__ import annotations

import pandas as pd


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


def is_active_service(value: object) -> int:
    return int(str(value).strip() == "Yes")


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix X and target y from the cleaned dataset.
    """
    df = df.copy()

    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found.")

    X = df.drop(columns=["customerID", "Churn"]).copy()
    y = (df["Churn"] == "Yes").astype(int)

    # Business-motivated engineered features
    X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
    X["is_new_customer"] = (X["tenure"] <= 12).astype(int)
    X["uses_electronic_check"] = (X["PaymentMethod"] == "Electronic check").astype(int)
    X["has_fiber_optic"] = (X["InternetService"] == "Fiber optic").astype(int)

    X["service_count"] = X[SERVICE_COLS].map(is_active_service).sum(axis=1)

    X["fiber_month_to_month"] = X["has_fiber_optic"] * X["is_month_to_month"]

    return X, y


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return numeric and categorical feature lists.
    """
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    return numeric_features, categorical_features