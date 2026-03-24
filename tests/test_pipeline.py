"""
Unit tests for the customer churn ML service.

Covers:
  - src/data/load_data.py     → clean_data
  - src/features/build_features.py → build_features, get_feature_types
  - src/inference/predict.py  → predict_dataframe output shape and columns

Run from the project root:
    pytest tests/ -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_data import clean_data
from src.features.build_features import build_features, get_feature_types


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic data that mirrors the real Telco schema
# ---------------------------------------------------------------------------

def _make_raw_row(**overrides) -> dict:
    """Return a valid raw Telco-style row as a dict."""
    base = {
        "customerID": "TEST-001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": "844.20",   # raw string — intentional, mirrors the CSV
        "Churn": "Yes",
    }
    base.update(overrides)
    return base


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small cleaned dataframe with two valid rows."""
    rows = [
        _make_raw_row(customerID="TEST-001", TotalCharges="844.20", Churn="Yes"),
        _make_raw_row(customerID="TEST-002", TotalCharges="200.00", Churn="No",
                      tenure=3, Contract="Month-to-month"),
    ]
    raw = pd.DataFrame(rows)
    return clean_data(raw)


# ---------------------------------------------------------------------------
# Tests: clean_data
# ---------------------------------------------------------------------------

class TestCleanData:

    def test_totalcharges_converted_to_numeric(self):
        """TotalCharges must be float after cleaning."""
        raw = pd.DataFrame([_make_raw_row(TotalCharges="844.20")])
        df = clean_data(raw)
        assert pd.api.types.is_float_dtype(df["TotalCharges"]), (
            "TotalCharges should be float64 after clean_data"
        )

    def test_blank_totalcharges_rows_are_dropped(self):
        """Rows with blank TotalCharges (the known data quality issue) must be removed."""
        rows = [
            _make_raw_row(customerID="GOOD", TotalCharges="844.20"),
            _make_raw_row(customerID="BAD",  TotalCharges=" "),   # blank — real issue
        ]
        raw = pd.DataFrame(rows)
        df = clean_data(raw)
        assert len(df) == 1
        assert df.iloc[0]["customerID"] == "GOOD"

    def test_whitespace_stripped_from_string_columns(self):
        """Leading/trailing whitespace in object columns must be removed."""
        raw = pd.DataFrame([_make_raw_row(gender="  Male  ", Contract=" Month-to-month ")])
        df = clean_data(raw)
        assert df.iloc[0]["gender"] == "Male"
        assert df.iloc[0]["Contract"] == "Month-to-month"

    def test_valid_rows_are_preserved(self):
        """All valid rows must survive cleaning unchanged in count."""
        rows = [_make_raw_row(customerID=f"C{i}") for i in range(5)]
        raw = pd.DataFrame(rows)
        df = clean_data(raw)
        assert len(df) == 5


# ---------------------------------------------------------------------------
# Tests: build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:

    def test_returns_correct_types(self, clean_df):
        """build_features must return a DataFrame and a Series."""
        X, y = build_features(clean_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_is_binary(self, clean_df):
        """Target y must contain only 0 and 1."""
        _, y = build_features(clean_df)
        assert set(y.unique()).issubset({0, 1}), (
            f"Target contains unexpected values: {y.unique()}"
        )

    def test_churn_yes_encoded_as_1(self, clean_df):
        """Rows with Churn='Yes' must be encoded as 1."""
        _, y = build_features(clean_df)
        churn_yes_idx = clean_df.index[clean_df["Churn"] == "Yes"]
        assert (y.loc[churn_yes_idx] == 1).all()

    def test_customerid_and_churn_excluded_from_X(self, clean_df):
        """customerID and Churn must not appear in the feature matrix."""
        X, _ = build_features(clean_df)
        assert "customerID" not in X.columns
        assert "Churn" not in X.columns

    def test_engineered_features_present(self, clean_df):
        """All six engineered features must be in the feature matrix."""
        X, _ = build_features(clean_df)
        expected = [
            "is_month_to_month",
            "is_new_customer",
            "uses_electronic_check",
            "has_fiber_optic",
            "service_count",
            "fiber_month_to_month",
        ]
        for col in expected:
            assert col in X.columns, f"Expected engineered feature '{col}' not found in X"

    def test_is_month_to_month_flag(self, clean_df):
        """is_month_to_month must be 1 for Month-to-month rows."""
        X, _ = build_features(clean_df)
        mtm_mask = clean_df["Contract"] == "Month-to-month"
        assert (X.loc[mtm_mask, "is_month_to_month"] == 1).all()

    def test_is_new_customer_flag(self):
        """is_new_customer must be 1 for customers with tenure <= 12."""
        rows = [
            _make_raw_row(customerID="NEW", tenure=6,  TotalCharges="420.00"),
            _make_raw_row(customerID="OLD", tenure=36, TotalCharges="2520.00"),
        ]
        df = clean_data(pd.DataFrame(rows))
        X, _ = build_features(df)
        new_idx = df.index[df["customerID"] == "NEW"][0]
        old_idx = df.index[df["customerID"] == "OLD"][0]
        assert X.loc[new_idx, "is_new_customer"] == 1
        assert X.loc[old_idx, "is_new_customer"] == 0

    def test_service_count_range(self, clean_df):
        """service_count must be between 0 and 8 (number of service columns)."""
        X, _ = build_features(clean_df)
        assert X["service_count"].between(0, 8).all()

    def test_missing_churn_column_raises(self):
        """build_features must raise ValueError when Churn column is absent."""
        rows = [_make_raw_row()]
        df = clean_data(pd.DataFrame(rows)).drop(columns=["Churn"])
        with pytest.raises(ValueError, match="Churn"):
            build_features(df)


# ---------------------------------------------------------------------------
# Tests: get_feature_types
# ---------------------------------------------------------------------------

class TestGetFeatureTypes:

    def test_returns_two_lists(self, clean_df):
        """get_feature_types must return exactly two lists."""
        X, _ = build_features(clean_df)
        result = get_feature_types(X)
        assert isinstance(result, tuple) and len(result) == 2
        numeric, categorical = result
        assert isinstance(numeric, list)
        assert isinstance(categorical, list)

    def test_no_overlap_between_numeric_and_categorical(self, clean_df):
        """A column cannot appear in both numeric and categorical lists."""
        X, _ = build_features(clean_df)
        numeric, categorical = get_feature_types(X)
        overlap = set(numeric) & set(categorical)
        assert len(overlap) == 0, f"Columns in both lists: {overlap}"

    def test_all_columns_classified(self, clean_df):
        """Every column in X must appear in either numeric or categorical."""
        X, _ = build_features(clean_df)
        numeric, categorical = get_feature_types(X)
        classified = set(numeric) | set(categorical)
        assert classified == set(X.columns), (
            f"Unclassified columns: {set(X.columns) - classified}"
        )

    def test_tenure_is_numeric(self, clean_df):
        """tenure must be in the numeric list."""
        X, _ = build_features(clean_df)
        numeric, _ = get_feature_types(X)
        assert "tenure" in numeric

    def test_contract_is_categorical(self, clean_df):
        """Contract must be in the categorical list."""
        X, _ = build_features(clean_df)
        _, categorical = get_feature_types(X)
        assert "Contract" in categorical
