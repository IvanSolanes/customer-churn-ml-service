"""
Training pipeline with full MLflow integration.

Every run logs:
  Parameters  : CV strategy, train/test sizes, random state, dataset info
  Metrics     : CV scores for every candidate model, calibration comparison,
                holdout ROC-AUC / PR-AUC / Brier score
  Tags        : selected model name, selected probability version
  Artifact    : final fitted model (logged + registered in the Registry)

After training, the model is automatically registered as a new version of
'churn-predictor' in the MLflow Model Registry and promoted to Production.

Run from the project root:
    python -m src.training.train_model

Open the MLflow UI afterwards:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import json
from pathlib import Path

from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)
from src.data.load_data import load_and_clean_data
from src.features.build_features import build_features, get_feature_types

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_N_SPLITS  = 5
CV_N_REPEATS = 3

METRICS_DIR         = Path("artifacts/metrics")
METRICS_OUTPUT_PATH = METRICS_DIR / "train_metrics.json"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessors(numeric_features, categorical_features):
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor_lr = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    preprocessor_tree = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return preprocessor_lr, preprocessor_tree


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(preprocessor_lr, preprocessor_tree) -> dict:
    return {
        "Dummy": Pipeline([
            ("preprocessor", preprocessor_lr),
            ("classifier",   DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor_lr),
            ("classifier",   LogisticRegression(
                max_iter=1000, class_weight="balanced",
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor_tree),
            ("classifier",   RandomForestClassifier(
                n_estimators=300,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_base_model(cv_summary: pd.DataFrame) -> tuple[str, dict]:
    cv_summary = cv_summary.sort_values(
        by=["roc_auc_mean", "pr_auc_mean", "f1_mean"], ascending=False,
    ).reset_index(drop=True)

    top1    = cv_summary.iloc[0]
    top2    = cv_summary.iloc[1]
    roc_gap = float(top1["roc_auc_mean"] - top2["roc_auc_mean"])

    if roc_gap < 0.001 and "Logistic Regression" in cv_summary["model"].values:
        selected = "Logistic Regression"
    else:
        selected = top1["model"]

    return selected, {
        "top_model_by_metric":    top1["model"],
        "runner_up_model":        top2["model"],
        "roc_auc_gap":            roc_gap,
        "tie_break_rule_applied": bool(roc_gap < 0.001),
        "final_selected_model":   selected,
    }


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

def select_probability_version(X_train, y_train, best_model) -> tuple[str, pd.DataFrame]:
    cal_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "Uncalibrated":      best_model,
        "Sigmoid calibrated": CalibratedClassifierCV(
            estimator=best_model, method="sigmoid", cv=5,
        ),
        "Isotonic calibrated": CalibratedClassifierCV(
            estimator=best_model, method="isotonic", cv=5,
        ),
    }
    rows = []
    for name, model in candidates.items():
        y_proba = cross_val_predict(
            model, X_train, y_train,
            cv=cal_cv, method="predict_proba", n_jobs=-1,
        )[:, 1]
        rows.append({
            "model_version":   name,
            "oof_brier_score": float(brier_score_loss(y_train, y_proba)),
            "oof_roc_auc":     float(roc_auc_score(y_train, y_proba)),
            "oof_pr_auc":      float(average_precision_score(y_train, y_proba)),
        })

    comparison = pd.DataFrame(rows).sort_values(
        by=["oof_brier_score", "oof_roc_auc"], ascending=[True, False],
    ).reset_index(drop=True)

    return comparison.loc[0, "model_version"], comparison


def build_final_model(probability_version: str, best_model):
    if probability_version == "Uncalibrated":
        return best_model
    method = "sigmoid" if probability_version == "Sigmoid calibrated" else "isotonic"
    return CalibratedClassifierCV(estimator=best_model, method=method, cv=5)


# ---------------------------------------------------------------------------
# MLflow logging helpers
# ---------------------------------------------------------------------------

def _log_cv_metrics(cv_summary: pd.DataFrame) -> None:
    """Log CV metrics for every candidate model with a namespaced prefix."""
    metric_cols = [
        "roc_auc_mean", "roc_auc_std",
        "pr_auc_mean",  "pr_auc_std",
        "f1_mean",      "f1_std",
        "precision_mean", "recall_mean", "accuracy_mean",
    ]
    for _, row in cv_summary.iterrows():
        prefix = row["model"].lower().replace(" ", "_")
        for col in metric_cols:
            if col in row:
                mlflow.log_metric(f"cv.{prefix}.{col}", float(row[col]))


def _log_calibration_metrics(comparison: pd.DataFrame) -> None:
    """Log OOF calibration metrics for every probability version."""
    for _, row in comparison.iterrows():
        prefix = row["model_version"].lower().replace(" ", "_")
        mlflow.log_metric(f"calibration.{prefix}.brier", float(row["oof_brier_score"]))
        mlflow.log_metric(f"calibration.{prefix}.roc_auc", float(row["oof_roc_auc"]))
        mlflow.log_metric(f"calibration.{prefix}.pr_auc",  float(row["oof_pr_auc"]))


def _register_and_promote(run_id: str) -> str:
    """
    Register the logged model as a new version in the Model Registry
    and assign the 'champion' alias to it.

    The alias replaces the deprecated stage system (Production/Staging).
    All inference code loads the model via models:/churn-predictor@champion.
    """
    client    = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model_ver = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)

    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias="champion",
        version=model_ver.version,
    )
    return model_ver.version


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Data
    df = load_and_clean_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor_lr, preprocessor_tree = build_preprocessors(
        numeric_features, categorical_features,
    )
    models = get_models(preprocessor_lr, preprocessor_tree)

    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=RANDOM_STATE,
    )
    scoring = {
        "accuracy":  "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall":    make_scorer(recall_score,    zero_division=0),
        "f1":        make_scorer(f1_score,        zero_division=0),
        "roc_auc":   "roc_auc",
        "pr_auc":    "average_precision",
    }

    run_name = f"churn-predictor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow run started: {run_id}")

        # Parameters
        mlflow.log_params({
            "cv_n_splits":      CV_N_SPLITS,
            "cv_n_repeats":     CV_N_REPEATS,
            "test_size":        TEST_SIZE,
            "random_state":     RANDOM_STATE,
            "n_features":       X_train.shape[1],
            "n_train_rows":     X_train.shape[0],
            "n_test_rows":      X_test.shape[0],
            "train_churn_rate": round(float(y_train.mean()), 4),
        })

        # Cross-validation
        rows = []
        for model_name, model in models.items():
            scores = cross_validate(
                model, X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False,
            )
            row = {"model": model_name}
            for metric_name in scoring:
                vals = scores[f"test_{metric_name}"]
                row[f"{metric_name}_mean"] = float(vals.mean())
                row[f"{metric_name}_std"]  = float(vals.std(ddof=1))
            rows.append(row)

        cv_summary = pd.DataFrame(rows)
        _log_cv_metrics(cv_summary)

        # Model selection
        selected_model_name, selection_info = select_base_model(cv_summary)
        best_model = models[selected_model_name]

        mlflow.set_tags({
            "selected_base_model":    selected_model_name,
            "tie_break_applied":      str(selection_info["tie_break_rule_applied"]),
            "top_model_by_metric":    selection_info["top_model_by_metric"],
        })
        mlflow.log_metric(
            "model_selection.roc_auc_gap",
            float(selection_info["roc_auc_gap"]),
        )

        # Calibration selection
        selected_prob_version, calibration_comparison = select_probability_version(
            X_train, y_train, best_model,
        )
        _log_calibration_metrics(calibration_comparison)
        mlflow.set_tag("selected_probability_version", selected_prob_version)

        # Fit final model
        final_model = build_final_model(selected_prob_version, best_model)
        final_model.fit(X_train, y_train)

        # Holdout metrics
        final_proba  = final_model.predict_proba(X_test)[:, 1]
        holdout_roc  = float(roc_auc_score(y_test, final_proba))
        holdout_pr   = float(average_precision_score(y_test, final_proba))
        holdout_brier = float(brier_score_loss(y_test, final_proba))

        mlflow.log_metrics({
            "holdout.roc_auc":     holdout_roc,
            "holdout.pr_auc":      holdout_pr,
            "holdout.brier_score": holdout_brier,
        })

        # Log model + register
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
            input_example=X_train.iloc[:3],
        )
        model_version = _register_and_promote(run_id)

        # Save local JSON audit trail
        results = {
            "mlflow_run_id":                run_id,
            "mlflow_model_name":            MLFLOW_MODEL_NAME,
            "mlflow_model_version":         model_version,
            "selected_model":               selected_model_name,
            "selected_probability_version": selected_prob_version,
            "selection_info":               selection_info,
            "train_shape":                  list(X_train.shape),
            "test_shape":                   list(X_test.shape),
            "holdout_metrics": {
                "roc_auc":     holdout_roc,
                "pr_auc":      holdout_pr,
                "brier_score": holdout_brier,
            },
            "cv_summary": cv_summary.sort_values(
                by=["roc_auc_mean", "pr_auc_mean", "f1_mean"], ascending=False,
            ).to_dict(orient="records"),
            "calibration_comparison": calibration_comparison.to_dict(orient="records"),
        }
        with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(METRICS_OUTPUT_PATH))

    print(f"\nTraining complete.")
    print(f"  Selected model            : {selected_model_name}")
    print(f"  Probability version       : {selected_prob_version}")
    print(f"  Holdout ROC-AUC           : {holdout_roc:.4f}")
    print(f"  MLflow run ID             : {run_id}")
    print(f"  Registered as             : {MLFLOW_MODEL_NAME} v{model_version} @champion")
    print(f"\nView results in the MLflow UI:")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
