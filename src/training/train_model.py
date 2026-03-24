from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from src.data.load_data import load_and_clean_data
from src.features.build_features import build_features, get_feature_types


ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

MODEL_OUTPUT_PATH = MODELS_DIR / "best_model.joblib"
METRICS_OUTPUT_PATH = METRICS_DIR / "train_metrics.json"


def build_preprocessors(numeric_features: list[str], categorical_features: list[str]):
    numeric_transformer_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    numeric_transformer_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor_lr = ColumnTransformer([
        ("num", numeric_transformer_lr, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    preprocessor_tree = ColumnTransformer([
        ("num", numeric_transformer_tree, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor_lr, preprocessor_tree


def get_models(preprocessor_lr, preprocessor_tree):
    return {
        "Dummy": Pipeline([
            ("preprocessor", preprocessor_lr),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor_lr),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor_tree),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
    }


def select_base_model(cv_summary: pd.DataFrame) -> tuple[str, dict]:
    """
    Select the base model using ROC-AUC / PR-AUC / F1 ranking,
    with a tie-aware rule that prefers Logistic Regression when
    the ROC-AUC gap is negligible.
    """
    cv_summary = cv_summary.sort_values(
        by=["roc_auc_mean", "pr_auc_mean", "f1_mean"],
        ascending=False,
    ).reset_index(drop=True)

    top1 = cv_summary.iloc[0]
    top2 = cv_summary.iloc[1]

    roc_gap = float(top1["roc_auc_mean"] - top2["roc_auc_mean"])

    if roc_gap < 0.001:
        if "Logistic Regression" in cv_summary["model"].values:
            selected_model_name = "Logistic Regression"
        else:
            selected_model_name = top1["model"]
    else:
        selected_model_name = top1["model"]

    selection_info = {
        "top_model_by_metric": top1["model"],
        "runner_up_model": top2["model"],
        "roc_auc_gap": roc_gap,
        "tie_break_rule_applied": bool(roc_gap < 0.001),
        "final_selected_model": selected_model_name,
    }

    return selected_model_name, selection_info


def get_probability_candidates(best_model):
    return {
        "Uncalibrated": best_model,
        "Sigmoid calibrated": CalibratedClassifierCV(
            estimator=best_model,
            method="sigmoid",
            cv=5,
        ),
        "Isotonic calibrated": CalibratedClassifierCV(
            estimator=best_model,
            method="isotonic",
            cv=5,
        ),
    }


def select_probability_version(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_model,
) -> tuple[str, pd.DataFrame]:
    """
    Compare uncalibrated vs calibrated probability versions using
    out-of-fold predictions on the training set only.
    """
    calibration_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = get_probability_candidates(best_model)

    rows = []
    for name, model in candidates.items():
        y_proba_oof = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=calibration_cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        rows.append({
            "model_version": name,
            "oof_brier_score": float(brier_score_loss(y_train, y_proba_oof)),
            "oof_roc_auc": float(roc_auc_score(y_train, y_proba_oof)),
            "oof_pr_auc": float(average_precision_score(y_train, y_proba_oof)),
        })

    comparison = pd.DataFrame(rows).sort_values(
        by=["oof_brier_score", "oof_roc_auc"],
        ascending=[True, False],
    ).reset_index(drop=True)

    selected_probability_version = comparison.loc[0, "model_version"]
    return selected_probability_version, comparison


def build_final_probability_model(selected_probability_version: str, best_model):
    if selected_probability_version == "Uncalibrated":
        return best_model
    if selected_probability_version == "Sigmoid calibrated":
        return CalibratedClassifierCV(
            estimator=best_model,
            method="sigmoid",
            cv=5,
        )
    return CalibratedClassifierCV(
        estimator=best_model,
        method="isotonic",
        cv=5,
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_and_clean_data()
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor_lr, preprocessor_tree = build_preprocessors(
        numeric_features,
        categorical_features,
    )
    models = get_models(preprocessor_lr, preprocessor_tree)

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    # Base-model comparison
    rows = []
    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {"model": model_name}
        for metric_name in scoring:
            values = scores[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=1))
        rows.append(row)

    cv_summary = pd.DataFrame(rows)
    selected_model_name, selection_info = select_base_model(cv_summary)
    best_model = models[selected_model_name]

    # Probability-version comparison on training only
    selected_probability_version, calibration_comparison = select_probability_version(
        X_train,
        y_train,
        best_model,
    )

    # Fit final selected probability model on full training data
    final_model = build_final_probability_model(
        selected_probability_version,
        best_model,
    )
    final_model.fit(X_train, y_train)

    # Save artifact
    joblib.dump(final_model, MODEL_OUTPUT_PATH)

    results = {
        "selected_model": selected_model_name,
        "selected_probability_version": selected_probability_version,
        "selection_info": selection_info,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "cv_summary": cv_summary.sort_values(
            by=["roc_auc_mean", "pr_auc_mean", "f1_mean"],
            ascending=False,
        ).to_dict(orient="records"),
        "calibration_comparison": calibration_comparison.to_dict(orient="records"),
    }

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Training complete.")
    print(f"Selected base model: {selected_model_name}")
    print(f"Selected probability version: {selected_probability_version}")
    print(f"Saved model to: {MODEL_OUTPUT_PATH}")
    print(f"Saved metrics to: {METRICS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()