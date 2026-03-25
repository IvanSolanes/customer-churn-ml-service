# Customer Churn ML Service

![CI](https://github.com/IvanSolanes/customer-churn-ml-service/actions/workflows/ci.yml/badge.svg)git add README.md

End-to-end machine learning project to predict customer churn for a subscription-style business.

## Project goal

The objective of this project is to identify customers who are likely to churn so that a business can take early retention actions.

A subscription business wants to identify customers at high risk of churn so
retention teams can act earlier. This project trains and evaluates machine
learning models to predict churn, tracks every experiment with MLflow,
registers the best model in the MLflow Model Registry, and serves predictions
through an API for operational use.

## Project scope

This repository will include:

- data loading and preprocessing
- feature engineering
- model training and evaluation
- reusable inference logic
- an API for predictions
- batch scoring workflow

## Dataset

This project currently uses the raw churn dataset stored in `data/raw/Telco-Customer-Churn.csv`.

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/IvanSolanes/customer-churn-ml-service.git
cd customer-churn-ml-service
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Train the model

```bash
python -m src.training.train_model
```

This will:
- compare a dummy baseline, logistic regression, and random forest using repeated stratified cross-validation,
- apply a tie-aware model selection rule,
- select the best probability calibration version using out-of-fold evaluation on training data only,
- log all parameters, metrics, and artifacts to MLflow,
- register the final model in the MLflow Model Registry as `churn-predictor`,
- assign the `@champion` alias to the new version.

### 3. Open the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Go to `http://127.0.0.1:5000` to explore experiment runs, compare metrics
across runs, and inspect the Model Registry.

### 4. Start the API

```bash
uvicorn src.api.main:app --reload
```

The API will be live at `http://127.0.0.1:8000`.
Interactive docs: `http://127.0.0.1:8000/docs`

---

## MLflow integration

Every training run is tracked in a local SQLite database (`mlflow.db`).

### What is logged per run

| Type | What |
|---|---|
| Parameters | CV strategy, test size, random state, dataset shape, churn rate |
| Metrics | CV scores per model (ROC-AUC, PR-AUC, F1, precision, recall), calibration OOF scores, holdout ROC-AUC / PR-AUC / Brier score |
| Tags | Selected base model, selected probability version, tie-break applied |
| Artifacts | Final fitted model, `train_metrics.json` audit trail |

### Model Registry

The best model from each run is registered as `churn-predictor` and assigned
the `@champion` alias. All inference code loads the model via:

```
models:/churn-predictor@champion
```

To promote a new model version without touching any code, reassign the alias:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("churn-predictor", "champion", version="3")
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health — reports model name, alias, version |
| `POST` | `/predict` | Single-customer churn prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 5,000 customers) |

### Example — single prediction

**Request**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 351.75,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes"
  }'
```

**Response**

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8354,
  "risk_tier": "High"
}
```

The `risk_tier` field maps probabilities to business-friendly buckets:

| Tier | Probability range |
|---|---|
| `Low` | below 0.35 |
| `Medium` | 0.35 to 0.60 |
| `High` | above 0.60 |

### Example — health check

```json
{
  "status": "ok",
  "model_name": "churn-predictor",
  "model_alias": "champion",
  "model_version": "3",
  "model_uri": "models:/churn-predictor@champion",
  "tracking_uri": "sqlite:///mlflow.db"
}
```

---

## Batch scoring (offline)

Score a full CSV file without the API:

```bash
python scripts/batch_score.py \
  --input  data/raw/Telco-Customer-Churn.csv \
  --output data/scored/scored_customers.csv
```

Output CSV contains `customerID`, `churn_prediction`, `churn_probability`,
and `risk_tier` for every row.

---

## Model selection logic

The training pipeline applies a **tie-aware selection rule**: if the ROC-AUC
gap between the top two models is smaller than 0.001, logistic regression is
preferred. This reflects a deliberate trade-off — logistic regression is
faster, more interpretable, and easier to audit, making it preferable over a
marginally better but more opaque model in a business context.

Probability calibration is selected separately using out-of-fold Brier score
on the training set only, keeping the holdout set fully unseen until final
evaluation.

---

## Tests

```bash
pytest tests/ -v
```

18 tests across two modules covering data cleaning, feature engineering,
feature type classification, and all API endpoints.

---

## Requirements

```
Python 3.12
```

```bash
pip install -r requirements.txt
pip install -e .
```
