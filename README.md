# Customer Churn ML Service

End-to-end machine learning project to predict customer churn for a subscription-style business.

## Project goal

The objective of this project is to identify customers who are likely to churn so that a business can take early retention actions.

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
```

### 2. Train the model

```bash
python -m src.training.train_model
```

This will:
- compare a dummy baseline, logistic regression, and random forest using repeated stratified cross-validation,
- apply a tie-aware model selection rule (prefers logistic regression when the ROC-AUC gap is negligible),
- select the best probability calibration version (uncalibrated, sigmoid, or isotonic) using out-of-fold evaluation on training data only,
- save the final model to `artifacts/models/best_model.joblib`,
- write training metrics to `artifacts/metrics/train_metrics.json`.

### 3. Start the API

```bash
uvicorn src.api.main:app --reload
```

The API will be live at `http://127.0.0.1:8000`.  
Interactive docs: `http://127.0.0.1:8000/docs`

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
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
- `Low` — below 0.35
- `Medium` — 0.35 to 0.60
- `High` — above 0.60

---

## Batch scoring (offline)

Score a full CSV file without the API:

```bash
python scripts/batch_score.py \
  --input  data/raw/Telco-Customer-Churn.csv \
  --output data/scored/scored_customers.csv
```

Output CSV contains `customerID`, `churn_prediction`, `churn_probability`, and `risk_tier` for every row. Useful for periodic bulk runs or retention campaign targeting.

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

## Requirements

```
Python 3.12
```

Install all dependencies:

```bash
pip install -r requirements.txt
```