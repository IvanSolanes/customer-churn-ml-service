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

## Current status

Initial project structure has been created and the raw dataset has been added.

## Dataset

This project currently uses the raw churn dataset stored in `data/raw/Telco-Customer-Churn.csv`.

The raw file is kept unchanged. Data cleaning, preprocessing, and feature preparation will be implemented in later steps.

## Training pipeline

Run the training workflow from the project root:

```bash
python -m src.training.train_model
```
This script:

compares candidate base models with repeated cross-validation,
applies a tie-aware model selection rule,
selects the best probability version using training data only,
saves the final selected model to artifacts/models/best_model.joblib,
and writes training metrics to artifacts/metrics/train_metrics.json.



## Requirements
- Python 3.12
- Install dependencies: `pip install -r requirements.txt`
```

---

## 4. Pin pandas to a Compatible Version in `requirements.txt`

Fix the root cause you just experienced for future users:
```
pandas>=2.2.2,<3.0
scikit-learn>=1.3
joblib>=1.3
```