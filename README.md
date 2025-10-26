# fraud-autoencoder-pyod

Lightweight Python project for fraud/anomaly detection using an AutoEncoder from PyOD.

## Overview
This repository provides code and utilities to train, evaluate, and serve an autoencoder-based anomaly detector for tabular transaction data using PyOD.

## Features
- Train AutoEncoder (PyOD) on transaction data
- Evaluation scripts: ROC AUC, precision, recall, F1
- Simple preprocessing pipeline (scaling, categorical encoding)
- Save/load trained models
- CLI-style example scripts for train / eval / predict

## Repository layout
- data/                 — (optional) raw and processed datasets
- notebooks/            — exploratory analysis and examples
- src/                  — training, evaluation, and inference code
- models/               — saved model artifacts
- requirements.txt
- README.md

## Requirements
- Python 3.8+
- pip

Install dependencies:
```
pip install -r requirements.txt
# or minimal
pip install numpy pandas scikit-learn pyod joblib
```

## Quickstart example (Python)
Basic training + evaluation flow:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import joblib

# load data: X (features), y (binary labels: 1=anomaly/fraud, 0=normal)
df = pd.read_csv("data/transactions.csv")
X = df.drop(columns=["label"])
y = df["label"].values

# split & preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# train
model = AutoEncoder(hidden_neurons=[64, 32, 32, 64], epochs=50, batch_size=64, contamination=0.01)
model.fit(X_train_s)

# predict & evaluate
scores = model.decision_function(X_test_s)  # higher => more abnormal
y_pred = model.predict(X_test_s)            # 1 for outlier

print("ROC AUC:", roc_auc_score(y_test, scores))
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
print("Precision, Recall, F1:", p, r, f)

# save artifacts
joblib.dump(model, "models/autoencoder.joblib")
joblib.dump(scaler, "models/scaler.joblib")
```

## CLI examples
Train:
```
python src/train.py --data data/transactions.csv --out models/ --epochs 50 --batch-size 64
```
Evaluate:
```
python src/evaluate.py --model models/autoencoder.joblib --scaler models/scaler.joblib --data data/transactions_test.csv
```
Predict:
```
python src/predict.py --model models/autoencoder.joblib --scaler models/scaler.joblib --input data/new_transactions.csv --output results/predictions.csv
```

## Configuration
Use a config file or CLI flags for:
- model hyperparameters (hidden sizes, epochs, batch size)
- contamination rate (expected anomaly fraction)
- preprocessing options (scaling, feature selection)

## Notes & best practices
- Ensure class imbalance handling and realistic contamination setting
- Validate with stratified splits or time-based splits for temporal data
- Monitor false positives—tune contamination and thresholds accordingly

## Acknowledgements
Built with PyOD and scikit-learn.

