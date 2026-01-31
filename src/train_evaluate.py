"""
train_evaluate.py

Purpose
-------
Trains a supervised ML classifier to predict trading book labels (A/B/C/D)
from exactly 4 static features:
- bond_type (categorical)
- currency (categorical)
- time_to_maturity_days (numeric)
- coupon_type (categorical)

It prints ONLY:
- Accuracy
- Precision
- Recall

How it works (high level)
-------------------------
1) Load <project_root>/data/bonds.csv (created by make_dataset.py)
2) Train/test split
3) Preprocess:
   - One-hot encode categorical columns
   - Pass numeric through unchanged
4) Train classifier (Decision Tree)
5) Evaluate on test set using ONLY accuracy / precision / recall
6) Save model artifact to <project_root>/models/model.joblib
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


FEATURES = ["bond_type", "currency", "time_to_maturity_days", "coupon_type"]
TARGET = "label"
VALID_LABELS = ["A", "B", "C", "D"]


def main() -> None:
    # --- Robust project-root paths (works regardless of working directory)
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "bonds.csv"
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"

    # --- Load dataset
    df = pd.read_csv(data_path)

    # Optional safety: ensure only expected labels are present
    df = df[df[TARGET].isin(VALID_LABELS)].copy()

    # --- Split into X/y
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # --- Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
        stratify=y,
    )

    # --- Preprocessing
    categorical_cols = ["bond_type", "currency", "coupon_type"]
    numeric_cols = ["time_to_maturity_days"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # --- Model (Decision Tree Classifier)
    # Decision trees naturally learn rule-like splits (e.g., time_to_maturity_days <= 365)
    model = DecisionTreeClassifier(
        random_state=7
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # --- Train
    clf.fit(X_train, y_train)

    # --- Predict
    y_pred = clf.predict(X_test)

    # --- Identify where the model failed (misclassified rows)
    mis_mask = (y_test.values != y_pred)

    if mis_mask.any():
        failures = X_test.loc[mis_mask].copy()
        failures["true_label"] = y_test.loc[mis_mask].values
        failures["pred_label"] = y_pred[mis_mask]

        # Helpful for your specific rule boundary: sort by closeness to 365
        failures["abs_days_from_365"] = (failures["time_to_maturity_days"] - 365).abs()
        failures = failures.sort_values("abs_days_from_365").drop(columns=["abs_days_from_365"])

        failures_path = project_root / "data" / "misclassifications.csv"
        failures.to_csv(failures_path, index=False)

    # --- Metrics (ONLY these three)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))

    # --- Save model
    joblib.dump(clf, model_path)


if __name__ == "__main__":
    main()
