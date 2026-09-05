"""
train_model.py — Train and save classification models for lead conversion prediction.

Author: [Your Name]
College Project — Machine Learning / Data Science
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/leads.csv")
MODEL_DIR  = os.path.dirname(__file__)

# ─── Load Data ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {len(df)} rows")
    return df

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df):
    df = df.drop(columns=["lead_id"])  # not a feature

    # Encode categoricals
    cat_cols = ["lead_source", "industry"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop(columns=["converted"])
    y = df["converted"]

    return X, y, le_dict

# ─── Train ─────────────────────────────────────────────────────────────────────
def train():
    df = load_data()
    X, y, le_dict = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression ──────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_preds = lr.predict(X_test_sc)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]

    results["Logistic Regression"] = {
        "accuracy":  accuracy_score(y_test, lr_preds),
        "roc_auc":   roc_auc_score(y_test, lr_proba),
        "report":    classification_report(y_test, lr_preds),
        "confusion": confusion_matrix(y_test, lr_preds).tolist(),
    }

    # ── Decision Tree ────────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:, 1]

    results["Decision Tree"] = {
        "accuracy":  accuracy_score(y_test, dt_preds),
        "roc_auc":   roc_auc_score(y_test, dt_proba),
        "report":    classification_report(y_test, dt_preds),
        "confusion": confusion_matrix(y_test, dt_preds).tolist(),
    }

    # ── Print Results ────────────────────────────────────────────────────────
    for name, res in results.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Accuracy : {res['accuracy']:.4f}")
        print(f"  ROC-AUC  : {res['roc_auc']:.4f}")
        print(f"\n{res['report']}")

    # ── Save Artifacts ───────────────────────────────────────────────────────
    artifacts = {
        "logistic_regression": lr,
        "decision_tree":       dt,
        "scaler":              scaler,
        "label_encoders":      le_dict,
        "feature_names":       list(X.columns),
        "results":             results,
    }

    save_path = os.path.join(MODEL_DIR, "model_artifacts.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n[INFO] Models saved to {save_path}")
    return artifacts

if __name__ == "__main__":
    train()
