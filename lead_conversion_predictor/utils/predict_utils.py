"""
predict_utils.py — Load model artifacts and run predictions.

Author: [Your Name]
College Project
"""

import pickle
import numpy as np
import os

ARTIFACTS_PATH = os.path.join(
    os.path.dirname(__file__), "../models/model_artifacts.pkl"
)

_cache = None


def load_artifacts():
    """Load model artifacts from disk (cached after first call)."""
    global _cache
    if _cache is None:
        with open(ARTIFACTS_PATH, "rb") as f:
            _cache = pickle.load(f)
    return _cache


def predict(input_dict: dict, model_choice: str = "Logistic Regression"):
    """
    Run prediction for a single lead.

    Parameters
    ----------
    input_dict  : dict with raw feature values (strings for categoricals)
    model_choice: "Logistic Regression" or "Decision Tree"

    Returns
    -------
    prediction  : int   (0 or 1)
    probability : float (0–1 for class 1)
    """
    arts = load_artifacts()
    le_dict   = arts["label_encoders"]
    scaler    = arts["scaler"]
    features  = arts["feature_names"]

    # Encode categoricals
    row = dict(input_dict)
    for col, le in le_dict.items():
        if row[col] in le.classes_:
            row[col] = le.transform([row[col]])[0]
        else:
            row[col] = 0  # fallback for unseen labels

    X = np.array([[row[f] for f in features]])

    if model_choice == "Logistic Regression":
        model = arts["logistic_regression"]
        X_in  = scaler.transform(X)
    else:
        model = arts["decision_tree"]
        X_in  = X

    prediction  = int(model.predict(X_in)[0])
    probability = float(model.predict_proba(X_in)[0][1])

    return prediction, probability


def get_model_results():
    """Return stored evaluation metrics."""
    arts = load_artifacts()
    return arts.get("results", {})
