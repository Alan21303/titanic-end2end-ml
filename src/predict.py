# src/predict.py
"""
Titanic Survival Prediction - Single Sample Inference

This script loads the trained Titanic survival model and allows
prediction for a single passenger (dict input).
"""

import os
import joblib
import pandas as pd

# Path to trained model
MODEL_PATH = os.path.join("model", "model.pkl")


def predict_one(sample_dict: dict):
    """
    Predict survival for a single passenger.

    Args:
        sample_dict (dict): Passenger features as a dictionary.

    Returns:
        tuple: (prediction, probability)
            - prediction (int): 0 = Not Survived, 1 = Survived
            - probability (float or None): Probability of survival if available
    """
    # Load model
    clf = joblib.load(MODEL_PATH)

    # Convert sample to DataFrame
    X = pd.DataFrame([sample_dict])

    # Prediction
    pred = clf.predict(X)[0]

    # Probability (if supported by model)
    proba = clf.predict_proba(X)[0, 1] if hasattr(clf, "predict_proba") else None

    return pred, proba


if __name__ == "__main__":
    # Example passenger input
    sample = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
    }

    pred, proba = predict_one(sample)

    # Display results
    print("Prediction:", "Survived" if pred == 1 else "Not survived")
    if proba is not None:
        print("Survival probability:", round(proba, 4))
