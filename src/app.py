# src/app.py
"""
Flask API for Titanic Survival Prediction.

This service exposes two endpoints:
- GET  /        → Health check / Info
- POST /predict → Predict survival probability given passenger details
"""

from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("model", "model.pkl")
clf = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    """Health check route."""
    return "Titanic ML API is running. Use POST /predict to get predictions."


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict survival for given passenger(s).
    Expects JSON input with passenger features.
    """
    data = request.get_json()

    # Accept single dict or list of dicts
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Invalid input format"}), 400

    # Make predictions
    preds = clf.predict(df)
    proba = clf.predict_proba(df)[:, 1] if hasattr(clf, "predict_proba") else None

    # Build response
    response = []
    for i, p in enumerate(preds):
        item = {"prediction": int(p)}
        if proba is not None:
            item["survival_probability"] = float(proba[i])
        response.append(item)

    return jsonify(response)


if __name__ == "__main__":
    # Run Flask app (for local development / Docker)
    app.run(host="0.0.0.0", port=5000, debug=True)
