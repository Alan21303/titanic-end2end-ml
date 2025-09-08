# src/app.py
from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load saved model
MODEL_PATH = os.path.join('model', 'model.pkl')
clf = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return "Titanic ML API: POST /predict with JSON sample"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Invalid input"}), 400

    preds = clf.predict(df)
    proba = clf.predict_proba(df)[:, 1] if hasattr(clf, "predict_proba") else None

    response = []
    for i, p in enumerate(preds):
        item = {"prediction": int(p)}
        if proba is not None:
            item["survival_probability"] = float(proba[i])
        response.append(item)

    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
