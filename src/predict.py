# src/predict.py
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join('model', 'model.pkl')

def predict_one(sample_dict):
    # Load saved model
    clf = joblib.load(MODEL_PATH)
    # Convert sample to DataFrame
    X = pd.DataFrame([sample_dict])
    # Predict
    pred = clf.predict(X)[0]
    # Predict probability
    proba = clf.predict_proba(X)[0, 1] if hasattr(clf, "predict_proba") else None
    return pred, proba

if __name__ == "__main__":
    sample = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    pred, proba = predict_one(sample)
    print("Prediction:", "Survived" if pred == 1 else "Not survived")
    if proba is not None:
        print("Survival probability:", round(proba, 4))
