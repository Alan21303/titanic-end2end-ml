# src/deep_search.py
"""
Titanic Survival Prediction - Hyperparameter Tuning

This script trains a RandomForestClassifier with preprocessing pipelines and
performs an extensive GridSearchCV to find the best hyperparameters.

Steps:
1. Load Titanic dataset
2. Split into train/validation sets
3. Define preprocessing for numeric & categorical features
4. Run grid search with cross-validation
5. Evaluate the best model
6. Save the trained model to /model/model.pkl
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    """Run model training with hyperparameter search."""

    # --- Load dataset ---
    data_path = os.path.join("data", "train.csv")
    df = pd.read_csv(data_path)

    target = "Survived"
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    X = df[features]
    y = df[target]

    # --- Train/test split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Preprocessing ---
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # --- Model pipeline ---
    rf = RandomForestClassifier(random_state=42)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf),
        ]
    )

    # --- Hyperparameter grid ---
    param_grid = {
        "classifier__n_estimators": [100, 200, 300, 500, 800],
        "classifier__max_depth": [None, 5, 10, 20, 30, 50],
        "classifier__min_samples_split": [2, 5, 10, 15],
        "classifier__min_samples_leaf": [1, 2, 4, 6],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__bootstrap": [True, False],
    }

    # --- Grid search with 5-fold CV ---
    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    # --- Best model ---
    best_model = grid_search.best_estimator_

    # --- Evaluation ---
    y_pred = best_model.predict(X_val)
    print("Best Parameters:", grid_search.best_params_)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))

    # --- Save best model ---
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, os.path.join("model", "model.pkl"))
    print("Best model saved at model/model.pkl")


if __name__ == "__main__":
    main()
