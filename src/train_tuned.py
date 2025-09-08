import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load dataset
    data_path = os.path.join('data', 'train.csv')
    df = pd.read_csv(data_path)

    # Features and target
    target = 'Survived'
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df[target]

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Sex', 'Embarked']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Base model
    rf = RandomForestClassifier(random_state=42)

    # Pipeline
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    # Hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # GridSearchCV with 5-fold CV
    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_val)
    print("Best Parameters:", grid_search.best_params_)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))

    # Save best model
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_model, os.path.join('model', 'model.pkl'))
    print("Best model saved at model/model.pkl")

if __name__ == "__main__":
    main()
