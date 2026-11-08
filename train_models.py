"""Train five classifiers on the UCI Bank Marketing dataset and save them.

Usage:
    python train_models.py bank-additional-full.csv
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "bank-additional-full.csv")
MODEL_DIR = Path("model")
TARGET = "y"
RANDOM_STATE = 42

# Load. The file is semicolon-separated.
df = pd.read_csv(DATA_PATH, sep=";")

# 'duration' is only known after the call ends, so it leaks the outcome.
# The UCI page recommends dropping it for a realistic model.
df = df.drop(columns=["duration"])

y = (df[TARGET] == "yes").astype(int)
X = df.drop(columns=[TARGET])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Stratified split keeps the yes/no ratio the same in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


def make_preprocessor():
    # A fresh preprocessor per model, so no fitted state is shared.
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    # No probability=True: it is very slow on ~33k rows. AUC uses decision_function instead.
    "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
}


def get_scores(pipe, X_data):
    """Continuous scores for AUC: probabilities if available, else decision values."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X_data)[:, 1]
    return pipe.decision_function(X_data)


MODEL_DIR.mkdir(exist_ok=True)
results = []

for name, estimator in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("clf", estimator),
    ])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    score = get_scores(pipe, X_test)

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, score),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, MODEL_DIR / filename)

# Metrics table for the README.
metrics_df = pd.DataFrame(results).round(4)
metrics_df.to_csv("metrics.csv", index=False)
print(metrics_df.to_string(index=False))

# Test split for the Streamlit app. It keeps the raw columns and the yes/no target.
test_df = X_test.copy()
test_df[TARGET] = y_test.map({1: "yes", 0: "no"}).values
test_df.to_csv("test_data.csv", index=False)
print(f"Saved {len(test_df)} test rows to test_data.csv")
