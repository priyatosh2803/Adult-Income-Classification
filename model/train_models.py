from pathlib import Path

import joblib
import numpy as np
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "adult.csv"
TEST_PATH = ROOT / "test_data.csv"

TARGET = "income"
LABELS = {"<=50K": 0, ">50K": 1}
CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native-country",
]
RANDOM_STATE = 42

MODELS = {
    "logistic_regression": ("Logistic Regression", LogisticRegression(max_iter=1000)),
    "decision_tree": ("Decision Tree", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)),
    "knn": ("KNN", KNeighborsClassifier(n_neighbors=5)),
    "naive_bayes": ("Naive Bayes", GaussianNB()),
    "svm": ("Support Vector Machine", SVC(kernel="rbf", random_state=RANDOM_STATE)),
}


def load_clean(path):
    df = pd.read_csv(path)
    text_cols = CATEGORICAL + [TARGET]
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    df[TARGET] = df[TARGET].str.rstrip(".")
    return df


def build_pipeline(estimator):
    encoder = ColumnTransformer(
        [("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("encode", encoder), ("scale", StandardScaler()), ("model", estimator)])


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, positive_scores(model, X)),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = load_clean(DATA_PATH)
    features = [c for c in df.columns if c != TARGET]

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    test_df.to_csv(TEST_PATH, index=False)

    X_train, y_train = train_df[features], train_df[TARGET].map(LABELS)
    X_test, y_test = test_df[features], test_df[TARGET].map(LABELS)

    print(f"Cleaned rows: {len(df)} | train: {len(train_df)} | test: {len(test_df)}")

    rows = []
    for file_key, (display_name, estimator) in MODELS.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODEL_DIR / f"{file_key}.joblib")
        rows.append({"ML Model Name": display_name, **compute_metrics(pipeline, X_test, y_test)})
        print(f"Trained and saved {display_name}")

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
