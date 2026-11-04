from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_TEST_PATH = BASE_DIR / "test_data.csv"

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
FEATURES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "educational-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


def clean(df):
    text_cols = CATEGORICAL + [TARGET]
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    df[TARGET] = df[TARGET].str.rstrip(".")
    return df


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


@st.cache_data
def evaluate_all(df):
    X = df[FEATURES]
    y = df[TARGET].map(LABELS)
    rows = []
    for name, filename in MODEL_FILES.items():
        model = load_model(filename)
        y_pred = model.predict(X)
        rows.append(
            {
                "ML Model Name": name,
                "Accuracy": accuracy_score(y, y_pred),
                "AUC": roc_auc_score(y, positive_scores(model, X)),
                "Precision": precision_score(y, y_pred),
                "Recall": recall_score(y, y_pred),
                "F1": f1_score(y, y_pred),
                "MCC": matthews_corrcoef(y, y_pred),
            }
        )
    return pd.DataFrame(rows).round(4)


st.set_page_config(page_title="Adult Income Classifier", layout="wide")
st.title("Adult Income Classification")
st.write(
    "Predicts whether annual income exceeds $50K using five pre-trained classifiers. "
    "Upload a held-out test CSV that contains the 14 feature columns and the `income` column. "
    "The saved models are loaded and evaluated; nothing is retrained."
)

uploaded = st.file_uploader("Upload test CSV", type="csv")
if uploaded is not None:
    raw = pd.read_csv(uploaded)
else:
    raw = pd.read_csv(DEFAULT_TEST_PATH)
    st.info("No file uploaded. Showing results on the bundled test_data.csv.")

missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

df = clean(raw)
if df.empty:
    st.error("No usable rows remain after removing missing values ('?').")
    st.stop()

y_true = df[TARGET].map(LABELS)
if y_true.isna().any():
    st.error("The income column must contain only '<=50K' or '>50K'.")
    st.stop()
if y_true.nunique() < 2:
    st.error("The test data must contain both income classes to compute AUC.")
    st.stop()

st.subheader("Model comparison on the test data")
with st.spinner("Evaluating all models..."):
    comparison = evaluate_all(df)
st.dataframe(comparison, hide_index=True)

st.subheader("Evaluate a single model")
selected = st.selectbox("Select a model", list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[selected])
y_pred = model.predict(df[FEATURES])
metrics = comparison.set_index("ML Model Name").loc[selected]

cols = st.columns(6)
for col, name in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    col.metric(name, f"{metrics[name]:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual <=50K", "Actual >50K"],
            columns=["Predicted <=50K", "Predicted >50K"],
        )
    )
with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_true, y_pred, target_names=["<=50K", ">50K"], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).T.round(4))
