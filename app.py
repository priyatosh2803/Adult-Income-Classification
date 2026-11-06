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
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(191, 219, 254, 0.48), transparent 34%),
            #f5f7fb;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero-card {
        padding: 1.65rem 1.9rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        color: #f8fafc;
        background: linear-gradient(118deg, #10243e 0%, #174a73 58%, #167d82 100%);
        box-shadow: 0 14px 32px rgba(15, 41, 69, 0.16);
    }
    .hero-eyebrow {
        margin-bottom: 0.35rem;
        color: #a7f3d0;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .hero-title {
        margin: 0;
        color: #ffffff;
        font-size: clamp(1.8rem, 3vw, 2.45rem);
        font-weight: 750;
        line-height: 1.15;
    }
    .hero-copy {
        max-width: 850px;
        margin: 0.65rem 0 0;
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .section-label {
        margin: 0 0 0.2rem;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .section-label--upload { color: #0369a1; }
    .section-label--comparison { color: #047857; }
    .section-label--model { color: #7c3aed; }
    .section-label--matrix { color: #b45309; }
    .section-label--report { color: #be185d; }
    .section-title {
        margin: 0 0 0.75rem;
        color: #172554;
        font-size: 1.25rem;
        font-weight: 700;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #dbe4ef;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 5px 18px rgba(15, 41, 69, 0.045);
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #7c9bb8;
        border-radius: 12px;
        background: #f2f8fc;
    }
    [data-testid="stMetric"] {
        min-height: 108px;
        padding: 0.9rem 1rem;
        border: 1px solid #dbeafe;
        border-top: 4px solid #2563eb;
        border-radius: 13px;
        background: linear-gradient(155deg, #ffffff 15%, #eff6ff 100%);
        box-shadow: 0 5px 14px rgba(37, 99, 235, 0.07);
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] {
        border-color: #ccfbf1;
        border-top-color: #0d9488;
        background: linear-gradient(155deg, #ffffff 15%, #f0fdfa 100%);
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {
        border-color: #ede9fe;
        border-top-color: #7c3aed;
        background: linear-gradient(155deg, #ffffff 15%, #f5f3ff 100%);
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] {
        border-color: #fce7f3;
        border-top-color: #db2777;
        background: linear-gradient(155deg, #ffffff 15%, #fdf2f8 100%);
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(5) [data-testid="stMetric"] {
        border-color: #fef3c7;
        border-top-color: #d97706;
        background: linear-gradient(155deg, #ffffff 15%, #fffbeb 100%);
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(6) [data-testid="stMetric"] {
        border-color: #e0e7ff;
        border-top-color: #4f46e5;
        background: linear-gradient(155deg, #ffffff 15%, #eef2ff 100%);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 11px;
        overflow: hidden;
    }
    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }
    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }
        .hero-card {
            padding: 1.25rem;
        }
    }
    </style>
    <div class="hero-card">
        <div class="hero-eyebrow">Machine Learning · Model Evaluation</div>
        <h1 class="hero-title">Adult Income Classification</h1>
        <p class="hero-copy">
            Compare five trained classifiers on held-out data. Upload a test CSV
            with the 14 input features and <code>income</code> target to explore
            their performance. Models are loaded as saved; nothing is retrained.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        '<div class="section-label section-label--upload">01 / Test data</div>'
        '<div class="section-title">Choose a dataset to evaluate</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_csv(DEFAULT_TEST_PATH)
        st.caption("Using the bundled held-out test_data.csv. Upload another test set to replace it.")

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

with st.container(border=True):
    st.markdown(
        '<div class="section-label section-label--comparison">02 / Benchmark</div>'
        '<div class="section-title">Compare model performance</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Evaluating all models..."):
        comparison = evaluate_all(df)
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "ML Model Name": st.column_config.TextColumn("Classifier", width="medium"),
            "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
            "AUC": st.column_config.NumberColumn("AUC", format="%.4f"),
            "Precision": st.column_config.NumberColumn("Precision", format="%.4f"),
            "Recall": st.column_config.NumberColumn("Recall", format="%.4f"),
            "F1": st.column_config.NumberColumn("F1 score", format="%.4f"),
            "MCC": st.column_config.NumberColumn("MCC", format="%.4f"),
        },
    )
    best_accuracy = comparison.loc[comparison["Accuracy"].idxmax()]
    st.caption(
        f"Top accuracy on this test set: {best_accuracy['ML Model Name']} "
        f"({best_accuracy['Accuracy']:.4f})."
    )

with st.container(border=True):
    st.markdown(
        '<div class="section-label section-label--model">03 / Model details</div>'
        '<div class="section-title">Inspect one classifier</div>',
        unsafe_allow_html=True,
    )
    selected = st.selectbox("Select a model", list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[selected])
y_pred = model.predict(df[FEATURES])
metrics = comparison.set_index("ML Model Name").loc[selected]
st.caption(f"Detailed results for **{selected}** on {len(df):,} usable test rows.")

cols = st.columns(6)
for col, name in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    col.metric(name, f"{metrics[name]:.4f}")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown(
            '<div class="section-label section-label--matrix">04 / Prediction counts</div>'
            '<div class="section-title">Confusion matrix</div>',
            unsafe_allow_html=True,
        )
        st.caption("Rows are actual classes; columns are predicted classes.")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual <=50K", "Actual >50K"],
                columns=["Predicted <=50K", "Predicted >50K"],
            ),
            height=130,
        )
with right:
    with st.container(border=True):
        st.markdown(
            '<div class="section-label section-label--report">05 / Per-class results</div>'
            '<div class="section-title">Classification report</div>',
            unsafe_allow_html=True,
        )
        report = classification_report(
            y_true, y_pred, target_names=["<=50K", ">50K"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).T.round(4), height=220)
