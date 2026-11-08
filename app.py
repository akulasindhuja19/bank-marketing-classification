import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="🏦",
    layout="wide"
)

MODEL_DIR = Path("model")

MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

st.title("🏦 Bank Marketing Prediction App")
st.markdown(
    """
    Predict whether a customer will subscribe to a term deposit using
    different Machine Learning models.
    """
)

# Select model
selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODELS.keys())
)

# Load model
model_path = MODEL_DIR / MODELS[selected_model]
model = joblib.load(model_path)

# Load test data
test_df = pd.read_csv("test_data.csv")

st.subheader("Dataset")

st.write(
    f"Loaded **{len(test_df)}** records from test_data.csv"
)

# Select a row
record_index = st.selectbox(
    "Select Customer Record",
    range(len(test_df))
)

selected_row = test_df.iloc[[record_index]]

actual = selected_row["y"].values[0]

X = selected_row.drop(columns=["y"])

st.subheader("Customer Information")
st.dataframe(X)

if st.button("Predict"):

    prediction = model.predict(X)[0]

    result = "yes" if prediction == 1 else "no"

    st.subheader("Prediction")

    if result == "yes":
        st.success("Customer is likely to subscribe to a term deposit.")
    else:
        st.error("Customer is unlikely to subscribe to a term deposit.")

    st.write("### Predicted Class")
    st.write(result)

    st.write("### Actual Class")
    st.write(actual)

    try:
        probability = model.predict_proba(X)[0][1]

        st.write("### Subscription Probability")

        st.progress(float(probability))

        st.metric(
            label="Probability of Subscription",
            value=f"{probability*100:.2f}%"
        )

    except Exception:
        st.info(
            "This model does not provide probability estimates."
        )