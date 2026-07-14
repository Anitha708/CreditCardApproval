import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Credit Card Fraud Detection")

st.write("Enter transaction details")

columns = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

inputs = []

# Input fields
for col in columns:
    value = st.number_input(col, value=0.0)
    inputs.append(value)


# Predict button (outside the loop)
if st.button("Predict", key="predict_btn"):

    data = pd.DataFrame([inputs], columns=columns)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Fraudulent Transaction ❌")
    else:
        st.success("Legitimate Transaction ✅")