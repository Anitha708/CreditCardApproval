import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details")

# Feature names
columns = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17",
    "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28", "Amount"
]

# Input fields
input_data = []

for col in columns:
    value = st.number_input(col, value=0.0, format="%.2f")
    input_data.append(value)

# Predict button
if st.button("Predict"):

    input_df = pd.DataFrame([input_data], columns=columns)

    # Prediction
    prediction = model.predict(input_df)[0]

    # Prediction Probability
    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    st.write(f"**Legitimate Probability:** {probability[0]:.2%}")
    st.write(f"**Fraud Probability:** {probability[1]:.2%}")

    if prediction == 0:
        st.success("✅ Legitimate Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")

    st.subheader("Transaction Details")
    st.dataframe(input_df)