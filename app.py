# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

st.title("💳 Credit Card Fraud Detection System")

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Helper: Predict fraud
def predict_fraud(input_dict):
    df = pd.DataFrame([input_dict])
    df["Amount"] = scaler.transform(df[["Amount"]])
    if "Time" in df.columns:
        df = df.drop("Time", axis=1)
    prob = model.predict_proba(df)[0][1]
    return int(prob > 0.65), prob

# Optional: Show feature importance
def show_feature_importance():
    importances = model.feature_importances_
    features = model.feature_names_in_
    sorted_idx = np.argsort(importances)[-10:]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importances[sorted_idx], y=features[sorted_idx], ax=ax)
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)

# Input controls
st.sidebar.header("🔢 Input Transaction Features")
inputs = {}
for i in range(1, 29):
    inputs[f"V{i}"] = st.sidebar.slider(f"V{i}", -30.0, 30.0, 0.0)

inputs["Amount"] = st.sidebar.slider("Amount", 0.0, 5000.0, 100.0)

# Sample data loader
if st.sidebar.button("🔁 Load Sample Fraud Case"):
    sample = {
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536346, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.5516, "V12": -0.6178,
        "V13": -0.9913, "V14": -0.3111, "V15": 1.4169, "V16": -0.4704,
        "V17": 0.2079, "V18": 0.0258, "V19": 0.4039, "V20": 0.2514,
        "V21": 0.0116, "V22": 0.1444, "V23": 0.0453, "V24": 0.3920,
        "V25": 0.4673, "V26": 0.4011, "V27": 0.1622, "V28": 0.0730,
        "Amount": 149.62
    }
    for k in sample:
        if k in inputs:
            inputs[k] = sample[k]

# Predict button
if st.button("🔍 Predict Fraud"):
    is_fraud, prob = predict_fraud(inputs)

    if is_fraud:
        st.error(f"🚨 Fraud Detected! Probability: {prob:.2%}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability: {prob:.2%}")

# Optional explanation
with st.expander("🔍 Show Feature Importance (Top 10)"):
    show_feature_importance()

st.markdown("---")
st.caption("Model: DecisionTreeClassifier | Threshold: 0.65 | F1 ≈ 0.82")
