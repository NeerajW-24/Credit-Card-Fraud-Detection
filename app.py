# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

st.title("💳 Credit Card Fraud Detection System (Enhanced)")
st.markdown("Upload a CSV or enter a single transaction manually to detect potential fraud.")

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prediction function
def predict_fraud(input_df):
    df = input_df.copy()
    df['Amount'] = scaler.transform(df[['Amount']])
    if 'Time' in df.columns:
        df.drop('Time', axis=1, inplace=True)
    probs = model.predict_proba(df)[:, 1]
    preds = (probs > 0.35).astype(int)
    return preds, probs

# --- Sidebar Input ---
st.sidebar.header("🔢 Manual Transaction Input")
manual_input = {}
for i in range(1, 29):
    manual_input[f"V{i}"] = st.sidebar.slider(f"V{i}", -30.0, 30.0, 0.0)
manual_input["Amount"] = st.sidebar.slider("Amount", 0.0, 5000.0, 100.0)

if st.sidebar.button("🔍 Predict Single Transaction"):
    single_df = pd.DataFrame([manual_input])
    pred, prob = predict_fraud(single_df)
    if pred[0] == 1:
        st.error(f"🚨 Fraud Detected! Probability: {prob[0]:.2%}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability: {prob[0]:.2%}")

# --- CSV Upload Section ---
st.header("📂 Upload Transactions (CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with columns V1–V28 and Amount", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if 'Amount' not in df.columns or any(f"V{i}" not in df.columns for i in range(1, 29)):
        st.error("❌ CSV must include 'Amount' and V1 to V28 columns.")
    else:
        preds, probs = predict_fraud(df)
        df['is_fraud'] = preds
        df['probability'] = probs

        # Display results
        st.subheader("🔎 Predictions")
        st.write(df.head())

        # Chart
        st.subheader("📊 Fraud vs Legitimate")
        chart_data = df['is_fraud'].value_counts().rename(index={0: 'Legit', 1: 'Fraud'})
        st.bar_chart(chart_data)

        # Download button
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Results as CSV", csv, "fraud_predictions.csv", "text/csv")

# --- Feature Importance ---
st.markdown("---")
with st.expander("📈 Show Feature Importances"):
    importances = model.feature_importances_
    features = model.feature_names_in_
    sorted_idx = np.argsort(importances)[-10:]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importances[sorted_idx], y=features[sorted_idx], ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.caption("Model: DecisionTreeClassifier | Threshold: 0.65 | F1 ≈ 0.82")
