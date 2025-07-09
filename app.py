import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit settings
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Credit Card Fraud Detection App")

# ℹ️ About Section
with st.expander("📘 What does this app do?"):
    st.markdown("""
    This app detects potential **credit card fraud** using a machine learning model trained on the public `creditcard.csv` dataset.

    **How it works:**
    - Each row in your uploaded CSV is one transaction.
    - The columns `V1` to `V28` are anonymized PCA features.
    - `Amount` is the only raw feature used directly.
    - The model outputs whether it's a fraud or legit transaction.

    🔍 Upload a CSV file with **V1 to V28 and Amount** to get predictions.
    """)

# 📤 File Upload Section
st.subheader("📤 Upload CSV File")
uploaded_file = st.file_uploader("Upload a file with columns: V1 to V28 and Amount", type=["csv"])

# 📄 Sample Input + Guidance
st.subheader("📄 Sample Input")
st.markdown("""
⚠️ **Note**: This model only works with data in the same format as the public `creditcard.csv` dataset  
(V1 to V28 + Amount) — all PCA-anonymized features.

📂 [Click here to download a sample input CSV](https://drive.google.com/file/d/19LX8BcbmDEdqgiuPPERi7bmb3goKSdeB/view?usp=sharing) to test the app.

""")

# Required feature names
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# 🧠 Prediction Section
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Check if all required columns are present
    if all(col in data.columns for col in feature_names):
        st.success("✅ File loaded successfully!")

        # Preprocess
        X = data[feature_names]
        X_scaled = scaler.transform(X)

        # Predict
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        # Append to dataframe
        data["Fraud Probability"] = probs
        data["Prediction"] = preds
        data["Prediction"] = data["Prediction"].map({0: "Legit", 1: "Fraud"})

        # Show predictions
        st.subheader("🔍 Prediction Results")
        show_all = st.checkbox("Show full results", value=False)

        if show_all:
            st.dataframe(data[["Prediction", "Fraud Probability"] + feature_names])
        else:
            st.dataframe(data[["Prediction", "Fraud Probability"] + feature_names].head(10))

        # Summary
        fraud_count = int(sum(preds))
        legit_count = int(len(preds) - fraud_count)
        st.info(f"📊 Detected **{fraud_count}** fraudulent and **{legit_count}** legitimate transactions.")

        # 📥 Download predictions
        st.subheader("📥 Download Results")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )
    else:
        st.error("❌ Uploaded file is missing required columns (V1 to V28 and Amount).")
else:
    st.warning("Upload a CSV file to get started.")

# ✅ Watermark at bottom-center
st.markdown("""
<style>
.watermark {
    position: fixed;
    bottom: 12px;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0.7;
    font-size: 16px;
    color: gray;
    z-index: 100;
}
</style>
<div class="watermark">Developed by Abhijnan Raj</div>
""", unsafe_allow_html=True)
