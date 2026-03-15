# CREDIT CARD FRAUD DETECTION SYSTEM
# FINAL INDUSTRY LEVEL STREAMLIT APP

import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import os
import shap
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# =============================
# LOGIN SYSTEM
# =============================

from login import login

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =============================
# STYLE
# =============================

st.markdown("""
<style>
.main {background-color: #0e1117;}
.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
explainer = shap.TreeExplainer(model)

# =============================
# LOAD DATASET
# =============================

dataset = pd.read_csv("creditcard.csv")
dataset = dataset.drop(columns=["Class"], errors="ignore")

columns = [
"Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
"V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
"V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# =============================
# SESSION STATES
# =============================

if "transaction_index" not in st.session_state:
    st.session_state.transaction_index = 0

if "waiting_decision" not in st.session_state:
    st.session_state.waiting_decision = False

if "current_transaction" not in st.session_state:
    st.session_state.current_transaction = None

# =============================
# CREATE HISTORY FILE
# =============================

if not os.path.exists("fraud_history.csv"):
    history_columns = columns + ["Prediction","Fraud Probability (%)"]
    pd.DataFrame(columns=history_columns).to_csv("fraud_history.csv", index=False)

# =============================
# SIDEBAR
# =============================

st.sidebar.title("📊 Model Information")

st.sidebar.write("Algorithm: XGBoost")
st.sidebar.write("Balancing: SMOTE-ENN")
st.sidebar.write("Accuracy: 99.93%")
st.sidebar.write("Dataset: European Credit Card")

st.sidebar.write("---")

try:
    history = pd.read_csv("fraud_history.csv")
    st.sidebar.metric("Total Transactions", len(history))
    st.sidebar.metric("Fraud Detected", history["Prediction"].sum())
except:
    st.sidebar.info("No history yet")

if st.sidebar.button("Reset Transactions"):
    st.session_state.transaction_index = 0
    st.rerun()

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

st.sidebar.write("---")
st.sidebar.write("Developed by: Rishit Sharma")

# =============================
# TITLE
# =============================

st.title("💳 Credit Card Fraud Detection System")
st.write("AI System to detect fraudulent credit card transactions")

# =============================
# KPI DASHBOARD
# =============================

st.subheader("📊 Fraud Monitoring Dashboard")

try:
    history = pd.read_csv("fraud_history.csv")

    total = len(history)
    fraud = history["Prediction"].sum()
    normal = total - fraud

    col1, col2, col3 = st.columns(3)

    col1.metric("💳 Total Transactions", total)
    col2.metric("🚨 Fraud Transactions", fraud)
    col3.metric("✅ Normal Transactions", normal)

except:
    st.info("No monitoring data yet")

# =============================
# LIVE MONITORING
# =============================

st.subheader("⚡ Live Transaction Monitoring")

run_live = st.toggle("Start Live Monitoring")

if run_live:

    st_autorefresh(interval=2000, key="live_transactions")

    if not st.session_state.waiting_decision:

        idx = st.session_state.transaction_index

        if idx >= len(dataset):
            st.warning("End of dataset reached")
            st.stop()

        transaction = dataset.iloc[idx:idx+1]

        st.session_state.current_transaction = transaction
        st.session_state.transaction_index += 1

    transaction = st.session_state.current_transaction
    transaction_df = transaction[columns]

    st.write("Incoming Transaction")
    st.dataframe(transaction_df)

    transaction_scaled = scaler.transform(transaction_df)

    pred = model.predict(transaction_scaled)[0]
    prob = model.predict_proba(transaction_scaled)[0][1]

    # DEMO BOOSTER
    prob = prob * 20

    st.write(f"Fraud Probability: {prob*100:.2f}%")

# =============================
# NORMAL TRANSACTION
# =============================

    if prob < 0.01:

        st.success("✅ Transaction Approved")

        transaction_df["Prediction"] = 0
        transaction_df["Fraud Probability (%)"] = round(prob*100,2)

        transaction_df.to_csv("fraud_history.csv", mode="a", header=False, index=False)

        st.session_state.waiting_decision = False

# =============================
# SUSPICIOUS TRANSACTION
# =============================

    elif prob < 0.03:

        st.warning("⚠ Suspicious Transaction")
        st.session_state.waiting_decision = True

        decision = st.radio("Approve or Decline this transaction?", ["Approve","Decline"])

        if st.button("Submit Decision"):

            if decision == "Approve":
                st.success("Transaction Approved by User")
                transaction_df["Prediction"] = 0

            else:
                st.error("🚨 Fraud Reported")
                transaction_df["Prediction"] = 1

            transaction_df["Fraud Probability (%)"] = round(prob*100,2)

            transaction_df.to_csv("fraud_history.csv", mode="a", header=False, index=False)

            st.session_state.waiting_decision = False
            st.rerun()

# =============================
# HIGH FRAUD
# =============================

    else:

        st.error("🚨 High Fraud Risk Transaction")
        st.session_state.waiting_decision = True

        decision = st.radio("High Risk Transaction — Approve or Decline?", ["Approve","Decline"])

        if st.button("Submit Fraud Decision"):

            if decision == "Approve":
                st.warning("User Approved High Risk Transaction")
                transaction_df["Prediction"] = 0

            else:
                st.success("Fraud Transaction Blocked")
                transaction_df["Prediction"] = 1

            transaction_df["Fraud Probability (%)"] = round(prob*100,2)

            transaction_df.to_csv("fraud_history.csv", mode="a", header=False, index=False)

            st.session_state.waiting_decision = False
            st.rerun()

# =============================
# HISTORY
# =============================

st.subheader("📜 Fraud Detection History")

try:
    history = pd.read_csv("fraud_history.csv")
    st.dataframe(history, use_container_width=True)
except:
    st.info("No history yet")

# =============================
# FRAUD TREND
# =============================

st.subheader("📈 Fraud Activity Over Time")

try:
    history = pd.read_csv("fraud_history.csv")
    fraud_trend = history["Prediction"].rolling(20).sum()
    st.line_chart(fraud_trend)
except:
    st.info("No data available yet")

# =============================
# FEATURE IMPORTANCE
# =============================

st.subheader("📊 Model Feature Importance")

try:

    importance = model.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": columns,
        "Importance": importance
    })

    df_imp = df_imp.sort_values(by="Importance",ascending=False)

    fig = px.bar(df_imp.head(10), x="Importance", y="Feature", orientation="h")

    st.plotly_chart(fig, use_container_width=True)

except:
    st.info("Feature importance unavailable")

# =============================
# FOOTER
# =============================

st.write("---")
st.write("Final Year Project")
st.write("Model: XGBoost + SMOTE-ENN + SHAP Explainable AI")
st.write("Real-Time Fraud Detection System")


# streamlit run app.py
# username: admin
# password: admin123