import streamlit as st
import pandas as pd
import joblib

# ====================================
# CONFIGURATION
# ====================================
MODEL_PATH = "bounce_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

# ====================================
# STREAMLIT FRONTEND
# ====================================
st.title("Bounce Prediction Risk Scoring App")
st.write("Enter customer details to predict bounce risk score.")

# Input fields
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
no_of_unsecured_loan = st.number_input("Number of Unsecured Loans", min_value=0, max_value=20, value=2)
no_of_total_trades = st.number_input("Number of Total Trades", min_value=1, max_value=50, value=5)
avg_utilization_percent = st.number_input("Average Utilization Percentage", min_value=0, max_value=100, value=30)
dpd_crossed_times = st.number_input("30+ DPD Crossed Times (Last 3 Months)", min_value=0, max_value=10, value=0)

# Predict button
if st.button("Predict Risk Score"):
    # Create input dataframe
    input_data = pd.DataFrame({
        "CIBIL_SCORE": [cibil_score],
        "NO_OF_UNSECURED_LOAN": [no_of_unsecured_loan],
        "NO_OF_TOTAL_TRADES": [no_of_total_trades],
        "AVG_UTILIZATION_PERCENT": [avg_utilization_percent],
        "30_DPD_CROSSED_TIMES_3M": [dpd_crossed_times]
    })

    # Predict probability
    probability = model.predict_proba(input_data)[0][1]  # Probability of bounce

    # Determine risk level
    if probability <= 0.30:
        risk_flag = "Low Risk"
    elif probability <= 0.70:
        risk_flag = "Medium Risk"
    else:
        risk_flag = "High Risk"

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Bounce Probability:** {probability:.2f}")
    st.write(f"**Risk Category:** {risk_flag}")

    # Progress bar visualization
    st.progress(int(probability * 100))
