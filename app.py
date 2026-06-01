import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgboost_model.joblib")

st.set_page_config(page_title="Credit Risk Intelligence", page_icon="📊")

st.title("📊 Credit Risk Intelligence System")
st.write("Enter customer details to predict credit risk.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0.0, value=50000.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=5)
existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)

if st.button("Predict Risk"):

    data = pd.DataFrame({
        "age": [age],
        "income": [income],
        "loan_amount": [loan_amount],
        "credit_score": [credit_score],
        "employment_years": [employment_years],
        "existing_loans": [existing_loans]
    })

    prediction = model.predict(data)[0]

    st.subheader("Prediction Result")

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(data)[0][1]

        st.write(f"### Probability of Default: {probability:.2%}")

        if probability >= 0.70:
            st.error("🔴 High Credit Risk")
        elif probability >= 0.40:
            st.warning("🟡 Medium Credit Risk")
        else:
            st.success("🟢 Low Credit Risk")

    else:
        if prediction == 1:
            st.error("🔴 High Credit Risk")
        else:
            st.success("🟢 Low Credit Risk")
