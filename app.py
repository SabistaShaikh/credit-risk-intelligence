import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Intelligence")

model = joblib.load("xgboost_model.joblib")

st.success("Model Loaded Successfully")
