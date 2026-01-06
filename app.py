import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Pharmevo | Sales Demand Predictor",
    page_icon="💊",
    layout="wide"
)

# =========================
# Load Model
# =========================
model = pickle.load(open("model.pkl", "rb"))

# =========================
# App Title
# =========================
st.markdown(
    "<h1 style='color:#1E88E5;'>Pharmevo Sales Demand Prediction System</h1>",
    unsafe_allow_html=True
)

st.write(
    "Predict pharmaceutical product demand (Units) based on historical sales patterns."
)

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🔍 Enter Sales Parameters")

product = st.sidebar.number_input("Product Code (Encoded)", min_value=0)
distributor = st.sidebar.number_input("Distributor Code (Encoded)", min_value=0)
client_type = st.sidebar.number_input("Client Type Code (Encoded)", min_value=0)
team = st.sidebar.number_input("Team Code (Encoded)", min_value=0)
month = st.sidebar.slider("Month", 1, 12)
price = st.sidebar.number_input("Price", min_value=0.0)
discount = st.sidebar.number_input("Discount", min_value=0.0)

# =========================
# Prediction
# =========================
if st.button("🔮 Predict Units"):
    input_data = np.array([[product, distributor, client_type, team, month, price, discount]])
    prediction = model.predict(input_data)

    st.success(f"📦 Predicted Units Sold: {int(prediction[0])}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("© 2026 Pharmevo Pharmaceuticals | ML Demand Forecasting System")
