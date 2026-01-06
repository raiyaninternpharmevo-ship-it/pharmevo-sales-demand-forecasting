import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Pharmevo | Advanced Sales Intelligence",
    page_icon="💊",
    layout="wide"
)

# ===============================
# Load Model
# ===============================
model = pickle.load(open("model.pkl", "rb"))

# ===============================
# Header
# ===============================
st.markdown(
    "<h1 style='color:#0B5ED7;'>Pharmevo – Advanced Sales Prediction System</h1>",
    unsafe_allow_html=True
)
st.write(
    "An integrated ML system for demand forecasting, product ranking, "
    "client value identification, and sales risk assessment."
)

st.markdown("---")

# ===============================
# Sidebar – Prediction Type
# ===============================
prediction_type = st.sidebar.selectbox(
    "🔍 Select Prediction Type",
    [
        "Next Month Demand Forecast",
        "Best Product Next Month",
        "High-Value Client Identification",
        "Sales Drop Risk Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Pharmevo Pharmaceuticals")

# ===============================
# COMMON INPUTS
# ===============================
product = st.number_input("Product Code", min_value=0)
distributor = st.number_input("Distributor Code", min_value=0)
client_type = st.number_input("Client Type Code", min_value=0)
month = st.slider("Month", 1, 12)
price = st.number_input("Price", min_value=0.0)
discount = st.number_input("Discount", min_value=0.0)

input_data = np.array([[product, distributor, client_type, month, price, discount]])

# ===============================
# 1️⃣ DEMAND FORECAST
# ===============================
if prediction_type == "Next Month Demand Forecast":
    if st.button("🔮 Predict Demand"):
        units = model.predict(input_data)[0]
        st.success(f"📦 Predicted Units Sold Next Month: {int(units)}")

# ===============================
# 2️⃣ BEST PRODUCT
# ===============================
elif prediction_type == "Best Product Next Month":
    if st.button("🏆 Identify Best Product"):
        simulated_products = []
        for p in range(product, product + 5):
            temp_input = np.array([[p, distributor, client_type, month, price, discount]])
            units = model.predict(temp_input)[0]
            simulated_products.append((p, units))

        best_product = max(simulated_products, key=lambda x: x[1])
        st.success(
            f"🏆 Best Performing Product Code: {best_product[0]} "
            f"(Expected Units: {int(best_product[1])})"
        )

# ===============================
# 3️⃣ HIGH-VALUE CLIENT
# ===============================
elif prediction_type == "High-Value Client Identification":
    if st.button("👥 Evaluate Client Value"):
        units = model.predict(input_data)[0]

        if units >= 500:
            value = "High Value Client"
            color = "🟢"
        elif units >= 200:
            value = "Medium Value Client"
            color = "🟡"
        else:
            value = "Low Value Client"
            color = "🔴"

        st.info(f"{color} Client Segment: **{value}**")

# ===============================
# 4️⃣ SALES DROP RISK
# ===============================
elif prediction_type == "Sales Drop Risk Prediction":
    if st.button("⚠️ Assess Sales Risk"):
        units = model.predict(input_data)[0]

        if units < 150:
            st.error("⚠️ High Risk of Sales Drop Next Month")
        else:
            st.success("✅ Sales Performance Expected to be Stable")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption(
    "© 2026 Pharmevo Pharmaceuticals | Advanced ML Sales Intelligence Platform"
)
