import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from price_optimizer import recommend_price, price_status

# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(page_title="Dynamic Pricing System", layout="wide")
st.title("📊 Dynamic Electronics Pricing Intelligence System")

# ---------------------------
# Load Data & Model
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "processed_pricing_data.csv")
model = joblib.load(BASE_DIR / "demand_model.pkl")

# ---------------------------
# Sidebar - Product Selection
# ---------------------------

st.sidebar.header("Product Selection")

product_list = df["product_name"].unique()
selected_product = st.sidebar.selectbox("Select Product", product_list)

product_df = df[df["product_name"] == selected_product]
latest_record = product_df.iloc[-1]

# ---------------------------
# Market Snapshot
# ---------------------------

st.subheader("📌 Current Market Snapshot")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Current Price", f"₹{latest_record['price'] * 91:.2f}")
col2.metric("Competitor Price", f"₹{latest_record['competitor_price'] * 91:.2f}")
col3.metric("Historical Demand (Observed Sales)", int(latest_record["historical_demand"]))
col4.metric("Cost", f"₹{latest_record['cost'] * 91:.2f}")
col5.metric("Popularity Score", round(latest_record["popularity"], 3))

# ---------------------------
# Price Simulation
# ---------------------------

st.subheader("📈 Price Simulation Analysis")

price_range = np.linspace(
    latest_record["price"] * 0.8,
    latest_record["price"] * 1.2,
    50
)

predicted_demand = []
predicted_profit = []

for price in price_range:

    input_data = pd.DataFrame([{
        "price": price,
        "competitor_price": latest_record["competitor_price"],
        "price_gap": price - latest_record["competitor_price"],
        "popularity": latest_record["popularity"],
        "month": latest_record["month"],
        "day_of_week": latest_record["day_of_week"],
        "rolling_7d_sales": latest_record["rolling_7d_sales"]
    }])

    forecasted_demand = model.predict(input_data)[0]
    profit = (price - latest_record["cost"]) * forecasted_demand

    predicted_demand.append(forecasted_demand)
    predicted_profit.append(profit)

# ---------------------------
# Optimal Price
# ---------------------------

optimal_price, expected_profit = recommend_price(latest_record)
status = price_status(latest_record["price"], optimal_price)

# ---------------------------
# Plot Demand Curve
# ---------------------------

fig1, ax1 = plt.subplots()
ax1.plot(price_range, predicted_demand)
ax1.axvline(optimal_price, linestyle='--')
ax1.set_title("Predicted Demand vs Price")
ax1.set_xlabel("Price")
ax1.set_ylabel("Predicted Demand")
st.pyplot(fig1)

# ---------------------------
# Plot Profit Curve
# ---------------------------

fig2, ax2 = plt.subplots()
ax2.plot(price_range, predicted_profit)
ax2.axvline(optimal_price, linestyle='--')
ax2.set_title("Profit vs Price")
ax2.set_xlabel("Price")
ax2.set_ylabel("Profit")
st.pyplot(fig2)

# ---------------------------
# Recommendation Section
# ---------------------------

st.subheader("🎯 Pricing Recommendation")

st.write(f"**Recommended Price:** ₹{optimal_price * 83:.2f}")
st.write(f"**Expected Profit:** ₹{expected_profit * 83:,.2f}")

if "Overpriced" in status:
    st.warning(status)
elif "Underpriced" in status:
    st.info(status)
else:
    st.success(status)