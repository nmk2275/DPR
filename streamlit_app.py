import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import optimizer helpers (these will load the trained model once on import)
from price_optimizer import recommend_price, price_status
import price_optimizer

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Dynamic Electronics Pricing Intelligence System",
    layout="wide",
)

# -------------------------
# Helpers
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed_pricing_data.csv"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic safety: ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# Load dataset
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Processed data not found at {DATA_PATH}. Run preprocess.py first.")
    st.stop()

if df.empty:
    st.error("Processed dataset is empty.")
    st.stop()

# -------------------------
# UI - Title & Sidebar
# -------------------------

st.title("Dynamic Electronics Pricing Intelligence System")
st.markdown("---")

# Sidebar: product selection
with st.sidebar.form(key="product_select_form"):
    st.header("Product Selector")
    product_list = df["product_name"].dropna().unique().tolist()
    product_list.sort()
    selected_product = st.selectbox("Select Product", product_list)
    show_raw = st.checkbox("Show raw latest record", value=False)
    st.form_submit_button("Load")

# -------------------------
# Latest record for selected product
# -------------------------
product_rows = df[df["product_name"] == selected_product]
if product_rows.empty:
    st.error(f"No data for product: {selected_product}")
    st.stop()

latest = product_rows.sort_values("date").iloc[-1]

# Optional raw display
if show_raw:
    st.subheader("Latest raw record")
    st.json(latest.to_dict())

# -------------------------
# Market Snapshot
# -------------------------
st.subheader("Market Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)

current_price = latest.get("price", np.nan)
competitor_price = latest.get("competitor_price", np.nan)
current_demand = latest.get("quantity", np.nan)
cost = latest.get("cost", np.nan)
popularity = latest.get("popularity", np.nan)

col1.metric("Current Price", f"${current_price:.2f}" if pd.notna(current_price) else "N/A")
col2.metric("Competitor Price", f"${competitor_price:.2f}" if pd.notna(competitor_price) else "N/A")
col3.metric("Current Demand", f"{int(current_demand)}" if pd.notna(current_demand) else "N/A")
col4.metric("Cost", f"${cost:.2f}" if pd.notna(cost) else "N/A")
col5.metric("Popularity", f"{popularity:.2f}" if pd.notna(popularity) else "N/A")

st.markdown("---")

# -------------------------
# Price Simulation Engine
# -------------------------
st.subheader("Price Simulation")

# Build candidate price range
if pd.isna(current_price):
    st.error("Current price missing for this product.")
    st.stop()

price_range = np.linspace(current_price * 0.8, current_price * 1.2, 50)

# Prepare constants from latest row
competitor_price_val = latest.get("competitor_price", 0.0)
popularity_val = latest.get("popularity", 0.0)
month_val = latest.get("month", pd.NaT)
day_of_week_val = latest.get("day_of_week", 0)
rolling_sales_val = latest.get("rolling_7d_sales", 0.0)
cost_val = latest.get("cost", 0.0)

# Prepare input dataframe for batch prediction
sim_inputs = pd.DataFrame({
    "price": price_range,
    "competitor_price": competitor_price_val,
    "price_gap": price_range - competitor_price_val,
    "popularity": popularity_val,
    "month": month_val,
    "day_of_week": day_of_week_val,
    "rolling_7d_sales": rolling_sales_val,
})

# Predict demand using the loaded model (price_optimizer loaded model on import)
model = getattr(price_optimizer, "model", None)
if model is None:
    st.error("Demand model not loaded. Ensure demand_model.pkl exists and price_optimizer loads it.")
    st.stop()

predicted_demand = model.predict(sim_inputs)
predicted_profit = (price_range - cost_val) * predicted_demand

# Store results in DataFrame
sim_results = pd.DataFrame({
    "price": price_range,
    "predicted_demand": predicted_demand,
    "predicted_profit": predicted_profit,
})

# -------------------------
# Visualization
# -------------------------
st.subheader("Visualizations")

# Get recommendation from helper
try:
    rec_price, rec_profit = recommend_price(latest)
    status = price_status(latest.get("price", 0.0), rec_price)
except Exception as e:
    rec_price, rec_profit = None, None
    status = None
    st.warning(f"Recommendation could not be calculated: {e}")

# Demand vs Price
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(sim_results["price"], sim_results["predicted_demand"], color="#1f77b4", lw=2)
ax1.set_xlabel("Price")
ax1.set_ylabel("Predicted Demand")
ax1.set_title("Demand vs Price")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
if rec_price is not None:
    ax1.axvline(rec_price, color="green", linestyle="--", lw=2)
    ax1.annotate(f"Optimal: ${rec_price:.2f}", xy=(rec_price, sim_results["predicted_demand"].max()),
                 xytext=(rec_price, sim_results["predicted_demand"].max()*0.9),
                 arrowprops=dict(arrowstyle="->", color="green"), color="green")

# Profit vs Price
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(sim_results["price"], sim_results["predicted_profit"], color="#ff7f0e", lw=2)
ax2.set_xlabel("Price")
ax2.set_ylabel("Predicted Profit")
ax2.set_title("Profit vs Price")
ax2.grid(axis="y", linestyle="--", alpha=0.4)
if rec_price is not None:
    ax2.axvline(rec_price, color="green", linestyle="--", lw=2)
    ax2.annotate(f"Optimal: ${rec_price:.2f}", xy=(rec_price, sim_results["predicted_profit"].max()),
                 xytext=(rec_price, sim_results["predicted_profit"].max()*0.9),
                 arrowprops=dict(arrowstyle="->", color="green"), color="green")

# Layout the plots side by side
col_a, col_b = st.columns(2)
with col_a:
    st.pyplot(fig1)
with col_b:
    st.pyplot(fig2)

st.markdown("---")

# -------------------------
# Recommendation Panel
# -------------------------
st.subheader("Optimal Price Recommendation")
if rec_price is None:
    st.error("Could not compute recommendation.")
else:
    if status == "Price is Optimal":
        st.success(f"Recommended Price: ${rec_price:.2f} — {status}")
    elif status and status.startswith("Overpriced"):
        st.warning(f"Recommended Price: ${rec_price:.2f} — {status}")
    else:
        st.info(f"Recommended Price: ${rec_price:.2f} — {status}")

    st.write(f"Expected Profit at recommended price: ${rec_profit:.2f}")

# -------------------------
# Notes and footer
# -------------------------
st.markdown("---")
st.caption("Model predictions are based on a trained XGBoost demand model. Use recommendations as decision support, not absolute rules.")
