import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import optimizer helpers (these will load the trained model once on import)
from price_optimizer import recommend_price, price_status
import price_optimizer

# Import dataset validation
from validate_dataset import validate_dataset_streamlit

# Import preprocessing utilities
from preprocess_inference import preprocess_for_inference

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Dynamic Electronics Pricing Intelligence System",
    layout="wide",
)

# -------------------------
# Currency Configuration
# -------------------------
USD_TO_INR = 91  # Exchange rate: 1 USD = 91 INR

# -------------------------
# Helpers
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed_pricing_data.csv"

def get_latest_row_per_product(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the latest row per product (sorted by date).
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        pd.DataFrame: One row per product (latest by date)
    """
    if "date" not in df.columns:
        st.error("Date column not found in dataset")
        st.stop()
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Sort by product and date, then get the last row per product
    latest = df.sort_values("date").groupby("product_name", as_index=False).tail(1)
    
    return latest

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic safety: ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data
def load_data_from_bytes(file_bytes) -> pd.DataFrame:
    """Load CSV from uploaded file bytes."""
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Basic safety: ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# -------------------------
# UI - Title & Sidebar
# -------------------------

st.title("Dynamic Electronics Pricing Intelligence System")
st.markdown("---")

# Sidebar: file uploader and product selection
with st.sidebar:
    st.header("Data & Product Selection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV (optional)",
        type="csv",
        help="Upload a custom pricing dataset. If not provided, default dataset is used."
    )
    
    # Load dataset (from uploaded file or default)
    if uploaded_file is not None:
        try:
            df = load_data_from_bytes(uploaded_file.getvalue())
            st.success("✓ Custom dataset loaded")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            st.markdown("*Falling back to default dataset...*")
            try:
                df = load_data(DATA_PATH)
            except FileNotFoundError:
                st.error(f"Processed data not found at {DATA_PATH}. Run preprocess.py first.")
                st.stop()
    else:
        # Load default dataset
        try:
            df = load_data(DATA_PATH)
        except FileNotFoundError:
            st.error(f"Processed data not found at {DATA_PATH}. Run preprocess.py first.")
            st.stop()
    
    if df.empty:
        st.error("Loaded dataset is empty.")
        st.stop()
    
    # Validate dataset has required columns
    if not validate_dataset_streamlit(df):
        st.stop()
    
    st.divider()
    
    # Preprocess the dataset
    try:
        df_processed = preprocess_for_inference(df)
        st.success("✓ Dataset preprocessed")
    except Exception as e:
        st.error(f"Error preprocessing dataset: {e}")
        st.stop()
    
    # Get latest row per product
    df_latest = get_latest_row_per_product(df_processed)
    
    # Product selection form
    with st.form(key="product_select_form"):
        st.subheader("Product Selector")
        product_list = df_latest["product_name"].dropna().unique().tolist()
        product_list.sort()
        selected_product = st.selectbox("Select Product", product_list)
        show_raw = st.checkbox("Show raw latest record", value=False)
        st.form_submit_button("Load")

# -------------------------
# Latest record for selected product
# -------------------------
product_rows = df_latest[df_latest["product_name"] == selected_product]
if product_rows.empty:
    st.error(f"No data for product: {selected_product}")
    st.stop()

latest = product_rows.iloc[0]  # Only one row per product after get_latest_row_per_product()

# Optional raw display
if show_raw:
    st.subheader("Latest raw record")
    # Convert dollar amounts to rupees for display
    display_record = latest.to_dict()
    
    # Currency conversion rate
    USD_TO_INR = 91
    
    # Price-related columns to convert
    price_columns = ["price", "competitor_price", "cost"]
    
    for col in price_columns:
        if col in display_record:
            display_record[col] = display_record[col] * USD_TO_INR
    
    st.json(display_record)

# -------------------------
# Market Snapshot
# -------------------------
st.subheader("Market Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)

current_price = latest.get("price", np.nan)
competitor_price = latest.get("competitor_price", np.nan)
current_demand = latest.get("historical_demand", np.nan)
cost = latest.get("cost", np.nan)
popularity = latest.get("popularity", np.nan)

col1.metric("Current Price", f"₹{current_price * USD_TO_INR:.2f}" if pd.notna(current_price) else "N/A")
col2.metric("Competitor Price", f"₹{competitor_price * USD_TO_INR:.2f}" if pd.notna(competitor_price) else "N/A")
col3.metric("Historical Demand (Observed Sales)", f"{int(current_demand)}" if pd.notna(current_demand) else "N/A")
col4.metric("Cost", f"₹{cost * USD_TO_INR:.2f}" if pd.notna(cost) else "N/A")
col5.metric("Popularity", f"{popularity:.2f}" if pd.notna(popularity) else "N/A")

today = pd.Timestamp.now().date()
is_black_friday = (today.month == 11) and (today.day >= 21)
is_new_year = (today.month == 1) and (today.day <= 5)
is_festival = today.month in (10, 11, 12)
if is_black_friday:
    st.info("Today is Black Friday")
if is_new_year:
    st.info("Today is New Year")
if is_festival:
    st.info("Today is Festival Season")

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

# Cyclical encodings (model expects these)
month_num = int(month_val) if pd.notna(month_val) else 1
dow_num = int(day_of_week_val) if pd.notna(day_of_week_val) else 0
month_sin_val = np.sin(2 * np.pi * month_num / 12)
month_cos_val = np.cos(2 * np.pi * month_num / 12)
dow_sin_val = np.sin(2 * np.pi * dow_num / 7)
dow_cos_val = np.cos(2 * np.pi * dow_num / 7)

# Calendar flags (use latest date if available)
date_val = latest.get("date", pd.Timestamp.now())
if hasattr(date_val, "month") and hasattr(date_val, "day"):
    is_black_friday_val = 1 if (date_val.month == 11 and date_val.day >= 21) else 0
    is_new_year_val = 1 if (date_val.month == 1 and date_val.day <= 5) else 0
    is_festival_season_val = 1 if date_val.month in (10, 11, 12) else 0
else:
    is_black_friday_val = int(latest.get("is_black_friday", 0))
    is_new_year_val = int(latest.get("is_new_year", 0))
    is_festival_season_val = int(latest.get("is_festival_season", 0))

# Base feature values (will be aligned to model's expected feature names)
base_features = {
    "price": current_price,
    "competitor_price": competitor_price_val,
    "price_gap": current_price - competitor_price_val,
    "popularity": popularity_val,
    "month": month_num,
    "day_of_week": dow_num,
    "month_sin": month_sin_val,
    "month_cos": month_cos_val,
    "dow_sin": dow_sin_val,
    "dow_cos": dow_cos_val,
    "rolling_7d_sales": rolling_sales_val,
    "is_black_friday": is_black_friday_val,
    "is_new_year": is_new_year_val,
    "is_festival_season": is_festival_season_val,
}

# Predict demand using the loaded model (price_optimizer loaded model on import)
model = getattr(price_optimizer, "model", None)
if model is None:
    st.error("Demand model not loaded. Ensure demand_model.pkl exists and price_optimizer loads it.")
    st.stop()

# Align simulation inputs to the model's actual feature set and order
model_features = None
try:
    booster = model.get_booster()
    model_features = getattr(booster, "feature_names", None)
except Exception:
    model_features = getattr(model, "feature_names_in_", None)

if not model_features:
    # Fallback to the training-time feature list (kept in sync with train_model.py)
    model_features = [
        "price",
        "competitor_price",
        "price_gap",
        "popularity",
        "month",
        "day_of_week",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "rolling_7d_sales",
        "is_black_friday",
        "is_new_year",
        "is_festival_season",
    ]

data = {}
for name in model_features:
    if name == "price":
        data[name] = price_range
    elif name == "price_gap":
        data[name] = price_range - competitor_price_val
    else:
        # Broadcast scalar feature across the simulation grid
        value = base_features.get(name, 0.0)
        data[name] = np.full_like(price_range, value, dtype=float)

sim_inputs = pd.DataFrame(data)

forecasted_demand = model.predict(sim_inputs)
predicted_profit = (price_range - cost_val) * forecasted_demand

# Store results in DataFrame
sim_results = pd.DataFrame({
    "price": price_range,
    "forecasted_demand": forecasted_demand,
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
ax1.plot(sim_results["price"], sim_results["forecasted_demand"], color="#1f77b4", lw=2)
ax1.set_xlabel("Price")
ax1.set_ylabel("Predicted Demand")
ax1.set_title("Demand vs Price")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
if rec_price is not None:
    ax1.axvline(rec_price, color="green", linestyle="--", lw=2)
    ax1.annotate(f"Optimal: ₹{rec_price * USD_TO_INR:.2f}", xy=(rec_price, sim_results["forecasted_demand"].max()),
                 xytext=(rec_price, sim_results["forecasted_demand"].max()*0.9),
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
    ax2.annotate(f"Optimal: ₹{rec_price * USD_TO_INR:.2f}", xy=(rec_price, sim_results["predicted_profit"].max()),
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
        st.success(f"Recommended Price: ₹{rec_price * USD_TO_INR:.2f} — {status}")
    elif status and status.startswith("Overpriced"):
        st.warning(f"Recommended Price: ₹{rec_price * USD_TO_INR:.2f} — {status}")
    else:
        # Use transparent markdown instead of st.info()
        st.markdown(f"**Recommended Price:** ₹{rec_price * USD_TO_INR:.2f} — {status}")

    st.write(f"Expected Profit at recommended price: ₹{rec_profit * USD_TO_INR:.2f}")
    
    # -------- Simulated Adaptive Update (visual demonstration) --------
    st.divider()
    st.subheader("Adaptive Learning Simulation")
    
    # Simulate demand at optimal price
    optimal_price_input = pd.DataFrame([{
        name: (rec_price if name == "price" else 
                rec_price - competitor_price_val if name == "price_gap" else 
                base_features.get(name, 0.0))
        for name in model_features
    }])
    predicted_demand_at_optimal = model.predict(optimal_price_input)[0]
    
    # Simulate rolling sales update with exponential smoothing
    rolling_7d_sales = latest.get("rolling_7d_sales", 0.0)
    new_rolling_sales = 0.7 * rolling_7d_sales + 0.3 * predicted_demand_at_optimal
    
    # Display the update
    col1_update, col2_update = st.columns(2)
    with col1_update:
        st.metric("Previous Rolling 7-day Sales", f"{rolling_7d_sales:.2f} units")
    with col2_update:
        st.metric("Simulated Updated Rolling Sales", f"{new_rolling_sales:.2f} units")
    
    st.info("📊 Simulated adaptive update: rolling sales adjusted based on predicted demand at optimal price.")

# -------------------------
# Notes and footer
# -------------------------
st.markdown("---")
st.caption("Model predictions are based on a trained XGBoost demand model. Use recommendations as decision support, not absolute rules.")
