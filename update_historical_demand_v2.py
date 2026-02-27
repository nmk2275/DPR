from pathlib import Path
import pandas as pd
import numpy as np

# Script: update_historical_demand_v2.py
# - Loads realistic_electronics_pricing_data.csv
# - Recalculates `historical_demand` using asymmetric competitive market model
# - Saves the updated CSV (overwrites original)

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "realistic_electronics_pricing_data.csv"

rng = np.random.default_rng(42)

# Load data
df = pd.read_csv(RAW_PATH)

# Ensure required columns exist and are numeric
required_cols = ["price", "competitor_price", "search_trend", "review_velocity", "social_buzz"]
for col in required_cols:
    if col not in df.columns:
        raise RuntimeError(f"Required column '{col}' not found in {RAW_PATH}")
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

# Compute popularity from raw signals (0 to ~1 scale)
df["popularity"] = (
    0.4 * (df["search_trend"] / 100.0)
    + 0.3 * (df["review_velocity"] / 30.0)
    + 0.3 * (df["social_buzz"] / 100.0)
)

# ========== BRAND-BASED ELASTICITY CLASSIFICATION ==========
# Define premium brands (low price elasticity)
premium_brands = {"Apple", "Sony", "Microsoft"}

# Use the existing 'brand' column directly
df["brand_category"] = df["brand"]

# Calculate base price per product (average cost as reference)
base_price_map = df.groupby("product_id")["cost"].mean().to_dict()
df["base_price"] = df["product_id"].map(base_price_map)

is_premium = df["brand_category"].isin(premium_brands)

# Generate per-product parameters with brand-based elasticity
product_ids = df["product_id"].unique()

params = []
for pid in product_ids:
    # Get product info to determine brand category
    product_rows = df[df["product_id"] == pid]
    brand = product_rows["brand"].iloc[0]
    
    base_demand = int(rng.integers(150, 301))               # 150-300 (same for all)
    
    # AGGRESSIVE brand-based elasticity parameters
    if brand in premium_brands:
        # Premium brands: low price sensitivity
        beta = float(rng.uniform(0.2, 0.4))                 # Low own-price elasticity (INCREASED)
        theta = float(rng.uniform(0.8, 1.5))                # Weak-to-moderate competitor penalty (INCREASED)
    else:
        # Non-premium brands: HIGH price sensitivity (AGGRESSIVE)
        beta = float(rng.uniform(1.5, 3.0))                 # High own-price elasticity (AGGRESSIVE 2.5x increase)
        theta = float(rng.uniform(3.0, 6.0))                # Strong competitor penalty (2x increase)
    
    gamma = float(rng.uniform(20.0, 60.0))                  # Popularity impact (same for all)
    
    params.append((pid, base_demand, beta, theta, gamma, brand))

params_df = pd.DataFrame(params, columns=["product_id", "base_demand", "beta", "theta", "gamma", "brand_category"])

# Merge params into main DF
df = df.merge(params_df[["product_id", "base_demand", "beta", "theta", "gamma"]], 
              on="product_id", how="left")

# Extract date information for temporal dynamics
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

# Noise epsilon per row
eps = rng.normal(loc=0.0, scale=5.0, size=len(df))

price = df["price"].to_numpy(dtype=float)
competitor_price = df["competitor_price"].to_numpy(dtype=float)
popularity = df["popularity"].to_numpy(dtype=float)
month = df["month"].to_numpy(dtype=int)
day_of_year = df["day_of_year"].to_numpy(dtype=int)

base = df["base_demand"].to_numpy(dtype=float)
beta = df["beta"].to_numpy(dtype=float)
theta = df["theta"].to_numpy(dtype=float)
gamma = df["gamma"].to_numpy(dtype=float)
base_price = df["base_price"].to_numpy(dtype=float)

# Normalize price around product base price (cost reference)
normalized_price = price / base_price

# Asymmetric competitive market model with scaled price (vectorized)
# Q = BaseDemand - beta * (normalized_price * 300) - theta * max(0, price - competitor_price) + gamma * popularity + eps
# The *300 scaling makes price effect comparable to demand scale (~300 units base demand)
price_gap_penalty = np.maximum(0.0, price - competitor_price)

Q = (
    base
    - beta * normalized_price * 300.0
    - theta * price_gap_penalty
    + gamma * popularity
    + eps
)

# ====== Regime Shift (after mid-year) ======
# If month >= 7, reduce base demand by 10%
regime_shift = np.where(month >= 7, 0.9, 1.0)
Q = Q * regime_shift

# ====== Gradual Market Drift ======
# drift_factor = 1 - (day_of_year / 1000)
# Represents declining market as year progresses
drift_factor = 1.0 - (day_of_year / 1000.0)
Q = Q * drift_factor

# ====== Occasional Shocks (2% probability) ======
# With 2% probability, multiply demand by random factor between 0.7 and 1.3
shock_prob = 0.02
shocks = rng.uniform(0.7, 1.3, size=len(df))
should_shock = rng.uniform(0.0, 1.0, size=len(df)) < shock_prob
Q = np.where(should_shock, Q * shocks, Q)

# Constraints: clip at minimum 5, convert to integer
Q = np.clip(Q, 5.0, None)
Q = np.round(Q).astype(int)

# Assign to historical_demand column
df["historical_demand"] = Q

# Drop temporary parameter columns before saving
df = df.drop(columns=["base_demand", "beta", "theta", "gamma", "base_price"])

# Save back to CSV (overwrite)
df.to_csv(RAW_PATH, index=False)

print(f"✓ Updated '{RAW_PATH.name}' with asymmetric competitive demand model ({len(df)} rows).")
print("\nSample records (showing price effects):")
print(df[["date","product_id","product_name","price","competitor_price","popularity","historical_demand"]].head(10).to_string(index=False))

print("\n" + "="*80)
print("Demand statistics by product:")
print("="*80)
agg = df.groupby("product_name")["historical_demand"].agg(["min", "mean", "max", "std"]).round(2)
print(agg.to_string())
