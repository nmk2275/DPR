from pathlib import Path
import pandas as pd
import numpy as np

# Script: update_historical_demand.py
# - Loads realistic_electronics_pricing_data.csv
# - Recalculates `historical_demand` using a structural model per product
# - Saves the CSV back (overwrites file)

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "realistic_electronics_pricing_data.csv"

rng = np.random.default_rng(12345)

# Load data
df = pd.read_csv(RAW_PATH)

# Ensure price and competitor_price exist and are numeric
for col in ["price", "competitor_price", "search_trend", "review_velocity", "social_buzz"]:
    if col not in df.columns:
        raise RuntimeError(f"Required column '{col}' not found in {RAW_PATH}")
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

# Compute popularity from raw signals (same formula used elsewhere)
# popularity ranges roughly 0..1
df["popularity"] = (
    0.4 * (df["search_trend"] / 100.0)
    + 0.3 * (df["review_velocity"] / 30.0)
    + 0.3 * (df["social_buzz"] / 100.0)
)

# Prepare per-product parameters
product_ids = df["product_id"].unique()

params = []
for pid in product_ids:
    base_demand = int(rng.integers(150, 301))               # 150-300
    beta = float(rng.uniform(0.05, 0.2))                    # price sensitivity
    gamma = float(rng.uniform(20.0, 60.0))                 # popularity impact
    delta = float(rng.uniform(0.02, 0.1))                  # competitive effect
    params.append((pid, base_demand, beta, gamma, delta))

params_df = pd.DataFrame(params, columns=["product_id", "base_demand", "beta", "gamma", "delta"]) 

# Merge params into main DF
df = df.merge(params_df, on="product_id", how="left")

# Noise epsilon per row
eps = rng.normal(loc=0.0, scale=5.0, size=len(df))

# Structural demand model (vectorized)
# Q = BaseDemand - beta * price + gamma * popularity + delta * (competitor_price - price) + eps
price = df["price"].to_numpy(dtype=float)
competitor_price = df["competitor_price"].to_numpy(dtype=float)
popularity = df["popularity"].to_numpy(dtype=float)

base = df["base_demand"].to_numpy(dtype=float)
beta = df["beta"].to_numpy(dtype=float)
gamma = df["gamma"].to_numpy(dtype=float)
delta = df["delta"].to_numpy(dtype=float)

Q = base - beta * price + gamma * popularity + delta * (competitor_price - price) + eps

# Constraints: clip at minimum 5, convert to integer
Q = np.clip(Q, 5.0, None)
Q = np.round(Q).astype(int)

# Assign to historical_demand column (overwrite or create)
df["historical_demand"] = Q

# Drop temporary params columns before saving
df = df.drop(columns=["base_demand", "beta", "gamma", "delta"])

# Save back to CSV (overwrite)
df.to_csv(RAW_PATH, index=False)

print(f"Updated '{RAW_PATH.name}' with recalculated 'historical_demand' ({len(df)} rows).")
print("Sample:")
print(df[["date","product_id","product_name","price","competitor_price","popularity","historical_demand"]].head(6).to_string(index=False))
