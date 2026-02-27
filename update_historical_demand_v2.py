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

# Generate per-product parameters
product_ids = df["product_id"].unique()

params = []
for pid in product_ids:
    base_demand = int(rng.integers(150, 301))               # 150-300
    beta = float(rng.uniform(0.08, 0.2))                    # own price sensitivity
    theta = float(rng.uniform(0.3, 0.8))                    # competitor penalty (strong)
    gamma = float(rng.uniform(20.0, 60.0))                  # popularity impact
    params.append((pid, base_demand, beta, theta, gamma))

params_df = pd.DataFrame(params, columns=["product_id", "base_demand", "beta", "theta", "gamma"])

# Merge params into main DF
df = df.merge(params_df, on="product_id", how="left")

# Noise epsilon per row
eps = rng.normal(loc=0.0, scale=5.0, size=len(df))

# Asymmetric competitive market model (vectorized)
# Q = BaseDemand - beta * price - theta * max(0, price - competitor_price) + gamma * popularity + eps
price = df["price"].to_numpy(dtype=float)
competitor_price = df["competitor_price"].to_numpy(dtype=float)
popularity = df["popularity"].to_numpy(dtype=float)

base = df["base_demand"].to_numpy(dtype=float)
beta = df["beta"].to_numpy(dtype=float)
theta = df["theta"].to_numpy(dtype=float)
gamma = df["gamma"].to_numpy(dtype=float)

# Asymmetric penalty: only applies if we price above competitor
price_gap_penalty = np.maximum(0.0, price - competitor_price)

Q = base - beta * price - theta * price_gap_penalty + gamma * popularity + eps

# Constraints: clip at minimum 5, convert to integer
Q = np.clip(Q, 5.0, None)
Q = np.round(Q).astype(int)

# Assign to historical_demand column
df["historical_demand"] = Q

# Drop temporary parameter columns before saving
df = df.drop(columns=["base_demand", "beta", "theta", "gamma"])

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
