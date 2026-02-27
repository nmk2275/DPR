from pathlib import Path
import pandas as pd

# Resolve file paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "realistic_electronics_pricing_data.csv"
OUTPUT_CSV = BASE_DIR / "processed_pricing_data.csv"

# Load dataset
df = pd.read_csv(INPUT_CSV)

# Convert date
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# 1. Time Features
# -----------------------------

df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week

# -----------------------------
# 2. Compute Popularity from Raw Signals
# -----------------------------

df['popularity'] = (
    0.4 * (df['search_trend'] / 100) +
    0.3 * (df['review_velocity'] / 30) +
    0.3 * (df['social_buzz'] / 100)
)

# -----------------------------
# 3. Price Difference Feature
# -----------------------------

df['price_gap'] = df['price'] - df['competitor_price']

# -----------------------------
# 4. Rolling Sales Momentum
# -----------------------------

df = df.sort_values(['product_id', 'date'])

df['rolling_7d_sales'] = (
    df.groupby('product_id')['quantity']
      .rolling(window=7, min_periods=1)
      .mean()
      .reset_index(0, drop=True)
)

# -----------------------------
# Final Clean Dataset
# -----------------------------

print(df.head())
df.to_csv(OUTPUT_CSV, index=False)

print("Preprocessing complete.")