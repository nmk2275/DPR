from pathlib import Path
import pandas as pd
import numpy as np

# Resolve file paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "realistic_electronics_pricing_data.csv"
OUTPUT_CSV = BASE_DIR / "processed_pricing_data.csv"

# Load dataset
df = pd.read_csv(INPUT_CSV)

# If raw file uses 'quantity', rename to 'historical_demand' for clarity
if 'quantity' in df.columns and 'historical_demand' not in df.columns:
    df = df.rename(columns={'quantity': 'historical_demand'})

# Convert date
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# 1. Time Features
# -----------------------------

df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week

# Cyclical Seasonality Encoding
# Month: 12-cycle (sin/cos transformation)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Day of Week: 7-cycle (sin/cos transformation)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

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
    df.groupby('product_id')['historical_demand']
      .rolling(window=7, min_periods=1)
      .mean()
      .reset_index(0, drop=True)
)

# Rolling 7-day demand volatility (standard deviation)
df['demand_volatility'] = (
    df.groupby('product_id')['historical_demand']
      .rolling(window=7, min_periods=1)
      .std()
      .reset_index(0, drop=True)
)

# Fill missing values with 0
df['demand_volatility'] = df['demand_volatility'].fillna(0)

# -----------------------------
# 5. Calendar Event Flags
# -----------------------------

# Black Friday: Last 10 days of November (Nov 21-30)
df['is_black_friday'] = ((df['date'].dt.month == 11) & 
                         (df['date'].dt.day >= 21)).astype(int)

# New Year: First 5 days of January (Jan 1-5)
df['is_new_year'] = ((df['date'].dt.month == 1) & 
                     (df['date'].dt.day <= 5)).astype(int)

# Festival Season: Months 10, 11, 12 (Oct, Nov, Dec)
df['is_festival_season'] = (df['date'].dt.month.isin([10, 11, 12])).astype(int)

# -----------------------------
# Final Clean Dataset
# -----------------------------

print(df.head())
df.to_csv(OUTPUT_CSV, index=False)

print("Preprocessing complete.")