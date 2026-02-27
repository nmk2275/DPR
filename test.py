import numpy as np
import pandas as pd
import joblib

model = joblib.load("demand_model.pkl")

if 'df' not in globals():
    try:
        # Try processed data first, then realistic data, then sample data
        try:
            df = pd.read_csv("processed_pricing_data.csv")
        except FileNotFoundError:
            df = pd.read_csv("realistic_electronics_pricing_data.csv")
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Time features
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            
            # Cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Popularity
            df['popularity'] = (
                0.4 * (df['search_trend'] / 100) +
                0.3 * (df['review_velocity'] / 30) +
                0.3 * (df['social_buzz'] / 100)
            )
            
            # Price gap
            df['price_gap'] = df['price'] - df['competitor_price']
            
            # Rolling sales momentum
            df = df.sort_values(['product_id', 'date'])
            df['rolling_7d_sales'] = (
                df.groupby('product_id')['historical_demand']
                  .rolling(window=7, min_periods=1)
                  .mean()
                  .reset_index(0, drop=True)
            )
            
            # Demand volatility
            df['demand_volatility'] = (
                df.groupby('product_id')['historical_demand']
                  .rolling(window=7, min_periods=1)
                  .std()
                  .reset_index(0, drop=True)
            )
            df['demand_volatility'] = df['demand_volatility'].fillna(0)
            
            # Calendar event flags
            df['is_black_friday'] = ((df['date'].dt.month == 11) & 
                                     (df['date'].dt.day >= 21)).astype(int)
            df['is_new_year'] = ((df['date'].dt.month == 1) & 
                                 (df['date'].dt.day <= 5)).astype(int)
            df['is_festival_season'] = (df['date'].dt.month.isin([10, 11, 12])).astype(int)
    except FileNotFoundError:
        raise FileNotFoundError("Could not find processed_pricing_data.csv, realistic_electronics_pricing_data.csv, or sample_data.csv")

sample = df.iloc[-1]  # pick one row

prices = np.linspace(sample["price"] * 0.8,
                     sample["price"] * 1.2, 50)

demands = []

for p in prices:
    row = sample.copy()
    row["price"] = p
    row["price_gap"] = p - row["competitor_price"]

    X = pd.DataFrame([row[model.feature_names_in_]])
    demands.append(model.predict(X)[0])

import matplotlib.pyplot as plt
plt.plot(prices, demands)
plt.show()