"""
preprocess_inference.py
Preprocessing utilities for Dynamic Pricing Intelligence System.
Used for preparing uploaded datasets for model inference.
"""

import pandas as pd
import numpy as np
from typing import Optional


def preprocess_for_inference(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Preprocess a DataFrame for model inference.
    
    Operations:
    1. Convert 'date' column to datetime
    2. Extract 'month' and 'day_of_week' from date
    3. Create 'price_gap' = price - competitor_price
    4. Compute 'popularity' from raw signals (search_trend, review_velocity, social_buzz)
       If signals missing, set popularity = 0.5 (neutral)
    5. Compute 'rolling_7d_sales' grouped by product_name
    6. Compute 'inventory_ratio' = inventory_level / max_inventory (if available)
       If inventory_level missing, set inventory_ratio = 0.5 (default)
    
    Args:
        df: Input DataFrame with pricing data
        inplace: If True, modify df in-place. If False, work on a copy.
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for model prediction
    """
    if not inplace:
        df = df.copy()
    
    # -------------------------
    # 1. Convert date to datetime
    # -------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Required column 'date' not found in DataFrame")
    
    # -------------------------
    # 2. Extract temporal features
    # -------------------------
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    
    # -------------------------
    # 3. Create price_gap feature
    # -------------------------
    if "price" not in df.columns or "competitor_price" not in df.columns:
        raise ValueError("Required columns 'price' and 'competitor_price' not found")
    
    df["price_gap"] = df["price"] - df["competitor_price"]
    
    # -------------------------
    # 4. Compute popularity
    # -------------------------
    required_signals = ["search_trend", "review_velocity", "social_buzz"]
    
    if all(col in df.columns for col in required_signals):
        # Compute popularity from signals (0 to ~1 scale)
        df["popularity"] = (
            0.4 * (df["search_trend"] / 100.0)
            + 0.3 * (df["review_velocity"] / 30.0)
            + 0.3 * (df["social_buzz"] / 100.0)
        )
        # Clip to [0, 1] range to be safe
        df["popularity"] = df["popularity"].clip(0.0, 1.0)
    else:
        # Set neutral popularity if signals not available
        df["popularity"] = 0.5
    
    # -------------------------
    # 5. Compute rolling 7-day sales momentum
    # -------------------------
    if "product_name" not in df.columns:
        raise ValueError("Required column 'product_name' not found")
    
    if "historical_demand" not in df.columns:
        raise ValueError("Required column 'historical_demand' not found")
    
    # Sort by product and date to ensure correct rolling calculation
    df = df.sort_values(["product_name", "date"]).reset_index(drop=True)
    
    # Compute rolling mean per product
    df["rolling_7d_sales"] = (
        df.groupby("product_name")["historical_demand"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )
    
    # -------------------------
    # 6. Compute inventory_ratio
    # -------------------------
    # Maximum inventory constant (must match training data)
    max_inventory = 500
    
    if "inventory_level" in df.columns:
        # Compute inventory_ratio if inventory_level is available
        df["inventory_ratio"] = df["inventory_level"] / max_inventory
        # Clip to valid range [0, 1]
        df["inventory_ratio"] = df["inventory_ratio"].clip(0.0, 1.0)
    else:
        # Set default inventory_ratio if inventory_level not available
        df["inventory_ratio"] = 0.5
    
    return df


def extract_features_for_model(df: pd.DataFrame, product_row: pd.Series) -> pd.DataFrame:
    """
    Extract and prepare features from a single product record for model prediction.
    
    Args:
        df: Full preprocessed DataFrame (for context, not always needed)
        product_row: Single row/Series with product data
    
    Returns:
        pd.DataFrame: Features ready for XGBoost model prediction
    """
    features = {
        "price": product_row.get("price", 0.0),
        "competitor_price": product_row.get("competitor_price", 0.0),
        "price_gap": product_row.get("price_gap", 0.0),
        "popularity": product_row.get("popularity", 0.5),
        "month": product_row.get("month", 1),
        "day_of_week": product_row.get("day_of_week", 0),
        "rolling_7d_sales": product_row.get("rolling_7d_sales", 0.0),
        "inventory_ratio": product_row.get("inventory_ratio", 0.5),
    }
    
    return pd.DataFrame([features])


if __name__ == "__main__":
    # Example usage (for testing)
    import pandas as pd
    
    # Sample data with all signals AND inventory_level
    sample_data_full = {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "product_name": ["iPhone 14", "iPhone 14", "Galaxy S23"],
        "price": [800, 810, 750],
        "competitor_price": [790, 800, 760],
        "cost": [500, 500, 450],
        "historical_demand": [100, 105, 95],
        "search_trend": [80.0, 85.0, 75.0],
        "review_velocity": [20.0, 22.0, 18.0],
        "social_buzz": [60.0, 65.0, 55.0],
        "inventory_level": [300, 295, 280],
    }
    df_full = pd.DataFrame(sample_data_full)
    
    print("=" * 80)
    print("TEST 1: Full dataset with all signals AND inventory_level")
    print("=" * 80)
    df_processed = preprocess_for_inference(df_full)
    print(df_processed[["date", "product_name", "price", "month", "day_of_week", 
                        "price_gap", "popularity", "rolling_7d_sales", "inventory_ratio"]].to_string())
    
    # Sample data without signals and without inventory_level (should use defaults)
    sample_data_minimal = {
        "date": ["2024-01-01", "2024-01-02"],
        "product_name": ["iPhone 14", "iPhone 14"],
        "price": [800, 810],
        "competitor_price": [790, 800],
        "cost": [500, 500],
        "historical_demand": [100, 105],
    }
    df_minimal = pd.DataFrame(sample_data_minimal)
    
    print("\n" + "=" * 80)
    print("TEST 2: Minimal dataset (no signals, no inventory_level, use defaults)")
    print("=" * 80)
    df_processed_minimal = preprocess_for_inference(df_minimal)
    print(df_processed_minimal[["date", "product_name", "price", "month", "day_of_week", 
                                "price_gap", "popularity", "rolling_7d_sales", "inventory_ratio"]].to_string())
    
    # Sample data with inventory_level but without signals
    sample_data_inventory_only = {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "product_name": ["iPhone 14", "iPhone 14", "Galaxy S23"],
        "price": [800, 810, 750],
        "competitor_price": [790, 800, 760],
        "cost": [500, 500, 450],
        "historical_demand": [100, 105, 95],
        "inventory_level": [400, 350, 200],
    }
    df_inventory_only = pd.DataFrame(sample_data_inventory_only)
    
    print("\n" + "=" * 80)
    print("TEST 3: With inventory_level but without signals (popularity=0.5 default)")
    print("=" * 80)
    df_processed_inventory = preprocess_for_inference(df_inventory_only)
    print(df_processed_inventory[["date", "product_name", "price", "month", "day_of_week", 
                                   "price_gap", "popularity", "rolling_7d_sales", "inventory_ratio"]].to_string())
    
    print("\n✓ All preprocessing tests passed!")
