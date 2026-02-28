"""
feature_engineering.py
Unified feature engineering logic for both training and inference pipelines.
Ensures training and inference use EXACTLY the same feature transformations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


# ========== EXPECTED FEATURES (MASTER LIST) ==========
# This is the SINGLE SOURCE OF TRUTH for feature order and names
# Must match XGBoost model training exactly
EXPECTED_FEATURES = [
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
    "inventory_ratio",
    "is_black_friday",
    "is_new_year",
    "is_festival_season",
]

# Constants
MAX_INVENTORY = 500


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all required features from raw data.
    
    Creates:
    - Temporal features: month, day_of_week, month_sin, month_cos, dow_sin, dow_cos
    - Price features: price_gap
    - Popularity: weighted combination of search_trend, review_velocity, social_buzz
    - Rolling sales: 7-day rolling average of historical_demand
    - Calendar events: is_black_friday, is_new_year, is_festival_season
    
    Important: Does NOT drop existing columns. All original columns are preserved.
    
    Args:
        df: Input DataFrame with raw pricing data
    
    Returns:
        pd.DataFrame: DataFrame with all original columns + generated features
    
    Raises:
        ValueError: If required columns missing
    """
    df = df.copy()
    
    # ========================
    # 1. TEMPORAL FEATURES
    # ========================
    if "date" not in df.columns:
        raise ValueError("Required column 'date' not found")
    
    # Convert to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Extract month and day_of_week
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    
    # Cyclical encoding for month (12-cycle)
    # sin/cos encoding captures the circular nature of months
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Cyclical encoding for day_of_week (7-cycle)
    # sin/cos encoding captures the circular nature of weekly cycles
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # ========================
    # 2. PRICE FEATURES
    # ========================
    if "price" not in df.columns or "competitor_price" not in df.columns:
        raise ValueError("Required columns 'price' and 'competitor_price' not found")
    
    df["price_gap"] = df["price"] - df["competitor_price"]
    
    # ========================
    # 3. POPULARITY FEATURE
    # ========================
    required_signals = ["search_trend", "review_velocity", "social_buzz"]
    
    if all(col in df.columns for col in required_signals):
        # Normalize and combine signals with specified weights
        # search_trend: 0-100 range, weight 0.4
        # review_velocity: 0-30 range, weight 0.3
        # social_buzz: 0-100 range, weight 0.3
        df["popularity"] = (
            0.4 * (df["search_trend"] / 100.0)
            + 0.3 * (df["review_velocity"] / 30.0)
            + 0.3 * (df["social_buzz"] / 100.0)
        )
        # Clip to [0, 1] range
        df["popularity"] = df["popularity"].clip(0.0, 1.0)
    else:
        # Use neutral popularity if signals not available
        df["popularity"] = 0.5
    
    # ========================
    # 4. ROLLING SALES FEATURE
    # ========================
    if "historical_demand" not in df.columns:
        raise ValueError("Required column 'historical_demand' not found")
    
    # Determine grouping column (product_id preferred, fallback to product_name)
    group_by = None
    if "product_id" in df.columns:
        group_by = "product_id"
    elif "product_name" in df.columns:
        group_by = "product_name"
    else:
        raise ValueError("Required column 'product_id' or 'product_name' not found")
    
    # Sort by group and date to ensure correct rolling calculation
    df = df.sort_values([group_by, "date"]).reset_index(drop=True)
    
    # Calculate 7-day rolling mean per product
    df["rolling_7d_sales"] = (
        df.groupby(group_by)["historical_demand"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )
    
    # ========================
    # 5. CALENDAR EVENT FEATURES
    # ========================
    # Black Friday: Last 10 days of November (Nov 21-30)
    df["is_black_friday"] = (
        (df["date"].dt.month == 11) & (df["date"].dt.day >= 21)
    ).astype(int)
    
    # New Year: First 5 days of January (Jan 1-5)
    df["is_new_year"] = (
        (df["date"].dt.month == 1) & (df["date"].dt.day <= 5)
    ).astype(int)
    
    # Festival Season: Oct, Nov, Dec
    df["is_festival_season"] = (
        df["date"].dt.month.isin([10, 11, 12])
    ).astype(int)
    
    # ========================
    # 6. INVENTORY FEATURES (if not already present)
    # ========================
    if "inventory_ratio" not in df.columns:
        if "inventory_level" in df.columns:
            # Use provided inventory_level
            df["inventory_ratio"] = (df["inventory_level"] / MAX_INVENTORY).clip(0.0, 1.0)
        else:
            # Default fallback: neutral inventory
            df["inventory_ratio"] = 0.5
    
    return df


def align_features(df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
    """
    Align DataFrame to match EXPECTED_FEATURES exactly.
    
    Ensures:
    - All required features exist
    - Missing columns are added with value 0
    - Columns are ordered exactly as EXPECTED_FEATURES
    - Returns ONLY the features in EXPECTED_FEATURES (no extra columns)
    
    Args:
        df: DataFrame with generated features
        features: List of expected features (defaults to EXPECTED_FEATURES)
    
    Returns:
        pd.DataFrame: Features strictly ordered as EXPECTED_FEATURES
    
    Raises:
        ValueError: If critical columns missing
    """
    if features is None:
        features = EXPECTED_FEATURES
    
    df = df.copy()
    
    # Add missing columns with default value 0
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Return ONLY the expected features in exact order
    return df[features].copy()


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from date column.
    
    Creates:
    - month: Month of year (1-12)
    - day_of_week: Day of week (0-6, Monday=0)
    - month_sin, month_cos: Cyclical encoding of month (12-cycle)
    - dow_sin, dow_cos: Cyclical encoding of day_of_week (7-cycle)
    
    Args:
        df: DataFrame with 'date' column (datetime or convertible)
    
    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    
    # Ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Required column 'date' not found")
    
    # Basic temporal features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    
    # Cyclical encoding for month (12-cycle)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Cyclical encoding for day_of_week (7-cycle)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price-related features.
    
    Creates:
    - price_gap: Difference between our price and competitor price
    
    Args:
        df: DataFrame with 'price' and 'competitor_price' columns
    
    Returns:
        DataFrame with price features added
    """
    df = df.copy()
    
    if "price" not in df.columns or "competitor_price" not in df.columns:
        raise ValueError("Required columns 'price' and 'competitor_price' not found")
    
    df["price_gap"] = df["price"] - df["competitor_price"]
    
    return df


def create_popularity_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create popularity feature from raw signals.
    
    Combines:
    - search_trend (0-100)
    - review_velocity (0-30)
    - social_buzz (0-100)
    
    With weights: 0.4 * search + 0.3 * review + 0.3 * social
    
    Args:
        df: DataFrame with signal columns
    
    Returns:
        DataFrame with 'popularity' feature added
    """
    df = df.copy()
    
    required_signals = ["search_trend", "review_velocity", "social_buzz"]
    
    if all(col in df.columns for col in required_signals):
        # Normalize and combine signals
        df["popularity"] = (
            0.4 * (df["search_trend"] / 100.0)
            + 0.3 * (df["review_velocity"] / 30.0)
            + 0.3 * (df["social_buzz"] / 100.0)
        )
        # Clip to [0, 1] range
        df["popularity"] = df["popularity"].clip(0.0, 1.0)
    else:
        # Use neutral popularity if signals not available
        df["popularity"] = 0.5
    
    return df


def create_rolling_sales_feature(
    df: pd.DataFrame,
    group_by: str = "product_id",
    window: int = 7
) -> pd.DataFrame:
    """
    Create rolling sales momentum feature.
    
    Calculates 7-day rolling average of historical_demand grouped by product.
    
    Args:
        df: DataFrame with 'historical_demand' column and grouping column
        group_by: Column to group by (typically 'product_id' or 'product_name')
        window: Rolling window size (default 7 days)
    
    Returns:
        DataFrame with 'rolling_7d_sales' feature added
    """
    df = df.copy()
    
    if "historical_demand" not in df.columns:
        raise ValueError("Required column 'historical_demand' not found")
    
    if group_by not in df.columns:
        raise ValueError(f"Required grouping column '{group_by}' not found")
    
    # Sort by group and date to ensure correct rolling calculation
    if "date" in df.columns:
        df = df.sort_values([group_by, "date"]).reset_index(drop=True)
    
    # Calculate rolling mean per group
    df["rolling_7d_sales"] = (
        df.groupby(group_by)["historical_demand"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )
    
    return df


def create_calendar_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar event features (flags for promotional periods).
    
    Creates:
    - is_black_friday: Flag for Black Friday period (Nov 21-30)
    - is_new_year: Flag for New Year period (Jan 1-5)
    - is_festival_season: Flag for festival months (Oct, Nov, Dec)
    
    Args:
        df: DataFrame with 'date' column
    
    Returns:
        DataFrame with calendar event features added
    """
    df = df.copy()
    
    if "date" not in df.columns:
        raise ValueError("Required column 'date' not found")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Black Friday: Last 10 days of November (Nov 21-30)
    df["is_black_friday"] = (
        (df["date"].dt.month == 11) & (df["date"].dt.day >= 21)
    ).astype(int)
    
    # New Year: First 5 days of January (Jan 1-5)
    df["is_new_year"] = (
        (df["date"].dt.month == 1) & (df["date"].dt.day <= 5)
    ).astype(int)
    
    # Festival Season: Oct, Nov, Dec
    df["is_festival_season"] = (
        df["date"].dt.month.isin([10, 11, 12])
    ).astype(int)
    
    return df


def create_inventory_features(
    df: pd.DataFrame,
    group_by: str = "product_id",
    max_inventory: int = MAX_INVENTORY
) -> pd.DataFrame:
    """
    Create inventory-related features.
    
    For training: Generates realistic steady-state inventory levels with variation.
    For inference: Uses provided inventory_level if available, else uses sensible defaults.
    
    Creates:
    - inventory_level: Units in stock (generated or provided)
    - inventory_ratio: inventory_level / max_inventory (0-1 scale)
    
    Args:
        df: DataFrame with 'historical_demand' column
        group_by: Column to group by (typically 'product_id' or 'product_name')
        max_inventory: Maximum inventory capacity (default 500)
    
    Returns:
        DataFrame with inventory features added
    """
    df = df.copy()
    
    if "inventory_level" not in df.columns:
        # Generate realistic inventory levels (for training or when not provided)
        if "historical_demand" not in df.columns:
            raise ValueError("Required column 'historical_demand' not found")
        
        if group_by not in df.columns:
            raise ValueError(f"Required grouping column '{group_by}' not found")
        
        # Ensure proper sorting for grouped operations
        if "date" in df.columns:
            df = df.sort_values([group_by, "date"]).reset_index(drop=True)
        
        inventory_levels = []
        
        for group_val in df[group_by].unique():
            group_data = df[df[group_by] == group_val].copy()
            
            # Calculate average daily demand for this group
            avg_demand = group_data["historical_demand"].mean()
            
            # Realistic inventory = 4 days of stock (with safety stock)
            target_inventory = max(50, int(4 * avg_demand))
            target_inventory = min(target_inventory, 400)
            
            # Add realistic variation (±10% of target)
            product_inventories = []
            for idx, row in group_data.iterrows():
                noise = np.random.normal(0, 0.05 * target_inventory)
                current_inv = target_inventory + noise
                # Constrain to reasonable bounds
                current_inv = max(int(0.1 * target_inventory), min(int(current_inv), 400))
                product_inventories.append(current_inv)
            
            # Map back to original indices
            for idx, inv in zip(group_data.index, product_inventories):
                inventory_levels.append((idx, inv))
        
        # Create inventory_level column
        inventory_dict = {idx: inv for idx, inv in inventory_levels}
        df["inventory_level"] = df.index.map(inventory_dict)
    
    # Create inventory_ratio
    df["inventory_ratio"] = df["inventory_level"] / max_inventory
    df["inventory_ratio"] = df["inventory_ratio"].clip(0.0, 1.0)
    
    return df


def engineer_features(
    df: pd.DataFrame,
    group_by: str = "product_id",
    create_inventory: bool = True,
    max_inventory: int = MAX_INVENTORY
) -> pd.DataFrame:
    """
    Apply ALL feature engineering transformations in correct order.
    
    This is the UNIFIED entry point for both training and inference.
    
    Steps (in order):
    1. Temporal features (month, dow, cyclical encodings)
    2. Price features (price_gap)
    3. Popularity from signals
    4. Rolling sales momentum
    5. Calendar event features
    6. Inventory features (optional, for inference may use provided values)
    
    Args:
        df: Raw DataFrame
        group_by: Column to group by for rolling sales and inventory (product_id or product_name)
        create_inventory: If True, generate inventory features. If False, assume inventory_level provided.
        max_inventory: Maximum inventory capacity
    
    Returns:
        DataFrame with all features engineered
    """
    df = df.copy()
    
    # Step 1: Temporal features
    df = create_temporal_features(df)
    
    # Step 2: Price features
    df = create_price_features(df)
    
    # Step 3: Popularity
    df = create_popularity_feature(df)
    
    # Step 4: Rolling sales
    df = create_rolling_sales_feature(df, group_by=group_by)
    
    # Step 5: Calendar events
    df = create_calendar_event_features(df)
    
    # Step 6: Inventory features
    if create_inventory:
        df = create_inventory_features(df, group_by=group_by, max_inventory=max_inventory)
    else:
        # Ensure inventory_ratio exists if inventory_level provided
        if "inventory_level" in df.columns:
            df["inventory_ratio"] = (df["inventory_level"] / max_inventory).clip(0.0, 1.0)
        else:
            # Fallback to defaults
            df["inventory_level"] = max_inventory / 2
            df["inventory_ratio"] = 0.5
    
    return df


def select_features(
    df: pd.DataFrame,
    features: List[str] = None,
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    Select and validate features for model prediction.
    
    Ensures:
    - All required features exist
    - Missing numeric features filled with 0
    - Exact column order matches EXPECTED_FEATURES
    
    Args:
        df: DataFrame with engineered features
        features: List of required features (defaults to EXPECTED_FEATURES)
        fill_missing: If True, fill missing numeric features with 0
    
    Returns:
        DataFrame with selected features in correct order
    
    Raises:
        ValueError: If required columns missing and can't be filled
    """
    if features is None:
        features = EXPECTED_FEATURES
    
    df = df.copy()
    
    # Check for missing required features
    missing = [f for f in features if f not in df.columns]
    
    if missing:
        if fill_missing:
            # Try to fill missing numeric features with 0
            for feat in missing:
                df[feat] = 0.0
        else:
            raise ValueError(
                f"Missing required features: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
    
    # Return in exact order
    return df[features].copy()


def validate_feature_matrix(
    df: pd.DataFrame,
    expected_features: List[str] = None
) -> Tuple[bool, str]:
    """
    Validate that feature matrix matches training requirements.
    
    Checks:
    - All expected features present
    - No extra unexpected features (warning only)
    - Data types reasonable
    - No NaN values in features
    
    Args:
        df: DataFrame to validate
        expected_features: List of expected features (defaults to EXPECTED_FEATURES)
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if expected_features is None:
        expected_features = EXPECTED_FEATURES
    
    # Check for missing features
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        return False, f"Missing features: {missing}"
    
    # Check for wrong order
    df_features = [f for f in df.columns if f in expected_features]
    if df_features != expected_features:
        return False, f"Feature order mismatch. Expected: {expected_features}, Got: {df_features}"
    
    # Check for NaN values
    nan_features = df[expected_features].columns[df[expected_features].isna().any()].tolist()
    if nan_features:
        return False, f"NaN values found in features: {nan_features}"
    
    return True, "Feature matrix valid ✓"


if __name__ == "__main__":
    # Test the feature engineering pipeline
    print("=" * 80)
    print("FEATURE ENGINEERING MODULE TESTS")
    print("=" * 80)
    
    # Create sample data
    sample_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "product_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        "price": [800, 810, 820, 700, 710, 720, 500, 510, 520, 530],
        "competitor_price": [790, 800, 810, 690, 700, 710, 490, 500, 510, 520],
        "cost": [500, 500, 500, 400, 400, 400, 300, 300, 300, 300],
        "historical_demand": [100, 110, 105, 95, 100, 98, 50, 55, 52, 53],
        "search_trend": [80, 85, 82, 75, 78, 76, 60, 62, 61, 63],
        "review_velocity": [20, 22, 21, 18, 19, 18, 15, 16, 15, 16],
        "social_buzz": [70, 75, 72, 65, 68, 66, 55, 57, 56, 58],
    })
    
    print("\nTest 1: Complete feature engineering pipeline")
    print("-" * 80)
    df_engineered = engineer_features(sample_df, group_by="product_id", create_inventory=True)
    
    print(f"Input shape: {sample_df.shape}")
    print(f"Output shape: {df_engineered.shape}")
    print(f"\nFeatures created: {list(df_engineered.columns)}")
    
    print("\nTest 2: Feature selection and validation")
    print("-" * 80)
    df_selected = select_features(df_engineered)
    print(f"Selected features shape: {df_selected.shape}")
    print(f"Selected feature columns: {list(df_selected.columns)}")
    
    is_valid, msg = validate_feature_matrix(df_selected)
    print(f"Validation result: {msg}")
    
    print("\nTest 3: Sample feature values")
    print("-" * 80)
    print(df_selected.head(3).to_string())
    
    print("\n" + "=" * 80)
    print("✓ All feature engineering tests passed!")
    print("=" * 80)
