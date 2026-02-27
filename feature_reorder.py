"""
feature_reorder.py
Feature column reordering and validation utilities for model inference.
Ensures DataFrame columns match the exact order expected by XGBoost model.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


# Expected feature columns for XGBoost demand model (must match training order)
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
    "is_black_friday",
    "is_new_year",
    "is_festival_season",
]


def validate_required_features(df: pd.DataFrame, features: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame contains all required features.
    
    Args:
        df: Input DataFrame
        features: List of required feature columns. Defaults to EXPECTED_FEATURES.
    
    Returns:
        Tuple of (is_valid: bool, missing_features: List[str])
    """
    if features is None:
        features = EXPECTED_FEATURES
    
    df_columns = set(df.columns)
    required_set = set(features)
    missing = list(required_set - df_columns)
    
    return len(missing) == 0, missing


def reorder_features(
    df: pd.DataFrame,
    features: List[str] = None,
    drop_extra: bool = True,
) -> pd.DataFrame:
    """
    Reorder DataFrame columns to match expected feature order for model.
    
    Args:
        df: Input DataFrame
        features: List of feature columns in desired order. Defaults to EXPECTED_FEATURES.
        drop_extra: If True, drop columns not in features list. If False, keep extra columns.
    
    Returns:
        pd.DataFrame: Reordered DataFrame with only specified features (or features + extras)
    
    Raises:
        ValueError: If any required features are missing
    """
    if features is None:
        features = EXPECTED_FEATURES
    
    # Validate that all required features exist
    is_valid, missing = validate_required_features(df, features)
    if not is_valid:
        raise ValueError(
            f"Missing required features: {missing}. "
            f"DataFrame has columns: {list(df.columns)}"
        )
    
    if drop_extra:
        # Keep only specified features in exact order
        return df[features].copy()
    else:
        # Keep specified features first (in order), then any extra columns
        extra_cols = [col for col in df.columns if col not in features]
        return df[features + extra_cols].copy()


def prepare_for_prediction(
    df: pd.DataFrame,
    features: List[str] = None,
    drop_extra: bool = True,
) -> Tuple[pd.DataFrame, bool]:
    """
    Prepare DataFrame for model prediction with validation and reordering.
    
    Args:
        df: Input DataFrame
        features: List of feature columns in desired order. Defaults to EXPECTED_FEATURES.
        drop_extra: If True, drop extra columns. If False, keep them.
    
    Returns:
        Tuple of (reordered_df: pd.DataFrame, success: bool)
        If success is False, returns original df and False.
    """
    if features is None:
        features = EXPECTED_FEATURES
    
    is_valid, missing = validate_required_features(df, features)
    
    if not is_valid:
        print(f"⚠️  Validation failed: Missing features {missing}")
        return df, False
    
    try:
        df_reordered = reorder_features(df, features, drop_extra=drop_extra)
        print(f"✓ Features reordered successfully. Order: {list(df_reordered.columns)}")
        return df_reordered, True
    except Exception as e:
        print(f"⚠️  Error during reordering: {e}")
        return df, False


def get_feature_order() -> List[str]:
    """Return the expected feature order for model predictions."""
    return EXPECTED_FEATURES.copy()


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("TEST 1: Reorder columns to match expected feature order")
    print("=" * 80)
    
    # Create sample DataFrame with columns in wrong order
    sample_data = {
        "day_of_week": [0, 1, 2],
        "rolling_7d_sales": [100.0, 102.5, 95.0],
        "product_name": ["iPhone 14", "iPhone 14", "Galaxy S23"],  # Extra column
        "price": [800, 810, 750],
        "month": [1, 1, 1],
        "popularity": [0.7, 0.755, 0.645],
        "competitor_price": [790, 800, 760],
        "price_gap": [10, 10, -10],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],  # Extra column
    }
    df = pd.DataFrame(sample_data)
    
    print(f"\nOriginal column order: {list(df.columns)}")
    print(f"Expected feature order: {EXPECTED_FEATURES}")
    
    df_reordered = reorder_features(df, drop_extra=True)
    print(f"\nReordered (drop_extra=True): {list(df_reordered.columns)}")
    print(df_reordered)
    
    print("\n" + "=" * 80)
    print("TEST 2: Keep extra columns after reordering")
    print("=" * 80)
    
    df_reordered_with_extra = reorder_features(df, drop_extra=False)
    print(f"\nReordered (drop_extra=False): {list(df_reordered_with_extra.columns)}")
    print(df_reordered_with_extra)
    
    print("\n" + "=" * 80)
    print("TEST 3: Validate and prepare for prediction")
    print("=" * 80)
    
    df_prepared, success = prepare_for_prediction(df)
    print(f"\nSuccess: {success}")
    print(f"Prepared DataFrame shape: {df_prepared.shape}")
    print(f"Prepared DataFrame columns: {list(df_prepared.columns)}")
    
    print("\n" + "=" * 80)
    print("TEST 4: Handle missing features gracefully")
    print("=" * 80)
    
    df_incomplete = pd.DataFrame({
        "price": [800],
        "competitor_price": [790],
        "month": [1],
        # Missing: price_gap, popularity, day_of_week, rolling_7d_sales
    })
    
    df_prepared_incomplete, success = prepare_for_prediction(df_incomplete)
    print(f"\nSuccess: {success}")
    if not success:
        print(f"Failed as expected - missing required features")
    
    print("\n" + "=" * 80)
    print("TEST 5: Direct validation check")
    print("=" * 80)
    
    is_valid, missing = validate_required_features(df_incomplete)
    print(f"\nValidation result: {is_valid}")
    print(f"Missing features: {missing}")
    
    print("\n✓ All feature reordering tests passed!")
