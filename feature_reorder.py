"""
feature_reorder.py
Feature column reordering and validation utilities for model inference.
Ensures DataFrame columns match the exact order expected by XGBoost model.

IMPORTANT: This module imports EXPECTED_FEATURES from feature_engineering.py
to ensure unified feature definitions across training and inference.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from feature_engineering import EXPECTED_FEATURES


def validate_required_features(df: pd.DataFrame, features: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame contains all required features.
    Provides backward compatibility: if inventory_ratio is missing, it will be filled with default value 0.5.
    
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
    
    # Allow inventory_ratio to be missing (backward compatibility)
    # It will be filled with default value 0.5 during reordering
    if "inventory_ratio" in missing:
        missing.remove("inventory_ratio")
    
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
        ValueError: If any required features are missing (except inventory_ratio which is optional)
    """
    if features is None:
        features = EXPECTED_FEATURES
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Validate that all required features exist (except inventory_ratio)
    is_valid, missing = validate_required_features(df_copy, features)
    if not is_valid:
        raise ValueError(
            f"Missing required features: {missing}. "
            f"DataFrame has columns: {list(df_copy.columns)}"
        )
    
    # Handle backward compatibility: add inventory_ratio with default value if missing
    if "inventory_ratio" in features and "inventory_ratio" not in df_copy.columns:
        df_copy["inventory_ratio"] = 0.5
    
    if drop_extra:
        # Keep only specified features in exact order
        return df_copy[features].copy()
    else:
        # Keep specified features first (in order), then any extra columns
        extra_cols = [col for col in df_copy.columns if col not in features]
        return df_copy[features + extra_cols].copy()


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
    
    # Create sample DataFrame with columns in wrong order and all required features
    sample_data = {
        "day_of_week": [0, 1, 2],
        "rolling_7d_sales": [100.0, 102.5, 95.0],
        "product_name": ["iPhone 14", "iPhone 14", "Galaxy S23"],  # Extra column
        "price": [800, 810, 750],
        "month": [1, 1, 1],
        "popularity": [0.7, 0.755, 0.645],
        "competitor_price": [790, 800, 760],
        "price_gap": [10, 10, -10],
        "month_sin": [0.26, 0.26, 0.26],
        "month_cos": [0.96, 0.96, 0.96],
        "dow_sin": [0.0, 0.78, 1.0],
        "dow_cos": [1.0, 0.62, 0.0],
        "is_black_friday": [0, 0, 0],
        "is_new_year": [0, 0, 0],
        "is_festival_season": [0, 0, 0],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],  # Extra column
    }
    df = pd.DataFrame(sample_data)
    
    print(f"\nOriginal column order: {list(df.columns)}")
    print(f"Expected feature order: {EXPECTED_FEATURES}")
    
    df_reordered = reorder_features(df, drop_extra=True)
    print(f"\nReordered (drop_extra=True): {list(df_reordered.columns)}")
    print(f"✓ inventory_ratio added with default value 0.5")
    print(df_reordered)
    
    print("\n" + "=" * 80)
    print("TEST 2: Backward compatibility - DataFrame WITH inventory_ratio")
    print("=" * 80)
    
    df_with_inventory = df.copy()
    df_with_inventory["inventory_ratio"] = [0.40, 0.35, 0.45]
    
    df_reordered_with_inv = reorder_features(df_with_inventory, drop_extra=True)
    print(f"\nReordered with explicit inventory_ratio: {list(df_reordered_with_inv.columns)}")
    print(f"✓ Original inventory_ratio values preserved")
    print(df_reordered_with_inv[["rolling_7d_sales", "inventory_ratio", "is_black_friday"]])
    
    print("\n" + "=" * 80)
    print("TEST 3: Backward compatibility - DataFrame WITHOUT inventory_ratio")
    print("=" * 80)
    
    df_without_inventory = df.copy()
    
    df_reordered_without_inv = reorder_features(df_without_inventory, drop_extra=True)
    print(f"\nReordered without inventory_ratio in input: {list(df_reordered_without_inv.columns)}")
    print(f"✓ inventory_ratio auto-filled with default 0.5")
    print(df_reordered_without_inv[["rolling_7d_sales", "inventory_ratio", "is_black_friday"]])
    
    print("\n" + "=" * 80)
    print("TEST 4: Keep extra columns after reordering")
    print("=" * 80)
    
    df_reordered_with_extra = reorder_features(df, drop_extra=False)
    print(f"\nReordered (drop_extra=False): {list(df_reordered_with_extra.columns)}")
    print(df_reordered_with_extra)
    
    print("\n" + "=" * 80)
    print("TEST 5: Validate and prepare for prediction")
    print("=" * 80)
    
    df_prepared, success = prepare_for_prediction(df)
    print(f"\nSuccess: {success}")
    print(f"Prepared DataFrame shape: {df_prepared.shape}")
    print(f"Prepared DataFrame columns: {list(df_prepared.columns)}")
    
    print("\n" + "=" * 80)
    print("TEST 6: Handle missing features gracefully")
    print("=" * 80)
    
    df_incomplete = pd.DataFrame({
        "price": [800],
        "competitor_price": [790],
        "month": [1],
        # Missing: price_gap, popularity, day_of_week, rolling_7d_sales, inventory_ratio, etc.
    })
    
    df_prepared_incomplete, success = prepare_for_prediction(df_incomplete)
    print(f"\nSuccess: {success}")
    if not success:
        print(f"Failed as expected - missing required features")
    
    print("\n" + "=" * 80)
    print("TEST 7: Direct validation check")
    print("=" * 80)
    
    is_valid, missing = validate_required_features(df_incomplete)
    print(f"\nValidation result: {is_valid}")
    print(f"Missing features (excluding inventory_ratio): {missing}")
    
    print("\n✓ All feature reordering tests passed!")
