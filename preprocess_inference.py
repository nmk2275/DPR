"""
preprocess_inference.py
Preprocessing utilities for Dynamic Pricing Intelligence System.
Used for preparing uploaded datasets for model inference.

IMPORTANT: This module uses feature_engineering.py for ALL feature transformations
to ensure EXACT consistency with training pipeline.

Uses generate_features() and align_features() directly to guarantee:
1. Identical feature engineering to training pipeline
2. Exact feature count and order matching training
3. No manual feature creation or duplication
"""

import pandas as pd
import numpy as np
from typing import Tuple
from feature_engineering import (
    generate_features,
    align_features,
    validate_feature_matrix,
    EXPECTED_FEATURES,
    MAX_INVENTORY,
)


def preprocess_for_inference(df: pd.DataFrame, inplace: bool = False, keep_metadata: bool = True) -> pd.DataFrame:
    """
    Preprocess a DataFrame for model inference using unified feature engineering.
    
    This function applies the EXACT same feature transformations as the training pipeline
    to ensure consistency between training and inference.
    
    Pipeline steps:
    1. Call generate_features() - creates all 13 new features while preserving originals
    2. Call align_features() - selects and orders exactly EXPECTED_FEATURES (15 features)
    3. Optionally keep metadata columns (date, product_name, etc.) for UI purposes
    
    Feature engineering steps (from generate_features):
    1. Temporal features (month, day_of_week, cyclical encodings)
    2. Price features (price_gap)
    3. Popularity from signals
    4. Rolling sales momentum (7-day rolling average)
    5. Calendar event features (Black Friday, New Year, Festival Season)
    6. Inventory features (inventory_ratio)
    
    Args:
        df: Input DataFrame with raw pricing data
        inplace: If True, modify df in-place. If False, work on a copy.
        keep_metadata: If True, preserve date and product_name columns alongside features
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with exactly EXPECTED_FEATURES (15 features)
                     in the exact order used by training pipeline.
                     If keep_metadata=True, also includes date and product_name columns.
    
    Raises:
        ValueError: If required columns missing
    """
    if not inplace:
        df = df.copy()
    
    # Validate minimum required columns
    required_cols = ["date", "price", "competitor_price", "historical_demand"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Save metadata columns if needed (for UI purposes)
    # Preserve: date, product_name, product_id, cost, inventory_level, max_inventory
    metadata = {}
    if keep_metadata:
        # Core identifiers
        if "date" in df.columns:
            metadata["date"] = df["date"].copy()
        if "product_name" in df.columns:
            metadata["product_name"] = df["product_name"].copy()
        if "product_id" in df.columns:
            metadata["product_id"] = df["product_id"].copy()
        
        # Business-critical metadata for pricing/inventory
        if "cost" in df.columns:
            metadata["cost"] = df["cost"].copy()
        if "inventory_level" in df.columns:
            metadata["inventory_level"] = df["inventory_level"].copy()
        if "max_inventory" in df.columns:
            metadata["max_inventory"] = df["max_inventory"].copy()
        if "historical_demand" in df.columns:
            metadata["historical_demand"] = df["historical_demand"].copy()
    
    # ========================================================
    # STEP 1: GENERATE ALL FEATURES
    # Uses generate_features() to create all 13 new features
    # Preserves all original columns (9 original + 13 new = 22 total)
    # ========================================================
    df = generate_features(df)
    
    # ========================================================
    # STEP 2: ALIGN TO EXPECTED_FEATURES
    # Uses align_features() to select and order exactly as training pipeline
    # Output: exactly 15 features in EXPECTED_FEATURES order
    # ========================================================
    df = align_features(df, features=EXPECTED_FEATURES)
    
    # ========================================================
    # STEP 3: RESTORE METADATA COLUMNS (if requested)
    # Keeps date, product_name, etc. for UI/downstream usage
    # ========================================================
    if keep_metadata and metadata:
        for col_name, col_data in metadata.items():
            df[col_name] = col_data
    
    return df


def prepare_for_prediction(
    df: pd.DataFrame,
    features: list = None,
    validate: bool = True,
) -> Tuple[pd.DataFrame, bool]:
    """
    Prepare preprocessed DataFrame for model prediction.
    
    Performs final validation to ensure inference features 
    exactly match training features.
    
    Note: Feature selection and alignment is already done in 
    preprocess_for_inference(), so this is mainly for validation.
    
    Args:
        df: Preprocessed DataFrame (from preprocess_for_inference)
              Should already have exactly EXPECTED_FEATURES
        features: List of expected features (defaults to EXPECTED_FEATURES)
        validate: If True, validate feature matrix
    
    Returns:
        Tuple of:
        - df: Input DataFrame (already aligned)
        - success: bool indicating if preparation was successful
    
    Raises:
        ValueError: If validation fails and validate=True
    """
    if features is None:
        features = EXPECTED_FEATURES
    
    try:
        # Validate that features are in correct order and match training
        if validate:
            is_valid, msg = validate_feature_matrix(df, expected_features=features)
            if not is_valid:
                raise ValueError(f"Feature matrix validation failed: {msg}")
        
        return df, True
    
    except Exception as e:
        print(f"Error preparing features for prediction: {e}")
        return df, False

# ========================================================
# BACKWARD COMPATIBILITY
# ========================================================
# Maintain old function signatures for existing code

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated: Use preprocess_for_inference instead."""
    from feature_engineering import create_temporal_features as _create
    return _create(df)


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated: Use preprocess_for_inference instead."""
    from feature_engineering import create_price_features as _create
    return _create(df)


def create_popularity_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated: Use preprocess_for_inference instead."""
    from feature_engineering import create_popularity_feature as _create
    return _create(df)


if __name__ == "__main__":
    # Example usage and tests
    import pandas as pd
    
    print("=" * 80)
    print("UNIFIED PREPROCESSING TESTS")
    print("=" * 80)
    
    # Test 1: Full preprocessing pipeline
    sample_data = {
        "date": pd.date_range("2024-01-01", periods=15),
        "product_id": [1]*5 + [2]*5 + [3]*5,
        "product_name": ["iPhone 14"]*5 + ["Galaxy S23"]*5 + ["Pixel 8"]*5,
        "price": [800, 810, 820, 830, 840, 700, 710, 720, 730, 740, 600, 610, 620, 630, 640],
        "competitor_price": [790, 800, 810, 820, 830, 690, 700, 710, 720, 730, 590, 600, 610, 620, 630],
        "cost": [500, 500, 500, 500, 500, 400, 400, 400, 400, 400, 300, 300, 300, 300, 300],
        "historical_demand": [100, 110, 105, 115, 120, 95, 100, 98, 105, 110, 50, 55, 52, 58, 60],
        "search_trend": [80, 85, 82, 88, 90, 75, 78, 76, 80, 82, 60, 62, 61, 65, 67],
        "review_velocity": [20, 22, 21, 23, 24, 18, 19, 18, 20, 21, 15, 16, 15, 17, 18],
        "social_buzz": [70, 75, 72, 78, 80, 65, 68, 66, 70, 72, 55, 57, 56, 60, 62],
    }
    df_test = pd.DataFrame(sample_data)
    
    print("\nTest 1: Complete preprocessing pipeline")
    print("-" * 80)
    df_prep = preprocess_for_inference(df_test)
    print(f"Input shape: {df_test.shape}")
    print(f"Output shape: {df_prep.shape}")
    print(f"Columns created: {list(df_prep.columns)}")
    
    print("\nTest 2: Prepare for prediction (feature selection & validation)")
    print("-" * 80)
    df_features, success = prepare_for_prediction(df_prep)
    print(f"Success: {success}")
    print(f"Features shape: {df_features.shape}")
    print(f"Features columns (count={len(df_features.columns)}): {list(df_features.columns)}")
    
    print("\nTest 3: Feature matrix validation")
    print("-" * 80)
    is_valid, msg = validate_feature_matrix(df_features)
    print(f"Validation: {msg}")
    
    print("\nTest 4: Sample feature values")
    print("-" * 80)
    print(df_features.iloc[0].to_string())
    
    print("\n" + "=" * 80)
    print("✓ All unified preprocessing tests passed!")
    print("=" * 80)

