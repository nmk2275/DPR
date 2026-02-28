"""
prediction_validator.py
Validation utilities for model predictions.

Ensures that feature DataFrames match training specifications before calling model.predict().
This is the final gatekeeper before predictions to catch any feature inconsistencies.
"""

import pandas as pd
from typing import Tuple
from feature_engineering import EXPECTED_FEATURES


def validate_prediction_features(
    features_df: pd.DataFrame,
    raise_error: bool = True,
    verbose: bool = True,
    exclude_columns: list = None,
) -> Tuple[bool, str]:
    """
    Validate that feature DataFrame matches EXPECTED_FEATURES exactly before prediction.
    
    This function should be called immediately before model.predict() to catch
    any feature mismatches that could lead to incorrect predictions.
    
    Checks:
    1. DataFrame is not empty
    2. Column names match EXPECTED_FEATURES exactly
    3. Column order matches EXPECTED_FEATURES exactly
    4. No unexpected extra columns (unless in exclude_columns)
    5. Feature shape is as expected
    
    Args:
        features_df: DataFrame to validate (should have already been aligned)
        raise_error: If True, raise ValueError on validation failure
        verbose: If True, print debug information
        exclude_columns: List of column names to exclude from validation
                        (e.g., ['date', 'product_name'] for metadata columns)
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    
    Raises:
        ValueError: If validation fails and raise_error=True
    """
    
    if exclude_columns is None:
        exclude_columns = []
    
    # ========================================================
    # CHECK 1: DataFrame is not empty
    # ========================================================
    if features_df is None or features_df.empty:
        msg = "ERROR: Features DataFrame is empty or None"
        if raise_error:
            raise ValueError(msg)
        return False, msg
    
    # Create a subset with only the expected features (exclude metadata)
    features_only = features_df.drop(columns=exclude_columns, errors='ignore')
    
    # ========================================================
    # CHECK 2: Shape validation
    # ========================================================
    n_rows, n_cols = features_only.shape
    expected_cols = len(EXPECTED_FEATURES)
    
    if verbose:
        print(f"[PREDICTION VALIDATOR] Feature shape: ({n_rows}, {n_cols})")
    
    if n_cols != expected_cols:
        msg = (f"ERROR: Feature count mismatch. "
               f"Expected {expected_cols} features, got {n_cols}")
        if raise_error:
            raise ValueError(msg)
        return False, msg
    
    # ========================================================
    # CHECK 3: Column names match exactly
    # ========================================================
    actual_cols = list(features_only.columns)
    if actual_cols != EXPECTED_FEATURES:
        missing = set(EXPECTED_FEATURES) - set(actual_cols)
        extra = set(actual_cols) - set(EXPECTED_FEATURES)
        
        msg_parts = ["ERROR: Feature columns mismatch"]
        if missing:
            msg_parts.append(f"Missing: {missing}")
        if extra:
            msg_parts.append(f"Extra: {extra}")
        
        msg = " | ".join(msg_parts)
        if raise_error:
            raise ValueError(msg)
        return False, msg
    
    # ========================================================
    # CHECK 4: Column order matches exactly
    # ========================================================
    if actual_cols != EXPECTED_FEATURES:
        msg = ("ERROR: Feature column order mismatch. "
               f"Expected: {EXPECTED_FEATURES}\n"
               f"Got: {actual_cols}")
        if raise_error:
            raise ValueError(msg)
        return False, msg
    
    # ========================================================
    # CHECK 5: No NaN values in critical features
    # ========================================================
    nan_cols = features_only.columns[features_only.isna().any()].tolist()
    if nan_cols:
        msg = f"WARNING: NaN values found in features: {nan_cols}"
        if raise_error:
            raise ValueError(f"ERROR: {msg}")
        if verbose:
            print(f"[PREDICTION VALIDATOR] {msg}")
    
    # ========================================================
    # All checks passed
    # ========================================================
    if verbose:
        print(f"[PREDICTION VALIDATOR] ✓ All validation checks passed")
        print(f"[PREDICTION VALIDATOR] ✓ {n_rows} rows × {n_cols} features")
        print(f"[PREDICTION VALIDATOR] ✓ Features: {actual_cols}")
    
    return True, "✓ Feature validation passed - ready for prediction"



def validate_before_predict(features_df: pd.DataFrame) -> None:
    """
    Simple wrapper for validation that raises on failure (strict mode).
    
    Use this before model.predict() when you want strict validation.
    
    Args:
        features_df: DataFrame to validate
    
    Raises:
        ValueError: If any validation check fails
    
    Example:
        from prediction_validator import validate_before_predict
        
        df = preprocess_for_inference(raw_data)
        df, _ = prepare_for_prediction(df)
        
        # Validate before prediction
        validate_before_predict(df)
        
        # Safe to predict now
        predictions = model.predict(df)
    """
    is_valid, message = validate_prediction_features(
        features_df,
        raise_error=True,
        verbose=True
    )
    if not is_valid:
        raise ValueError(message)


# ========================================================
# USAGE EXAMPLES
# ========================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 90)
    print("PREDICTION VALIDATOR - EXAMPLES AND TESTS")
    print("=" * 90)
    
    # Example 1: Valid features
    print("\n" + "=" * 90)
    print("Example 1: Valid feature DataFrame")
    print("=" * 90)
    valid_df = pd.DataFrame({
        feature: [0.5] * 5
        for feature in EXPECTED_FEATURES
    })
    is_valid, msg = validate_prediction_features(valid_df)
    print(f"Result: {msg}")
    
    # Example 2: Missing column
    print("\n" + "=" * 90)
    print("Example 2: Missing column (should fail)")
    print("=" * 90)
    invalid_df = valid_df.drop("price", axis=1)
    is_valid, msg = validate_prediction_features(
        invalid_df,
        raise_error=False,
        verbose=True
    )
    print(f"Result: {'✓ PASS' if is_valid else '✗ FAIL'} - {msg}")
    
    # Example 3: Wrong column order
    print("\n" + "=" * 90)
    print("Example 3: Wrong column order (should fail)")
    print("=" * 90)
    wrong_order_df = valid_df[reversed(EXPECTED_FEATURES)]
    is_valid, msg = validate_prediction_features(
        wrong_order_df,
        raise_error=False,
        verbose=True
    )
    print(f"Result: {'✓ PASS' if is_valid else '✗ FAIL'} - {msg}")
    
    # Example 4: Using strict validation
    print("\n" + "=" * 90)
    print("Example 4: Using validate_before_predict() (strict mode)")
    print("=" * 90)
    try:
        validate_before_predict(valid_df)
        print("✓ Validation passed - ready for model.predict()")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    print("\n" + "=" * 90)
    print("✓ VALIDATOR TESTS COMPLETE")
    print("=" * 90)
