"""
validate_dataset.py
Dataset validation utility for Dynamic Pricing Intelligence System.
Checks CSV files for required columns before processing.
"""

import streamlit as st
import pandas as pd
from typing import List, Tuple


def validate_dataset(df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names. 
                         If None, uses default columns for pricing system.
    
    Returns:
        Tuple[bool, List[str]]: 
            - bool: True if all required columns present, False otherwise
            - List[str]: List of missing column names (empty if valid)
    """
    if required_columns is None:
        required_columns = [
            "date",
            "product_name",
            "price",
            "competitor_price",
            "cost",
            "historical_demand"
        ]
    
    # Check if DataFrame is empty
    if df.empty:
        return False, ["Dataset is empty"]
    
    # Find missing columns
    df_columns = set(df.columns)
    required_set = set(required_columns)
    missing_columns = list(required_set - df_columns)
    
    if missing_columns:
        return False, missing_columns
    
    return True, []


def validate_dataset_streamlit(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate dataset and show Streamlit error if invalid.
    Stops execution if validation fails.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        bool: True if valid (execution continues), False if invalid (stops)
    """
    is_valid, missing_cols = validate_dataset(df, required_columns)
    
    if not is_valid:
        error_msg = "❌ Invalid dataset: Missing required columns:\n"
        for col in missing_cols:
            error_msg += f"  - {col}\n"
        st.error(error_msg)
        st.info("Please upload a CSV with all required columns: date, product_name, price, competitor_price, cost, historical_demand")
        return False
    
    return True


if __name__ == "__main__":
    # Example usage (for testing)
    import pandas as pd
    
    # Valid dataset example
    valid_data = {
        "date": ["2024-01-01", "2024-01-02"],
        "product_name": ["iPhone 14", "iPhone 14"],
        "price": [800, 810],
        "competitor_price": [790, 800],
        "cost": [500, 500],
        "historical_demand": [100, 105]
    }
    df_valid = pd.DataFrame(valid_data)
    is_valid, missing = validate_dataset(df_valid)
    print(f"Valid dataset: {is_valid}, Missing: {missing}")
    
    # Invalid dataset example (missing columns)
    invalid_data = {
        "date": ["2024-01-01"],
        "product_name": ["iPhone 14"],
        "price": [800]
    }
    df_invalid = pd.DataFrame(invalid_data)
    is_valid, missing = validate_dataset(df_invalid)
    print(f"Invalid dataset: {is_valid}, Missing: {missing}")
