from pathlib import Path
import pandas as pd
import numpy as np
from feature_engineering import engineer_features, MAX_INVENTORY

# Resolve file paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "realistic_electronics_pricing_data.csv"
OUTPUT_CSV = BASE_DIR / "processed_pricing_data.csv"

# Load dataset
df = pd.read_csv(INPUT_CSV)

# If raw file uses 'quantity', rename to 'historical_demand' for clarity
if 'quantity' in df.columns and 'historical_demand' not in df.columns:
    df = df.rename(columns={'quantity': 'historical_demand'})

# ============================================================
# UNIFIED FEATURE ENGINEERING (shared with inference pipeline)
# ============================================================
# This applies the EXACT same transformations as preprocess_inference.py
df = engineer_features(df, group_by='product_id', create_inventory=True, max_inventory=MAX_INVENTORY)

# Add max_inventory as a constant column (for reference)
df['max_inventory'] = MAX_INVENTORY

# Sort by product_id and date for reproducibility
df = df.sort_values(['product_id', 'date']).reset_index(drop=True)

# -----------------------------
# Final Clean Dataset
# -----------------------------

print(df.head())
df.to_csv(OUTPUT_CSV, index=False)

print("Preprocessing complete.")