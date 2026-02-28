import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
from pathlib import Path

# Import unified feature definitions
from feature_engineering import EXPECTED_FEATURES, MAX_INVENTORY

# Resolve file paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "processed_pricing_data.csv"
MODEL_PATH = BASE_DIR / "demand_model.pkl"

# -------------------------
# Load Processed Data
# -------------------------

df = pd.read_csv(INPUT_CSV)

# -------------------------
# Feature Selection
# -------------------------
# Features MUST match EXPECTED_FEATURES from feature_engineering.py
# This ensures training and inference use identical features

features = EXPECTED_FEATURES

X = df[features]
# Target renamed to historical_demand
y = df["historical_demand"]

# -------------------------
# Train/Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Model Definition
# -------------------------

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

# -------------------------
# Train
# -------------------------

model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("=" * 80)
print("Model Performance:")
print("=" * 80)
print(f"MAE: {round(mae, 2)}")
print(f"R2 Score: {round(r2, 4)}")

# -------------------------
# Feature Importance
# -------------------------

print("\n" + "=" * 80)
print("Feature Importance (Top 15):")
print("=" * 80)

# Get feature importance from the model
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# Verify inventory_ratio is included
if 'inventory_ratio' in feature_importance['feature'].values:
    inv_importance = feature_importance[feature_importance['feature'] == 'inventory_ratio']['importance'].values[0]
    inv_rank = feature_importance[feature_importance['feature'] == 'inventory_ratio'].index[0] + 1
    print(f"\n✓ inventory_ratio is included in the model:")
    print(f"  - Importance: {inv_importance:.6f}")
    print(f"  - Rank: {inv_rank} out of {len(features)}")
else:
    print("\n✗ Warning: inventory_ratio not found in feature importance!")

# -------------------------
# Save Model
# -------------------------

joblib.dump(model, MODEL_PATH)

print("Model saved successfully.")