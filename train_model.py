import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
from pathlib import Path

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

features = [
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
    "is_festival_season"
]

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

print("Model Performance:")
print("MAE:", round(mae,2))
print("R2 Score:", round(r2,4))

# -------------------------
# Save Model
# -------------------------

joblib.dump(model, MODEL_PATH)

print("Model saved successfully.")