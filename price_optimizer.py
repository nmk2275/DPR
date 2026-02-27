from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Resolve path to model relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "demand_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

def recommend_price(product_row):

    cost = product_row["cost"]
    competitor_price = product_row["competitor_price"]
    month = product_row["month"]
    popularity = product_row["popularity"]
    rolling_sales = product_row["rolling_7d_sales"]
    current_price = product_row["price"]
    demand_volatility = product_row.get("demand_volatility", 0.0)
    is_festival_season = product_row.get("is_festival_season", 0)

    # Candidate price range (±20%)
    price_range = np.linspace(current_price * 0.8,
                              current_price * 1.2,
                              40)

    best_profit = -1
    best_price = current_price

    for price in price_range:

        input_data = pd.DataFrame([{
            "price": price,
            "competitor_price": competitor_price,
            "price_gap": price - competitor_price,
            "popularity": popularity,
            "month": month,
            "day_of_week": product_row["day_of_week"],
            "month_sin": product_row.get("month_sin", 0.0),
            "month_cos": product_row.get("month_cos", 0.0),
            "dow_sin": product_row.get("dow_sin", 0.0),
            "dow_cos": product_row.get("dow_cos", 0.0),
            "rolling_7d_sales": rolling_sales,
            "is_black_friday": product_row.get("is_black_friday", 0),
            "is_new_year": product_row.get("is_new_year", 0),
            "is_festival_season": is_festival_season
        }])

        forecasted_demand = model.predict(input_data)[0]
        
        # Reward formula with volatility penalty
        # During festival season, reduce volatility penalty
        volatility_penalty_weight = 0.1 if is_festival_season == 1 else 0.2
        reward = (price - cost) * forecasted_demand - volatility_penalty_weight * demand_volatility

        if reward > best_profit:
            best_profit = reward
            best_price = price

    best_price = np.clip(best_price, current_price * 0.9, current_price * 1.1)
    return best_price, best_profit


def price_status(current_price, optimal_price):

    if current_price > optimal_price * 1.05:
        return "Overpriced - Suggest Discount"

    elif current_price < optimal_price * 0.95:
        return "Underpriced - Increase Margin"

    else:
        return "Price is Optimal"