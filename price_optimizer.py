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
            "rolling_7d_sales": rolling_sales
        }])

        predicted_demand = model.predict(input_data)[0]
        profit = (price - cost) * predicted_demand

        if profit > best_profit:
            best_profit = profit
            best_price = price

    return best_price, best_profit


def price_status(current_price, optimal_price):

    if current_price > optimal_price * 1.05:
        return "Overpriced - Suggest Discount"

    elif current_price < optimal_price * 0.95:
        return "Underpriced - Increase Margin"

    else:
        return "Price is Optimal"