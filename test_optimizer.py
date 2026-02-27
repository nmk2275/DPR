print("Script started")

from pathlib import Path
import pandas as pd
try:
    from price_optimizer import recommend_price, price_status
except Exception as e:
    print("Import error:", e)
    raise

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "processed_pricing_data.csv"

df = pd.read_csv(INPUT_CSV)

# Take latest record of one product
sample = df[df["product_id"] == 1].iloc[-1]

optimal_price, expected_profit = recommend_price(sample)

status = price_status(sample["price"], optimal_price)

print("Product:", sample["product_name"])
print("Current Price:", sample["price"])
print("Recommended Price:", round(optimal_price, 2))
print("Expected Profit:", round(expected_profit, 2))
print("Status:", status)