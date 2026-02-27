import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# -------------------------------
# 1. PRODUCT CATALOG
# -------------------------------

catalog = [
    ("Apple", "iPhone 14", "Smartphone", 800),
    ("Samsung", "Galaxy S23", "Smartphone", 750),
    ("OnePlus", "11", "Smartphone", 650),
    ("Dell", "XPS 13", "Laptop", 1200),
    ("HP", "Pavilion 15", "Laptop", 900),
    ("Apple", "MacBook Air M2", "Laptop", 1300),
    ("Sony", "WH-1000XM5", "Headphones", 350),
    ("JBL", "Tune 760NC", "Headphones", 150),
    ("Apple", "AirPods Pro", "Headphones", 250),
    ("Sony", "PlayStation 5", "Gaming Console", 500),
    ("Microsoft", "Xbox Series X", "Gaming Console", 500)
]

products = []

product_id = 1
premium_brands = {"Apple", "Sony", "Microsoft"}

for brand, model, category, base_price in catalog:
    cost = int(base_price * random.uniform(0.6, 0.8))
    
    # Set brand-based elasticity parameters once per product
    if brand in premium_brands:
        # Premium brands: low price sensitivity
        beta = np.random.uniform(0.05, 0.12)      # Own-price elasticity
        theta = np.random.uniform(0.2, 0.5)       # Competitive penalty
    else:
        # Non-premium brands: high price sensitivity
        beta = np.random.uniform(0.4, 0.9)        # Own-price elasticity
        theta = np.random.uniform(1.5, 3.0)       # Competitive penalty
    
    gamma = np.random.uniform(20.0, 60.0)        # Popularity coefficient
    
    products.append({
        "product_id": product_id,
        "product_name": f"{brand} {model}",
        "brand": brand,
        "category": category,
        "base_price": base_price,
        "cost": cost,
        "beta": beta,
        "theta": theta,
        "gamma": gamma
    })
    product_id += 1

# -------------------------------
# 2. TIME SERIES
# -------------------------------

dates = pd.date_range("2024-01-01", periods=365)

rows = []

for product in products:

    lifecycle_factor = 1.0

    for day_index, date in enumerate(dates):

        month = date.month
        day_of_week = date.dayofweek

        # -------------------------------
        # 3. RAW MARKET SIGNALS
        # -------------------------------

        # Search Trend (Google Trends style 0-100)
        search_trend = np.random.normal(60, 10)

        # Launch spike first 30 days
        if day_index < 30:
            search_trend += 30

        # Festival boost
        if month in [11, 12]:
            search_trend += 20

        search_trend = max(0, min(100, search_trend))

        # Review velocity (new reviews per day)
        review_velocity = np.random.normal(20, 5)
        review_velocity = max(0, review_velocity)

        # Social buzz (Reddit mentions)
        social_buzz = np.random.normal(50, 15)
        social_buzz = max(0, social_buzz)

        # -------------------------------
        # 4. PRICING
        # -------------------------------

        price = product["base_price"] + np.random.randint(-50, 50)
        competitor_price = product["base_price"] + np.random.randint(-80, 80)

        # -------------------------------
        # 5. SEASONAL EFFECT
        # -------------------------------

        if month in [11, 12]:
            season_factor = 1.3
        else:
            season_factor = 1.0

        # -------------------------------
        # 6. POPULARITY (HIDDEN, NOT STORED)
        # -------------------------------

        true_popularity = (
            0.4 * (search_trend / 100) +
            0.3 * (review_velocity / 30) +
            0.3 * (social_buzz / 100)
        )

        # -------------------------------
        # 7. DEMAND GENERATION (BRAND-BASED ELASTICITY MODEL)
        # -------------------------------
        # Q = BaseDemand - β·Price - θ·max(0, Price - CompetitorPrice) + γ·Popularity + ε
        
        # Randomized baseline demand per observation (higher baseline for all products)
        base_demand = np.random.randint(200, 350)
        
        # Use per-product elasticity parameters (set once at product initialization)
        beta = product["beta"]
        theta = product["theta"]
        gamma = product["gamma"]
        
        # Competitive penalty: only applies if we price above competitor
        competitive_penalty = max(0.0, price - competitor_price)
        
        # Normalize price relative to base price for elasticity calculation
        # This prevents high-priced products (laptop) from collapsing to floor
        normalized_price = price / product["base_price"]
        
        # Demand formula: Q = BaseDemand - β·(Price/BasePrice) - θ·max(0, Price - CompetitorPrice) + γ·Popularity + ε
        demand = (
            base_demand
            - beta * normalized_price * 100.0  # Scale normalized price by 100 for demand effect
            - theta * competitive_penalty / 10.0  # Scale competitive penalty down
            + gamma * true_popularity
            + np.random.normal(0, 5)
        ) * season_factor * lifecycle_factor
        
        # Enforce realistic lower bound (minimum 10 instead of 5) and integer units
        demand = max(10.0, demand)
        demand = int(round(demand))

        rows.append([
            date,
            product["product_id"],
            product["product_name"],
            product["brand"],
            product["category"],
            price,
            competitor_price,
            product["cost"],
            search_trend,
            review_velocity,
            social_buzz,
            demand
        ])

df = pd.DataFrame(rows, columns=[
    "date",
    "product_id",
    "product_name",
    "brand",
    "category",
    "price",
    "competitor_price",
    "cost",
    "search_trend",
    "review_velocity",
    "social_buzz",
    "quantity"
])

df.to_csv("realistic_electronics_pricing_data.csv", index=False)

print("Realistic electronics dataset created successfully.")