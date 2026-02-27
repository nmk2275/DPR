from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from feature_reorder import reorder_features, get_feature_order

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
    inventory_level = product_row.get("inventory_level", 250)
    max_inventory = product_row.get("max_inventory", 500)
    product_name = product_row.get("product_name", "")
    
    # ========== PRODUCT CATEGORY CLASSIFICATION ==========
    # Define premium vs. mid-tier products based on brand/category
    premium_brands = {"Apple", "Sony"}
    gaming_consoles = {"PlayStation", "Xbox"}
    mid_tier_brands = {"JBL", "OnePlus", "HP", "Dell"}
    
    is_premium = any(brand in product_name for brand in premium_brands)
    is_gaming = any(console in product_name for console in gaming_consoles)
    is_mid_tier = any(brand in product_name for brand in mid_tier_brands)
    
    # Determine price elasticity based on category
    # Premium/Gaming: less price-sensitive (lower elasticity)
    # Mid-tier: highly price-sensitive (higher elasticity)
    if is_premium or is_gaming:
        elasticity_factor = 0.7  # Less sensitive to price changes
    elif is_mid_tier:
        elasticity_factor = 1.3  # More sensitive to price changes
    else:
        elasticity_factor = 1.0  # Standard sensitivity
    
    # ========== HYPERPARAMETERS ==========
    # Hyperparameter for competitive pricing penalty (scaled by elasticity)
    # INCREASED from 0.05 to 0.25: Exceeding competitor price is now 5x more costly
    competitive_penalty_weight = 0.25 * elasticity_factor
    
    # Hyperparameter for soft competitive anchoring
    # Pulls pricing towards competitor price (soft constraint, not hard)
    anchor_penalty_weight = 0.15
    
    # Hyperparameter for uncertainty penalty
    uncertainty_penalty_weight = 0.1
    
    # Hyperparameter for inventory incentive
    inventory_weight = 0.2
    
    # Hyperparameter for trust penalty (price stability)
    trust_penalty_weight = 0.15
    
    # Monte Carlo simulation parameters
    num_simulations = 20
    noise_std = 0.01
    
    # Compute inventory pressure (0 = empty, 1 = full)
    inventory_pressure = inventory_level / max_inventory
    
    # ========== REALISTIC COMPETITIVE PRICE BAND ==========
    if is_premium or is_gaming:
        band = 0.12      # Premium flexibility
    elif is_mid_tier:
        band = 0.05      # Highly competitive
    else:
        band = 0.08      # Standard electronics

    lower_bound = max(cost * 1.05, competitor_price * (1 - band))
    upper_bound = competitor_price * (1 + band)

    if lower_bound >= upper_bound:
        lower_bound = cost * 1.05
        upper_bound = competitor_price * 1.05

    price_range = np.linspace(lower_bound, upper_bound, 40)

    best_profit = -1
    best_price = current_price

    for price in price_range:

        # Monte Carlo demand estimation
        simulated_demands = []
        
        for _ in range(num_simulations):
            # Add Gaussian noise to popularity
            noisy_popularity = popularity + np.random.normal(loc=0, scale=noise_std)
            # Clip to valid range [0, 1]
            noisy_popularity = np.clip(noisy_popularity, 0, 1)
            
            input_data = pd.DataFrame([{
                "price": price,
                "competitor_price": competitor_price,
                "price_gap": price - competitor_price,
                "popularity": noisy_popularity,
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
            
            # Enforce strict feature ordering before prediction
            input_data = reorder_features(input_data, get_feature_order(), drop_extra=True)
            
            predicted_demand = model.predict(input_data)[0]
            simulated_demands.append(predicted_demand)
        
        # Compute statistics from Monte Carlo simulations
        mean_demand = np.mean(simulated_demands)

        # ==========================
        # Realistic Asymmetric Elasticity Correction
        # ==========================
        if competitor_price > 0:
            relative_price_change = (price - competitor_price) / competitor_price

            elasticity_strength = 2.0

            if relative_price_change > 0:
                # Strong penalty when pricing above competitor
                elasticity_adjustment = 1 - (elasticity_strength * relative_price_change)
            else:
                # Mild boost when pricing below competitor
                elasticity_adjustment = 1 - (0.5 * elasticity_strength * relative_price_change)

            elasticity_adjustment = max(0.2, elasticity_adjustment)
            mean_demand *= elasticity_adjustment

        demand_std = np.std(simulated_demands)
        
        # ========== ADAPTIVE COMPETITIVE PENALTY ==========
        # For mid-tier products: severely penalize exceeding competitor price
        price_gap = price - competitor_price
        if is_mid_tier:
            if price_gap > 0:
                # Quadratic penalty increases sharply when exceeding competitor price
                competitive_penalty = competitive_penalty_weight * (price_gap ** 2) * 3.5
            else:
                # No penalty when at or below competitor price
                competitive_penalty = 0.0
        else:
            # Premium/Gaming: standard penalty (less aggressive)
            competitive_penalty = competitive_penalty_weight * max(0, price_gap) ** 2
        
        # Uncertainty penalty: penalize high demand variability
        uncertainty_penalty = uncertainty_penalty_weight * demand_std
        
        # Trust penalty: penalize large deviations from current price
        trust_penalty = trust_penalty_weight * abs(price - current_price)
        
        # Soft competitive anchoring: pull pricing towards competitor price
        anchor_penalty = anchor_penalty_weight * abs(price - competitor_price)
        
        # Inventory bonus: incentivize reducing inventory (lower inventory_pressure = higher bonus)
        inventory_bonus = inventory_weight * (1 - inventory_pressure)
        
        # Final reward formula with all components
        # During festival season, reduce volatility penalty
        volatility_penalty_weight = 0.1 if is_festival_season == 1 else 0.2
        reward = ((price - cost) * mean_demand 
                  - volatility_penalty_weight * demand_volatility
                  - competitive_penalty
                  - uncertainty_penalty
                  - trust_penalty
                  - anchor_penalty
                  + inventory_bonus)

        if reward > best_profit:
            best_profit = reward
            best_price = price

    # ========== ENFORCE BOUNDS ==========
    # Clip to the same competitive band bounds (belt-and-suspenders)
    best_price = np.clip(best_price, lower_bound, upper_bound)
    
    return best_price, best_profit


def price_status(current_price, optimal_price):

    if current_price > optimal_price * 1.05:
        return "Overpriced- Suggest Discount"

    elif current_price < optimal_price * 0.95:
        return "Underpriced - Increase Margin"

    else:
        return "Price is Optimal"