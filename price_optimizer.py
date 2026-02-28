from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from feature_reorder import reorder_features, get_feature_order
from prediction_validator import validate_prediction_features
from bundle_strategy import simulate_bundle_strategy

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
            
            # Validate features before prediction
            is_valid, validation_msg = validate_prediction_features(
                input_data,
                raise_error=False,
                verbose=False,
                exclude_columns=[]
            )
            if not is_valid:
                raise ValueError(f"Feature validation failed: {validation_msg}")
            
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


def compare_pricing_strategies(
    product_row,
    compare_bundle_with_product_id=None,
    bundle_discount_pct=0.10,
    bundle_demand_boost_pct=0.20
):
    """
    Compare base, discount, and bundle pricing strategies for maximum profit.
    
    Evaluates three pricing strategies and selects the one with highest profit:
    1. Base Strategy: Optimized price from demand model prediction
    2. Discount Strategy: Apply discounts (5%-20%) with elasticity adjustment
    3. Bundle Strategy: Combine with another product at a discount
    
    This function preserves the existing demand prediction pipeline and adds
    strategic comparison on top.
    
    Args:
        product_row: pandas Series with product data
                    Must include: cost, competitor_price, month, popularity,
                                 rolling_7d_sales, price, and features for model
        compare_bundle_with_product_id: Product ID to bundle with (optional)
                                       If provided, enables bundle strategy comparison
        bundle_discount_pct: Discount on bundle price (default 0.10 = 10%)
        bundle_demand_boost_pct: Demand increase from bundling (default 0.20 = 20%)
    
    Returns:
        Dictionary with keys:
        - 'best_strategy_name': str - "Base", "Discount", or "Bundle"
        - 'recommended_price': float - Price for best strategy
        - 'expected_demand': float - Demand at recommended price
        - 'expected_profit': float - Profit from best strategy
        - 'profit_margin': float - Profit margin percentage
        - 'explanation': str - Human-readable explanation of choice
        - 'all_strategies': dict - Details of all evaluated strategies
        
        Example all_strategies structure:
        {
            'base': {'price': 870, 'demand': 450, 'profit': 42000, ...},
            'discount': {'discount_rate': 0.05, 'price': 826, 'demand': 420, ...},
            'bundle': {'bundle_with': 'iPhone', 'price': 1200, 'demand': 300, ...}
        }
    
    Example:
        result = compare_pricing_strategies(
            product_row=df.iloc[0],
            compare_bundle_with_product_id=None  # Bundle disabled
        )
        
        print(f"Strategy: {result['best_strategy_name']}")
        print(f"Price: ₹{result['recommended_price']:.2f}")
        print(f"Demand: {result['expected_demand']:.0f} units")
        print(f"Profit: ₹{result['expected_profit']:.2f}")
        print(f"Reason: {result['explanation']}")
    """
    
    import warnings
    warnings.filterwarnings('ignore')
    
    # ========================================================
    # STRATEGY 1: BASE STRATEGY (Existing Demand Prediction)
    # ========================================================
    
    base_price, base_profit = recommend_price(product_row)
    
    # Estimate base demand using cost and profit
    cost = product_row["cost"]
    base_demand = base_profit / (base_price - cost) if (base_price - cost) > 0 else 0
    base_margin = ((base_price - cost) / base_price * 100) if base_price > 0 else 0
    
    base_strategy = {
        'price': base_price,
        'demand': base_demand,
        'profit': base_profit,
        'margin': base_margin,
        'description': 'ML model optimized pricing'
    }
    
    # ========================================================
    # STRATEGY 2: DISCOUNT STRATEGY (5%-20% Discounts)
    # ========================================================
    
    discount_strategies = []
    discount_rates = [0.05, 0.10, 0.15, 0.20]
    
    # Determine elasticity for this product
    product_name = product_row.get("product_name", "")
    premium_brands = {"Apple", "Sony"}
    gaming_consoles = {"PlayStation", "Xbox"}
    mid_tier_brands = {"JBL", "OnePlus", "HP", "Dell"}
    
    is_premium = any(brand in product_name for brand in premium_brands)
    is_gaming = any(console in product_name for console in gaming_consoles)
    is_mid_tier = any(brand in product_name for brand in mid_tier_brands)
    
    if is_premium or is_gaming:
        elasticity = 0.7
    elif is_mid_tier:
        elasticity = 1.3
    else:
        elasticity = 1.0
    
    current_price = product_row["price"]
    estimated_demand = product_row.get("historical_demand", 100)
    
    for discount_rate in discount_rates:
        discounted_price = current_price * (1 - discount_rate)
        
        # Estimate demand change using elasticity
        price_change_pct = (discounted_price - current_price) / current_price
        demand_multiplier = 1 + (elasticity * price_change_pct)
        new_demand = max(0, estimated_demand * demand_multiplier)
        
        # Calculate profit
        discount_profit = (discounted_price - cost) * new_demand
        discount_margin = ((discounted_price - cost) / discounted_price * 100) if discounted_price > 0 else 0
        
        discount_strategies.append({
            'discount_rate': discount_rate,
            'price': discounted_price,
            'demand': new_demand,
            'profit': discount_profit,
            'margin': discount_margin
        })
    
    # Pick best discount strategy
    best_discount = max(discount_strategies, key=lambda x: x['profit'])
    
    discount_strategy = {
        'discount_rate': best_discount['discount_rate'],
        'price': best_discount['price'],
        'demand': best_discount['demand'],
        'profit': best_discount['profit'],
        'margin': best_discount['margin'],
        'description': f"{int(best_discount['discount_rate']*100)}% discount strategy"
    }
    
    # ========================================================
    # STRATEGY 3: BUNDLE STRATEGY (Optional)
    # ========================================================
    
    bundle_strategy = None
    bundle_enabled = compare_bundle_with_product_id is not None
    
    if bundle_enabled:
        try:
            # For this demo, we'll create a synthetic second product
            # In production, you'd fetch the actual product from your database
            # For now, simulate a complementary product
            
            # Get a complementary product (simulated)
            second_product_name = "Complementary Product"
            second_price = current_price * 0.6  # 60% of main product price
            second_cost = cost * 0.6
            second_demand = estimated_demand * 0.8
            
            bundle_result = simulate_bundle_strategy(
                product_a_name=product_name,
                product_b_name=second_product_name,
                price_a=current_price,
                price_b=second_price,
                cost_a=cost,
                cost_b=second_cost,
                demand_a=estimated_demand,
                demand_b=second_demand,
                bundle_discount_pct=bundle_discount_pct,
                demand_boost_pct=bundle_demand_boost_pct
            )
            
            bundle_strategy = {
                'bundle_with': second_product_name,
                'price': bundle_result['bundle_price'],
                'demand': bundle_result['bundle_demand'],
                'profit': bundle_result['bundle_profit'],
                'margin': bundle_result['bundle_margin'],
                'is_profitable': bundle_result['is_profitable'],
                'uplift_pct': bundle_result['uplift_percentage'],
                'description': 'Bundle with complementary product'
            }
        except Exception as e:
            # If bundle strategy fails, just skip it
            bundle_strategy = None
            bundle_enabled = False
    
    # ========================================================
    # COMPARE ALL STRATEGIES AND SELECT BEST
    # ========================================================
    
    strategies_to_compare = {
        'base': base_strategy,
        'discount': discount_strategy,
    }
    
    if bundle_enabled and bundle_strategy is not None:
        strategies_to_compare['bundle'] = bundle_strategy
    
    # Find strategy with highest profit
    best_strategy_name = max(
        strategies_to_compare.keys(),
        key=lambda x: strategies_to_compare[x]['profit']
    )
    
    best_strategy_data = strategies_to_compare[best_strategy_name]
    
    # ========================================================
    # BUILD EXPLANATION
    # ========================================================
    
    explanation = _build_strategy_explanation(
        best_strategy_name,
        best_strategy_data,
        strategies_to_compare,
        base_strategy
    )
    
    # ========================================================
    # RETURN RESULTS
    # ========================================================
    
    return {
        'best_strategy_name': best_strategy_name.upper(),
        'recommended_price': best_strategy_data['price'],
        'expected_demand': best_strategy_data['demand'],
        'expected_profit': best_strategy_data['profit'],
        'profit_margin': best_strategy_data['margin'],
        'explanation': explanation,
        'all_strategies': strategies_to_compare
    }


def _build_strategy_explanation(best_name, best_data, all_strategies, base_strategy):
    """
    Build human-readable explanation for strategy selection.
    
    Args:
        best_name: Name of best strategy ('base', 'discount', 'bundle')
        best_data: Data dict for best strategy
        all_strategies: Dict of all strategies
        base_strategy: Base strategy reference
    
    Returns:
        str: Explanation of why this strategy was selected
    """
    
    profit_vs_base = best_data['profit'] - base_strategy['profit']
    profit_pct = (profit_vs_base / base_strategy['profit'] * 100) if base_strategy['profit'] > 0 else 0
    
    if best_name == 'base':
        return (
            f"Base strategy selected. ML-optimized price of ₹{best_data['price']:.2f} "
            f"maximizes profit at ₹{best_data['profit']:.2f}"
        )
    
    elif best_name == 'discount':
        discount_rate = best_data['discount_rate']
        return (
            f"Discount strategy recommended: {int(discount_rate*100)}% off. "
            f"Volume increase justifies lower margin. Profit: ₹{best_data['profit']:.2f} "
            f"({profit_pct:+.1f}% vs base strategy)"
        )
    
    elif best_name == 'bundle':
        bundle_with = best_data.get('bundle_with', 'complementary product')
        margin = best_data['margin']
        return (
            f"Bundle with {bundle_with} recommended due to higher combined margin ({margin:.1f}%). "
            f"Bundled price ₹{best_data['price']:.2f} captures cross-sell demand. "
            f"Profit: ₹{best_data['profit']:.2f} ({profit_pct:+.1f}% vs base strategy)"
        )
    
    else:
        return f"Strategy: {best_name} selected with profit ₹{best_data['profit']:.2f}"