print("Script started")

from pathlib import Path
import pandas as pd
try:
    from price_optimizer import recommend_price, price_status, compare_pricing_strategies
    from elasticity_utils import calculate_demand_from_price_change
    from demand_curve import (
        calculate_demand_at_price,
        calculate_demand_at_price_with_popularity,
        adjust_elasticity_by_popularity
    )
except Exception as e:
    print("Import error:", e)
    raise

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "processed_pricing_data.csv"

df = pd.read_csv(INPUT_CSV)

# Take latest record of one product
sample = df[df["product_id"] == 5].iloc[-1]

optimal_price, expected_profit = recommend_price(sample)

status = price_status(sample["price"], optimal_price)

print("Product:", sample["product_name"])
print("Current Price:", sample["price"])
print("Recommended Price:", round(optimal_price, 2))
print("Expected Profit:", round(expected_profit, 2))
print("Status:", status)

# Estimate base demand for discount analysis
# Using historical_demand as proxy for current demand
estimated_demand = sample.get("historical_demand", 100)


# ========================================================
# DISCOUNT STRATEGY SIMULATION
# ========================================================

def simulate_discount_strategy(price, cost, predicted_demand, elasticity=1.0):
    """
    Simulate different discount strategies and recommend the best one.
    
    Tests discount rates (5%, 10%, 15%, 20%) and estimates demand impact
    based on price elasticity. Calculates profit for each scenario and
    returns the best discount strategy.
    
    Args:
        price: Current price (float)
        cost: Product cost (float)
        predicted_demand: Base demand at current price (float)
        elasticity: Price elasticity of demand (default 1.0)
                   Values > 1 = price sensitive
                   Values < 1 = price inelastic
                   Values = 1 = unit elastic
    
    Returns:
        dict with keys:
        - 'best_discount_rate': float (0.0 to 0.2, e.g., 0.05 for 5%)
        - 'new_price': float (discounted price)
        - 'new_demand': float (estimated demand after discount)
        - 'profit': float (total profit at new price)
        - 'all_scenarios': list of dicts with all tested scenarios
    
    Example:
        result = simulate_discount_strategy(
            price=100, 
            cost=60, 
            predicted_demand=500, 
            elasticity=1.5
        )
        print(f"Best: {result['best_discount_rate']*100:.0f}% discount")
        print(f"New price: ₹{result['new_price']:.2f}")
        print(f"Max profit: ₹{result['profit']:.2f}")
    """
    import numpy as np
    
    # Validate inputs
    if price <= 0 or cost <= 0 or predicted_demand <= 0 or elasticity <= 0:
        raise ValueError("price, cost, predicted_demand, and elasticity must be positive")
    
    if cost >= price:
        raise ValueError("cost must be less than price")
    
    # Discount rates to test (5%, 10%, 15%, 20%)
    discount_rates = [0.05, 0.10, 0.15, 0.20]
    
    scenarios = []
    
    for discount_rate in discount_rates:
        # Calculate new price after discount
        new_price = price * (1 - discount_rate)
        
        # Estimate demand change using proper elasticity formula
        # Formula: NewDemand = OldDemand × (1 - Elasticity × PriceChangePercent)
        # This ensures: price decrease (negative pct) → demand increase (correct)
        price_change_pct = (new_price - price) / price  # negative for discounts
        
        new_demand = calculate_demand_from_price_change(
            old_demand=predicted_demand,
            elasticity=elasticity,
            price_change_pct=price_change_pct,
            min_demand=1.0,
            max_demand_multiplier=5.0
        )
        
        # Calculate profit
        profit = (new_price - cost) * new_demand
        
        scenario = {
            'discount_rate': discount_rate,
            'new_price': new_price,
            'new_demand': new_demand,
            'profit': profit,
            'discount_pct_label': f"{int(discount_rate * 100)}%"
        }
        scenarios.append(scenario)
    
    # Find best scenario (highest profit)
    best_scenario = max(scenarios, key=lambda x: x['profit'])
    
    return {
        'best_discount_rate': best_scenario['discount_rate'],
        'new_price': best_scenario['new_price'],
        'new_demand': best_scenario['new_demand'],
        'profit': best_scenario['profit'],
        'all_scenarios': scenarios
    }


# ========================================================
# TEST DISCOUNT STRATEGY WITH SAMPLE DATA
# ========================================================

print("\n" + "="*60)
print("DISCOUNT STRATEGY ANALYSIS")
print("="*60)

# Determine elasticity based on product category
product_name = sample["product_name"]
premium_brands = {"Apple", "Sony"}
gaming_consoles = {"PlayStation", "Xbox"}
mid_tier_brands = {"JBL", "OnePlus", "HP", "Dell"}

is_premium = any(brand in product_name for brand in premium_brands)
is_gaming = any(console in product_name for console in gaming_consoles)
is_mid_tier = any(brand in product_name for brand in mid_tier_brands)

if is_premium or is_gaming:
    elasticity = 0.7  # Less price-sensitive
elif is_mid_tier:
    elasticity = 1.3  # More price-sensitive
else:
    elasticity = 1.0  # Standard sensitivity

# Test discount strategy
discount_result = simulate_discount_strategy(
    price=sample["price"],
    cost=sample["cost"],
    predicted_demand=estimated_demand,
    elasticity=elasticity
)

print(f"\nProduct: {product_name}")
print(f"Current Price: ₹{sample['price']:.2f}")
print(f"Base Demand (at current price): {estimated_demand:.0f} units")
print(f"Product Category Elasticity: {elasticity:.1f}")
print(f"\nDiscount Strategy Scenarios:")
print("-" * 60)

for scenario in discount_result['all_scenarios']:
    print(f"\n  {scenario['discount_pct_label']} Discount:")
    print(f"    New Price: ₹{scenario['new_price']:.2f}")
    print(f"    Estimated Demand: {scenario['new_demand']:.0f} units")
    print(f"    Total Profit: ₹{scenario['profit']:.2f}")

print("\n" + "="*60)
print(f"RECOMMENDED DISCOUNT: {int(discount_result['best_discount_rate']*100)}%")
print(f"New Price: ₹{discount_result['new_price']:.2f}")
print(f"Estimated Demand: {discount_result['new_demand']:.0f} units")
print(f"Maximum Profit: ₹{discount_result['profit']:.2f}")
print("="*60)


# ========================================================
# BUNDLE STRATEGY SIMULATION
# ========================================================

from bundle_strategy import simulate_bundle_strategy

print("\n" + "="*60)
print("BUNDLE STRATEGY ANALYSIS")
print("="*60)

# Get a second product for bundling analysis
# Find a different product to bundle with sample
sample2_mask = (df["product_id"] == 10)  # Different product
if sample2_mask.any():
    sample2 = df[sample2_mask].iloc[-1]
    
    # Simulate bundle strategy
    bundle_result = simulate_bundle_strategy(
        product_a_name=sample["product_name"],
        product_b_name=sample2["product_name"],
        price_a=sample["price"],
        price_b=sample2["price"],
        cost_a=sample["cost"],
        cost_b=sample2["cost"],
        demand_a=sample.get("historical_demand", 100),
        demand_b=sample2.get("historical_demand", 100),
        bundle_discount_pct=0.10,  # 10% discount on combined price
        demand_boost_pct=0.20      # 20% demand boost from bundling
    )
    
    print(f"\nBundle Composition:")
    print(f"  Product A: {bundle_result['product_a_name']}")
    print(f"  Product B: {bundle_result['product_b_name']}")
    
    print(f"\nIndividual Product Performance:")
    print(f"  {bundle_result['product_a_name']}: ₹{bundle_result['individual_profit_a']:.2f} profit")
    print(f"  {bundle_result['product_b_name']}: ₹{bundle_result['individual_profit_b']:.2f} profit")
    print(f"  Combined Individual Profit: ₹{bundle_result['individual_total_profit']:.2f}")
    
    print(f"\nBundle Strategy Performance:")
    combined_price = sample["price"] + sample2["price"]
    print(f"  Combined Price (no discount): ₹{combined_price:.2f}")
    print(f"  Bundle Price (10% discount): ₹{bundle_result['bundle_price']:.2f}")
    print(f"  Bundle Demand: {bundle_result['bundle_demand']:.0f} units")
    print(f"  Bundle Profit Margin: {bundle_result['bundle_margin']:.1f}%")
    print(f"  Bundle Total Profit: ₹{bundle_result['bundle_profit']:.2f}")
    
    print(f"\nProfitability Analysis:")
    print(f"  Profit Uplift: ₹{bundle_result['profit_uplift']:.2f}")
    print(f"  Uplift Percentage: {bundle_result['uplift_percentage']:.1f}%")
    
    if bundle_result['is_profitable']:
        print(f"  ✅ BUNDLE IS MORE PROFITABLE than individual sales")
    else:
        print(f"  ⚠️  BUNDLE LESS PROFITABLE - Recommend individual sales instead")
    
    print("="*60)
else:
    print("⚠️  Not enough products in dataset for bundle simulation")
    print("="*60)


# ========================================================
# STRATEGY COMPARISON (ALL THREE APPROACHES)
# ========================================================

print("\n" + "="*70)
print("COMPREHENSIVE STRATEGY COMPARISON")
print("="*70)

# Compare all three strategies and get the best one
strategy_result = compare_pricing_strategies(
    product_row=sample,
    compare_bundle_with_product_id=None,  # Bundle comparison enabled
    bundle_discount_pct=0.10,
    bundle_demand_boost_pct=0.20
)

print(f"\n📊 STRATEGY ANALYSIS FOR: {sample['product_name']}")
print(f"Current Price: ₹{sample['price']:.2f}")
print(f"Current Demand (historical): {sample.get('historical_demand', 'N/A'):.0f} units")

print(f"\n" + "─"*70)
print(f"STRATEGY COMPARISON")
print(f"─"*70)

# Display all strategies
for strategy_name, strategy_data in strategy_result['all_strategies'].items():
    print(f"\n{strategy_name.upper()} Strategy:")
    if strategy_name == 'discount':
        discount_pct = strategy_data['discount_rate']
        print(f"  Discount: {int(discount_pct*100)}%")
    
    print(f"  Price: ₹{strategy_data['price']:.2f}")
    print(f"  Expected Demand: {strategy_data['demand']:.0f} units")
    print(f"  Profit: ₹{strategy_data['profit']:.2f}")
    print(f"  Margin: {strategy_data['margin']:.1f}%")

print(f"\n" + "="*70)
print(f"🎯 RECOMMENDED STRATEGY: {strategy_result['best_strategy_name']}")
print(f"="*70)

print(f"\nPrice: ₹{strategy_result['recommended_price']:.2f}")
print(f"Expected Demand: {strategy_result['expected_demand']:.0f} units")
print(f"Expected Profit: ₹{strategy_result['expected_profit']:.2f}")
print(f"Profit Margin: {strategy_result['profit_margin']:.1f}%")

print(f"\n💡 Reason: {strategy_result['explanation']}")
print(f"\n" + "="*70)