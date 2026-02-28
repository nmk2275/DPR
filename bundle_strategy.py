"""
bundle_strategy.py
Bundle pricing and demand simulation for Dynamic Pricing Intelligence System.

This module provides functions to simulate bundle strategies that combine
multiple products at a discounted price point to increase overall profit.

Functions:
    - simulate_bundle_strategy(): Simulate a two-product bundle pricing strategy
    - calculate_bundle_uplift(): Calculate revenue and margin impact of bundling
"""

from typing import Dict, Tuple


def simulate_bundle_strategy(
    product_a_name: str,
    product_b_name: str,
    price_a: float,
    price_b: float,
    cost_a: float,
    cost_b: float,
    demand_a: float,
    demand_b: float,
    bundle_discount_pct: float = 0.10,
    demand_boost_pct: float = 0.20
) -> Dict:
    """
    Simulate a two-product bundle pricing strategy.
    
    Bundles two products together at a discounted price and estimates demand
    increase due to bundle attractiveness. Calculates profitability of bundling
    compared to individual product sales.
    
    Bundling Logic:
    - Bundle Price = (Price_A + Price_B) * (1 - bundle_discount_pct)
    - Bundle Demand = min(Demand_A, Demand_B) + (demand_boost_pct * min(Demand_A, Demand_B))
      This assumes the bundle attracts additional sales equal to a percentage of the
      minimum individual product demand.
    - Bundle Cost = Cost_A + Cost_B (sum of both product costs)
    - Bundle Profit = (Bundle_Price - Bundle_Cost) * Bundle_Demand
    
    Args:
        product_a_name: Name/identifier of first product (string)
        product_b_name: Name/identifier of second product (string)
        price_a: Current price of product A (float, must be > 0)
        price_b: Current price of product B (float, must be > 0)
        cost_a: Cost of product A (float, must be > 0 and < price_a)
        cost_b: Cost of product B (float, must be > 0 and < price_b)
        demand_a: Estimated demand for product A (float, must be > 0)
        demand_b: Estimated demand for product B (float, must be > 0)
        bundle_discount_pct: Discount as percentage of combined price
                           (default 0.10 = 10% discount)
                           Range: 0.0 to 1.0
        demand_boost_pct: Demand increase as percentage of minimum individual demand
                        (default 0.20 = 20% boost)
                        Range: 0.0 to 1.0
    
    Returns:
        Dictionary with keys:
        - 'bundle_price': float - Price of the bundle
        - 'bundle_demand': float - Estimated demand for bundle
        - 'bundle_profit': float - Total profit from bundle sales
        - 'bundle_margin': float - Profit margin percentage
        - 'individual_profit_a': float - Current profit from product A
        - 'individual_profit_b': float - Current profit from product B
        - 'individual_total_profit': float - Combined profit without bundling
        - 'profit_uplift': float - Additional profit from bundling
        - 'uplift_percentage': float - Percentage increase in profit
        - 'product_a_name': str - Product A identifier
        - 'product_b_name': str - Product B identifier
        - 'is_profitable': bool - Whether bundle is more profitable than individual sales
    
    Raises:
        ValueError: If any input validation fails
    
    Example:
        result = simulate_bundle_strategy(
            product_a_name="Apple iPhone 14",
            product_b_name="Apple AirPods Pro",
            price_a=799,
            price_b=249,
            cost_a=500,
            cost_b=150,
            demand_a=150,
            demand_b=200,
            bundle_discount_pct=0.10,
            demand_boost_pct=0.20
        )
        print(f"Bundle Price: ₹{result['bundle_price']:.2f}")
        print(f"Bundle Demand: {result['bundle_demand']:.0f} units")
        print(f"Bundle Profit: ₹{result['bundle_profit']:.2f}")
        print(f"Profit Uplift: {result['uplift_percentage']:.1f}%")
    """
    
    # ========================================================
    # INPUT VALIDATION
    # ========================================================
    
    # Validate all inputs are positive
    invalid_inputs = []
    
    if price_a <= 0:
        invalid_inputs.append(f"price_a must be > 0, got {price_a}")
    if price_b <= 0:
        invalid_inputs.append(f"price_b must be > 0, got {price_b}")
    if cost_a <= 0:
        invalid_inputs.append(f"cost_a must be > 0, got {cost_a}")
    if cost_b <= 0:
        invalid_inputs.append(f"cost_b must be > 0, got {cost_b}")
    if demand_a <= 0:
        invalid_inputs.append(f"demand_a must be > 0, got {demand_a}")
    if demand_b <= 0:
        invalid_inputs.append(f"demand_b must be > 0, got {demand_b}")
    
    # Validate cost is less than price for each product
    if cost_a >= price_a:
        invalid_inputs.append(f"cost_a ({cost_a}) must be < price_a ({price_a})")
    if cost_b >= price_b:
        invalid_inputs.append(f"cost_b ({cost_b}) must be < price_b ({price_b})")
    
    # Validate discount and boost percentages are reasonable
    if not (0.0 <= bundle_discount_pct <= 1.0):
        invalid_inputs.append(f"bundle_discount_pct must be 0.0-1.0, got {bundle_discount_pct}")
    if not (0.0 <= demand_boost_pct <= 1.0):
        invalid_inputs.append(f"demand_boost_pct must be 0.0-1.0, got {demand_boost_pct}")
    
    if invalid_inputs:
        raise ValueError("Input validation failed:\n" + "\n".join(invalid_inputs))
    
    # ========================================================
    # CALCULATE BUNDLE METRICS
    # ========================================================
    
    # Combined price without discount
    combined_price = price_a + price_b
    
    # Bundle price with discount applied
    bundle_price = combined_price * (1 - bundle_discount_pct)
    
    # Bundle cost is sum of individual costs
    bundle_cost = cost_a + cost_b
    
    # Bundle demand: minimum of individual demands + boost from bundling
    # The boost represents additional sales attracted by the bundle offer
    base_bundle_demand = min(demand_a, demand_b)
    demand_boost = base_bundle_demand * demand_boost_pct
    bundle_demand = base_bundle_demand + demand_boost
    
    # Bundle profit
    bundle_profit = (bundle_price - bundle_cost) * bundle_demand
    
    # Bundle margin (profit as % of price)
    bundle_margin = ((bundle_price - bundle_cost) / bundle_price) * 100 if bundle_price > 0 else 0
    
    # ========================================================
    # CALCULATE INDIVIDUAL SALES PROFIT (FOR COMPARISON)
    # ========================================================
    
    individual_profit_a = (price_a - cost_a) * demand_a
    individual_profit_b = (price_b - cost_b) * demand_b
    individual_total_profit = individual_profit_a + individual_profit_b
    
    # ========================================================
    # CALCULATE UPLIFT METRICS
    # ========================================================
    
    profit_uplift = bundle_profit - individual_total_profit
    
    # Uplift percentage (avoid division by zero)
    if individual_total_profit > 0:
        uplift_percentage = (profit_uplift / individual_total_profit) * 100
    else:
        uplift_percentage = 0.0
    
    is_profitable = bundle_profit > individual_total_profit
    
    # ========================================================
    # RETURN RESULTS
    # ========================================================
    
    return {
        # Bundle specifics
        'bundle_price': bundle_price,
        'bundle_demand': bundle_demand,
        'bundle_profit': bundle_profit,
        'bundle_margin': bundle_margin,
        
        # Individual product baseline
        'individual_profit_a': individual_profit_a,
        'individual_profit_b': individual_profit_b,
        'individual_total_profit': individual_total_profit,
        
        # Uplift analysis
        'profit_uplift': profit_uplift,
        'uplift_percentage': uplift_percentage,
        'is_profitable': is_profitable,
        
        # Product identifiers
        'product_a_name': product_a_name,
        'product_b_name': product_b_name,
    }


def calculate_bundle_uplift(
    bundle_profit: float,
    individual_total_profit: float
) -> Tuple[float, float]:
    """
    Calculate the profit uplift from bundling strategy.
    
    Computes the absolute profit increase and percentage increase
    when using bundle pricing versus individual product sales.
    
    Args:
        bundle_profit: Total profit from bundle sales (float)
        individual_total_profit: Combined profit from individual sales (float)
    
    Returns:
        Tuple of (profit_uplift, uplift_percentage)
        - profit_uplift: Absolute profit increase
        - uplift_percentage: Percentage increase (0 if individual_total_profit is 0)
    
    Example:
        uplift, uplift_pct = calculate_bundle_uplift(10000, 8000)
        print(f"Additional Profit: ₹{uplift:.2f}")  # ₹2000.00
        print(f"Uplift: {uplift_pct:.1f}%")  # 25.0%
    """
    profit_uplift = bundle_profit - individual_total_profit
    
    if individual_total_profit > 0:
        uplift_percentage = (profit_uplift / individual_total_profit) * 100
    else:
        uplift_percentage = 0.0
    
    return profit_uplift, uplift_percentage
