# Category-Aware Dynamic Pricing Strategy

## Overview
The `recommend_price()` function in `price_optimizer.py` has been enhanced with sophisticated category-aware pricing logic that accounts for product elasticity and competitive dynamics.

## Product Categories

### 1. **Premium Products** (Elasticity Factor: 0.7x)
- **Examples**: Apple iPhone 14, Apple MacBook Air, Apple AirPods Pro, Sony WH-1000XM5
- **Characteristics**:
  - Low price elasticity (customers less sensitive to price changes)
  - Strong brand loyalty and value perception
  - Can command premium pricing
  - Recommendations allow ±10% flexibility from current price
  - Competitive penalty: Standard (less aggressive)

### 2. **Gaming Consoles** (Elasticity Factor: 0.7x)
- **Examples**: Microsoft Xbox Series X, Sony PlayStation 5
- **Characteristics**:
  - Low price elasticity (established ecosystems)
  - Premium positioning with loyal customer base
  - Similar flexibility as premium products
  - Recommendations allow ±10% flexibility from current price
  - Competitive penalty: Standard (less aggressive)

### 3. **Mid-Tier Products** (Elasticity Factor: 1.3x)
- **Examples**: JBL Tune 760NC, OnePlus 11, HP Pavilion 15, Dell XPS 13
- **Characteristics**:
  - High price elasticity (very price-sensitive customers)
  - Strong direct competition
  - Demand drops significantly if price exceeds competitor
  - **Hard constraint**: Never exceed competitor price by more than 5%
  - Competitive penalty: Severe (2x multiplier) when exceeding competitor price
  - Recommendations strictly bounded

## Pricing Logic

### Elasticity-Based Competitive Penalty
The competitive penalty is dynamically scaled by elasticity:
```
competitive_penalty_weight = 0.05 * elasticity_factor
```

- **Premium/Gaming** (0.7x): Lighter penalty = 0.035
- **Mid-tier** (1.3x): Heavier penalty = 0.065

### Adaptive Competitive Penalty Calculation

**For Mid-Tier Products:**
```python
if price_gap > 0:
    competitive_penalty = weight * (price_gap)² * 2.0  # Severe penalty
else:
    competitive_penalty = 0.0  # No penalty at/below competitor price
```

**For Premium/Gaming:**
```python
competitive_penalty = weight * max(0, price_gap)²  # Standard quadratic
```

### Price Range Constraints

**Mid-Tier Products:**
```
min_price = current_price * 0.8
max_price = min(current_price * 1.2, competitor_price * 1.05)
```
→ Hard constraint: Never exceed competitor price by more than 5%

**Premium/Gaming Products:**
```
min_price = current_price * 0.8
max_price = current_price * 1.2
```
→ Standard flexibility around current price

## Test Results

### Apple iPhone 14 (Premium)
```
Current:        ₹65,736.00
Competitor:     ₹72,376.00
Recommended:    ₹72,309.60 (-0.09% vs competitor)
Expected Profit: ₹2,829,133.30
```
✓ Premium product can price near competitor despite lower elasticity

### JBL Tune 760NC (Mid-Tier)
```
Current:        ₹9,130.00
Competitor:     ₹18,011.00
Recommended:    ₹10,956.00 (-39.17% vs competitor)
Expected Profit: ₹684,117.13
```
✓ Mid-tier product strongly constrained, stays well below competitor

### Microsoft Xbox Series X (Gaming)
```
Current:        ₹40,255.00
Competitor:     ₹43,990.00
Recommended:    ₹44,280.50 (+0.66% vs competitor)
Expected Profit: ₹4,436,196.48
```
✓ Gaming console allows slight premium positioning

### HP Pavilion 15 (Mid-Tier)
```
Current:        ₹75,779.00
Competitor:     ₹73,621.00
Recommended:    ₹76,019.06 (+3.26% vs competitor)
Expected Profit: ₹3,428,612.79
```
✓ Mid-tier laptop stays within hard constraint (<5% above competitor)

## Reward Function

The final reward incorporates all strategic elements:

```
reward = (price - cost) × mean_demand
         - volatility_penalty × demand_volatility
         - competitive_penalty (category-aware)
         - uncertainty_penalty × demand_std
         - trust_penalty × |price - current_price|
         + inventory_bonus × (1 - inventory_pressure)
```

## Key Implementation Features

✅ **Product Recognition**: Automatic classification by brand/name
✅ **Elasticity Scaling**: Penalties and constraints scale with price sensitivity
✅ **Monte Carlo Robustness**: 20 demand simulations account for uncertainty
✅ **Adaptive Penalties**: Different rules for different product categories
✅ **Hard Constraints**: Mid-tier products never exceed +5% vs competitor
✅ **Festival Season Awareness**: Reduced volatility penalties during peak seasons
✅ **Inventory Intelligence**: Higher discounts when inventory is high
✅ **Price Stability**: Penalties prevent drastic daily changes

## Benefits

1. **Demand Preservation**: Mid-tier products stay competitive
2. **Profit Optimization**: Premium products exploit lower elasticity
3. **Market Responsiveness**: Adapts to competitive pressures
4. **Risk Management**: Uncertainty-aware with Monte Carlo estimation
5. **Inventory Efficiency**: Accelerates clearance when needed
6. **Customer Trust**: Gradual, stable price adjustments
7. **Category Intelligence**: Different strategies for different product types

## Future Enhancements

- Machine learning to auto-classify elasticity from historical data
- A/B testing framework to validate category assignments
- Real-time competitor price monitoring and adjustment
- Demand-forecast driven pricing (not just historical)
- Cross-product bundle pricing optimization
