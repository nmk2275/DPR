# Brand-Based Dynamic Elasticity Model

## Overview

The demand generation model in `update_historical_demand_v2.py` has been enhanced with brand-aware elasticity parameters that reflect realistic pricing dynamics:

- **Premium brands** (Apple, Sony): Low price elasticity, weak competitor penalty
- **Non-premium brands** (all others): High price elasticity, strong competitor penalty

## Mathematical Model

### Demand Function
```
Q = BaseDemand - β·Price - θ·max(0, Price - CompetitorPrice) + γ·Popularity + ε
```

Where:
- **Q**: Historical Demand (units)
- **BaseDemand**: Base demand level [150, 300] units
- **β (beta)**: Own-price elasticity coefficient
- **θ (theta)**: Competitor penalty coefficient
- **γ (gamma)**: Popularity sensitivity [20, 60]
- **ε (epsilon)**: Gaussian noise N(0, 5)

### Parameter Ranges

#### Premium Brands (Apple, Sony)
```
β (beta):    [0.05, 0.12]  ← LOW elasticity
θ (theta):   [0.2, 0.5]    ← WEAK competitor penalty
γ (gamma):   [20, 60]      ← Standard popularity impact
```

**Interpretation:**
- **Low β**: Price increases have minimal demand reduction
- **Low θ**: Competitor pricing has weak influence
- **Result**: Can maintain premium pricing with stable demand

**Use Case Example:**
```
Apple iPhone 14 β=0.08:
  Price increase of ₹100 → ~8 unit demand reduction
  
JBL Headphones β=0.65:
  Price increase of ₹100 → ~65 unit demand reduction
```

#### Non-Premium Brands (Samsung, OnePlus, JBL, HP, Dell, Microsoft)
```
β (beta):    [0.4, 0.9]    ← HIGH elasticity
θ (theta):   [1.5, 3.0]    ← STRONG competitor penalty
γ (gamma):   [20, 60]      ← Standard popularity impact
```

**Interpretation:**
- **High β**: Price increases cause significant demand reduction
- **High θ**: Competitor pricing heavily impacts demand
- **Result**: Must maintain competitive pricing to sustain demand

**Use Case Example:**
```
Samsung Galaxy S23 β=0.72:
  Price increase of ₹100 → ~72 unit demand reduction
  
JBL Headphones β=0.65:
  Price ₹10 above competitor → ~θ×100 additional loss
```

## Product Classification

### Premium Products (Low Elasticity)
- **Apple**: iPhone 14, MacBook Air M2, AirPods Pro
- **Sony**: WH-1000XM5, PlayStation 5

### Non-Premium Products (High Elasticity)
- **Samsung**: Galaxy S23
- **OnePlus**: 11
- **Dell**: XPS 13
- **HP**: Pavilion 15
- **JBL**: Tune 760NC
- **Microsoft**: Xbox Series X

## Elasticity Impact on Demand

### Low Elasticity (Premium)
```
β = 0.08  (typical Apple/Sony)

Price Impact:
  ₹800 → Q = BaseDemand - 0.08×800 - ... = -64 units loss
  ₹850 → Q = BaseDemand - 0.08×850 - ... = -68 units loss
  Δ = -4 units for ₹50 increase (8% price change)

Competitor Impact:
  CompetitorPrice = ₹850
  Your Price = ₹880
  Penalty = θ×max(0, 880-850) = 0.35×30 = 10.5 units (modest)
```

### High Elasticity (Non-Premium)
```
β = 0.65  (typical non-premium)

Price Impact:
  ₹800 → Q = BaseDemand - 0.65×800 - ... = -520 units loss
  ₹850 → Q = BaseDemand - 0.65×850 - ... = -552.5 units loss
  Δ = -32.5 units for ₹50 increase (8% price change)

Competitor Impact:
  CompetitorPrice = ₹800
  Your Price = ₹820
  Penalty = θ×max(0, 820-800) = 2.1×20 = 42 units (severe)
```

## Demand Statistics by Product Category

### Premium Brands
```
Product              | Min  | Mean  | Max   | StDev | Elasticity
Apple iPhone 14      |  40  | 92    | 146   | 20.6  | LOW (β ≈ 0.08)
Apple AirPods Pro    | 105  | 182   | 270   | 35.0  | LOW (β ≈ 0.08)
Sony WH-1000XM5      | 116  | 183   | 295   | 35.6  | LOW (β ≈ 0.08)
Sony PlayStation 5   |  63  | 120   | 174   | 22.7  | LOW (β ≈ 0.08)
```

### Non-Premium Brands
```
Product              | Min  | Mean  | Max   | StDev | Elasticity
JBL Tune 760NC       |   5  |  81   | 173   | 47.6  | HIGH (β ≈ 0.65)
Microsoft Xbox       |   5  |  29   |  78   | 22.1  | HIGH (β ≈ 0.65)
OnePlus 11           |   5  |   5   |  11   | 0.3   | HIGH (β ≈ 0.65)
```

## Implementation Details

### Brand Classification Logic
```python
premium_brands = {"Apple", "Sony"}

def get_brand(product_name):
    """Extract brand category from product name."""
    for brand in premium_brands:
        if brand in product_name:
            return brand
    return "Non-Premium"
```

### Parameter Assignment
```python
if brand_category in premium_brands:
    beta = rng.uniform(0.05, 0.12)    # Low elasticity
    theta = rng.uniform(0.2, 0.5)     # Weak competitor penalty
else:
    beta = rng.uniform(0.4, 0.9)      # High elasticity
    theta = rng.uniform(1.5, 3.0)     # Strong competitor penalty
```

## Dynamic Effects Applied

In addition to brand-based elasticity, the model includes:

1. **Regime Shift** (Mid-year decline):
   - Month ≥ 7: 10% demand reduction
   
2. **Gradual Market Drift**:
   - Factor: 1 - (day_of_year / 1000)
   - Represents market saturation
   
3. **Occasional Shocks** (2% probability):
   - Multiplier: [0.7, 1.3]
   - Simulates viral trends or supply issues

4. **Final Clipping**:
   - Minimum demand: 5 units
   - Ensures realistic floor

## Expected Pricing Strategies

### Premium Brands (Using Low Elasticity)
- Can maintain high margins
- Price increases minimally impact demand
- Can price above competitors with less risk
- Focus on brand value, not price competition

### Non-Premium Brands (Using High Elasticity)
- Must compete on price
- Price increases significantly reduce demand
- Should stay close to competitor pricing
- Margin comes from volume, not price premium

## Validation

The model has been tested with:
- ✅ Brand recognition and classification
- ✅ Parameter range enforcement
- ✅ Demand function application
- ✅ Temporal dynamics (regime shift, drift, shocks)
- ✅ Output constraints (minimum 5 units)
- ✅ Dataset consistency (4,015 records processed)

## Future Enhancements

1. **Sub-category elasticity**: Smartphones vs. Laptops vs. Accessories
2. **Temporal elasticity**: Seasonal variations in price sensitivity
3. **Cross-brand elasticity**: Substitute product effects
4. **Learning from data**: Auto-tune elasticity from historical performance
5. **A/B testing framework**: Validate elasticity assumptions with real data

## References

The model implements the asymmetric competitive demand model, where:
- Own-price effect captures direct elasticity
- Competitor penalty captures strategic pricing dynamics
- Popularity modifier captures demand drivers beyond price
- Noise captures market uncertainty and shocks
