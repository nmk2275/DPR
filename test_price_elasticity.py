"""
test_price_elasticity.py
Price Elasticity Analysis for Dynamic Pricing Intelligence System

This script analyzes how demand responds to price changes for a selected product.
It loads a trained demand model, selects the latest product record, varies the price,
and visualizes the price-demand relationship.

Usage:
    python test_price_elasticity.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(model_path: str = "demand_model.pkl", 
              data_path: str = "processed_pricing_data.csv") -> tuple:
    """
    Load the trained demand model and pricing data.
    
    Args:
        model_path: Path to the serialized XGBoost model
        data_path: Path to the processed pricing dataset
    
    Returns:
        tuple: (model, DataFrame) containing the loaded model and data
    
    Raises:
        FileNotFoundError: If either file cannot be found
    """
    try:
        model = joblib.load(model_path)
        print(f"✓ Loaded demand model from '{model_path}'")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded pricing data from '{data_path}' ({len(df)} records)")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    return model, df


def get_product_sample(df: pd.DataFrame) -> pd.Series:
    """
    Select the latest record from the dataset as the base product.
    
    Args:
        df: Input DataFrame with pricing data
    
    Returns:
        pd.Series: The latest product record
    """
    sample = df.iloc[-1].copy()
    product_name = sample.get("product_name", "Unknown Product")
    print(f"✓ Selected product: {product_name} (latest record)")
    return sample


def generate_price_range(base_price: float, 
                         min_multiplier: float = 0.8,
                         max_multiplier: float = 1.2,
                         num_points: int = 50) -> np.ndarray:
    """
    Generate a range of prices around the base price.
    
    Args:
        base_price: Current product price
        min_multiplier: Minimum price as fraction of base (default 80%)
        max_multiplier: Maximum price as fraction of base (default 120%)
        num_points: Number of price points to evaluate
    
    Returns:
        np.ndarray: Array of prices to test
    """
    prices = np.linspace(base_price * min_multiplier,
                        base_price * max_multiplier,
                        num_points)
    return prices


def predict_demand_curve(model, base_row: pd.Series, 
                        prices: np.ndarray) -> np.ndarray:
    """
    Predict demand for a range of prices while keeping other features constant.
    
    Args:
        model: Trained XGBoost model with feature_names_in_ attribute
        base_row: Base product record (will be modified with different prices)
        prices: Array of prices to evaluate
    
    Returns:
        np.ndarray: Array of predicted demand values
    """
    demands = []
    
    for price in prices:
        # Create a copy and update price and price_gap
        row = base_row.copy()
        row["price"] = price
        row["price_gap"] = price - row.get("competitor_price", 0)
        
        # Extract features in the correct order expected by the model
        features = row[model.feature_names_in_]
        X = pd.DataFrame([features])
        
        # Predict demand
        demand = model.predict(X)[0]
        demands.append(demand)
    
    return np.array(demands)


def calculate_elasticity(prices: np.ndarray, 
                        demands: np.ndarray) -> tuple:
    """
    Calculate price elasticity metrics.
    
    Args:
        prices: Array of price points
        demands: Array of corresponding demand predictions
    
    Returns:
        tuple: (slope, elasticity_type, r_squared) where elasticity_type is a string
    """
    # Fit a linear regression to calculate slope
    coefficients = np.polyfit(prices, demands, 1)
    slope = coefficients[0]
    
    # Determine elasticity type
    threshold = 0.01  # Small threshold to account for numerical precision
    if slope > threshold:
        elasticity_type = "POSITIVE (inelastic/unusual)"
    elif slope < -threshold:
        elasticity_type = "NEGATIVE (elastic)"
    else:
        elasticity_type = "FLAT (unit elastic)"
    
    # Calculate R² for goodness of fit
    y_pred = np.polyval(coefficients, prices)
    ss_res = np.sum((demands - y_pred) ** 2)
    ss_tot = np.sum((demands - np.mean(demands)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, elasticity_type, r_squared


def plot_elasticity(prices: np.ndarray, 
                   demands: np.ndarray,
                   product_name: str,
                   base_price: float,
                   slope: float,
                   elasticity_type: str,
                   output_path: str = "price_elasticity.png") -> None:
    """
    Create a professional visualization of the price-demand relationship.
    
    Args:
        prices: Array of price points
        demands: Array of predicted demand values
        product_name: Name of the product
        base_price: Original/base price of the product
        slope: Slope of the price-demand relationship
        elasticity_type: String describing the elasticity type
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the demand curve
    ax.plot(prices, demands, linewidth=2.5, color='#2E86AB', label='Predicted Demand')
    
    # Mark the base price point
    base_idx = np.argmin(np.abs(prices - base_price))
    base_demand = demands[base_idx]
    ax.scatter([base_price], [base_demand], color='#A23B72', s=100, 
              zorder=5, label=f'Current Price (${base_price:.2f})', marker='o')
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Labels and title
    ax.set_xlabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Demand (units)', fontsize=12, fontweight='bold')
    ax.set_title(f'Price Elasticity Analysis: {product_name}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add elasticity info as text box
    info_text = f'Slope: {slope:.6f}\nElasticity: {elasticity_type}'
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='best', fontsize=10)
    fig.tight_layout()
    
    # Save and display
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to '{output_path}'")
    plt.show()


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("PRICE ELASTICITY ANALYSIS")
    print("="*70 + "\n")
    
    try:
        # Load model and data
        model, df = load_data()
        
        # Select product sample
        sample = get_product_sample(df)
        product_name = sample.get("product_name", "Unknown Product")
        base_price = sample["price"]
        
        print(f"  Base Price: ${base_price:.2f}")
        print(f"  Competitor Price: ${sample.get('competitor_price', 'N/A'):.2f}")
        print()
        
        # Generate price range and predict demand
        print("Analyzing price elasticity...")
        prices = generate_price_range(base_price)
        demands = predict_demand_curve(model, sample, prices)
        
        # Calculate elasticity metrics
        slope, elasticity_type, r_squared = calculate_elasticity(prices, demands)
        
        # Display results
        print("\n" + "-"*70)
        print("ELASTICITY RESULTS")
        print("-"*70)
        print(f"Price Range:          ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"Demand Range:         {demands.min():.2f} - {demands.max():.2f} units")
        print(f"Slope:                {slope:.6f}")
        print(f"Elasticity Type:      {elasticity_type}")
        print(f"Linear Fit (R²):      {r_squared:.4f}")
        print("-"*70 + "\n")
        
        # Plot results
        plot_elasticity(prices, demands, product_name, base_price, 
                       slope, elasticity_type)
        
        print("✓ Analysis complete!\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=__import__('sys').stderr)
        exit(1)


if __name__ == "__main__":
    main()
