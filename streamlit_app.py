import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Import optimizer helpers (these will load the trained model once on import)
from price_optimizer import recommend_price, price_status
import price_optimizer

# Import dataset validation
from validate_dataset import validate_dataset_streamlit

# Import preprocessing utilities
from preprocess_inference import preprocess_for_inference

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Dynamic Electronics Pricing Intelligence System",
    layout="wide",
)

# -------------------------
# Currency Configuration
# -------------------------
USD_TO_INR = 91  # Exchange rate: 1 USD = 91 INR

# -------------------------
# Helpers
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed_pricing_data.csv"

def get_latest_row_per_product(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the latest row per product (sorted by date).
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        pd.DataFrame: One row per product (latest by date)
    """
    if "date" not in df.columns:
        st.error("Date column not found in dataset")
        st.stop()
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Sort by product and date, then get the last row per product
    latest = df.sort_values("date").groupby("product_name", as_index=False).tail(1)
    
    return latest

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic safety: ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data
def load_data_from_bytes(file_bytes) -> pd.DataFrame:
    """Load CSV from uploaded file bytes."""
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Basic safety: ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# -------------------------
# UI - Title & Sidebar
# -------------------------

st.title("Dynamic Electronics Pricing Intelligence System")
st.markdown("---")

# Sidebar: file uploader and product selection
with st.sidebar:
    st.header("Data & Product Selection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV (optional)",
        type="csv",
        help="Upload a custom pricing dataset. If not provided, default dataset is used."
    )
    
    # Load dataset (from uploaded file or default)
    if uploaded_file is not None:
        try:
            df = load_data_from_bytes(uploaded_file.getvalue())
            st.success("✓ Custom dataset loaded")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            st.markdown("*Falling back to default dataset...*")
            try:
                df = load_data(DATA_PATH)
            except FileNotFoundError:
                st.error(f"Processed data not found at {DATA_PATH}. Run preprocess.py first.")
                st.stop()
    else:
        # Load default dataset
        try:
            df = load_data(DATA_PATH)
        except FileNotFoundError:
            st.error(f"Processed data not found at {DATA_PATH}. Run preprocess.py first.")
            st.stop()
    
    if df.empty:
        st.error("Loaded dataset is empty.")
        st.stop()
    
    # Validate dataset has required columns
    if not validate_dataset_streamlit(df):
        st.stop()
    
    st.divider()
    
    # Preprocess the dataset
    try:
        df_processed = preprocess_for_inference(df)
        st.success("✓ Dataset preprocessed")
    except Exception as e:
        st.error(f"Error preprocessing dataset: {e}")
        st.stop()
    
    # Get latest row per product
    df_latest = get_latest_row_per_product(df_processed)
    
    # Product selection form
    with st.form(key="product_select_form"):
        st.subheader("Product Selector")
        product_list = df_latest["product_name"].dropna().unique().tolist()
        product_list.sort()
        selected_product = st.selectbox("Select Product", product_list)
        show_raw = st.checkbox("Show raw latest record", value=False)
        st.form_submit_button("Load")

# -------------------------
# Latest record for selected product
# -------------------------
product_rows = df_latest[df_latest["product_name"] == selected_product]
if product_rows.empty:
    st.error(f"No data for product: {selected_product}")
    st.stop()

latest = product_rows.iloc[0]  # Only one row per product after get_latest_row_per_product()

# Optional raw display
if show_raw:
    st.subheader("Latest raw record")
    # Convert dollar amounts to rupees for display
    display_record = latest.to_dict()
    
    # Currency conversion rate
    USD_TO_INR = 91
    
    # Price-related columns to convert
    price_columns = ["price", "competitor_price", "cost"]
    
    for col in price_columns:
        if col in display_record:
            display_record[col] = display_record[col] * USD_TO_INR
    
    st.json(display_record)

# -------------------------
# Market Snapshot
# -------------------------
st.subheader("Market Snapshot")
col1, col2, col3, col4, col5 = st.columns(5)

current_price = latest.get("price", np.nan)
competitor_price = latest.get("competitor_price", np.nan)
current_demand = latest.get("historical_demand", np.nan)
cost = latest.get("cost", np.nan)
popularity = latest.get("popularity", np.nan)

col1.metric("Current Price", f"₹{current_price * USD_TO_INR:.2f}" if pd.notna(current_price) else "N/A")
col2.metric("Competitor Price", f"₹{competitor_price * USD_TO_INR:.2f}" if pd.notna(competitor_price) else "N/A")
col3.metric("Historical Demand (Observed Sales)", f"{int(current_demand)}" if pd.notna(current_demand) else "N/A")
col4.metric("Cost", f"₹{cost * USD_TO_INR:.2f}" if pd.notna(cost) else "N/A")
col5.metric("Popularity", f"{popularity:.2f}" if pd.notna(popularity) else "N/A")

today = pd.Timestamp.now().date()
is_black_friday = (today.month == 11) and (today.day >= 21)
is_new_year = (today.month == 1) and (today.day <= 5)
is_festival = today.month in (10, 11, 12)
if is_black_friday:
    st.info("Today is Black Friday")
if is_new_year:
    st.info("Today is New Year")
if is_festival:
    st.info("Today is Festival Season")

st.markdown("---")

# -------------------------
# Demand Calculation Breakdown
# -------------------------
with st.expander("🔍 Demand Calculation Breakdown"):
    st.markdown("### Demand Generation Formula")
    
    st.markdown("""
    The demand is computed using a **data-driven approach** with the following formula:
    """)
    
    # Display the demand generation formula
    st.latex(r"""
    Q = Q_0 - \beta \cdot \frac{P}{P_{base}} - \theta \cdot \max(0, P - P_c) + \gamma \cdot Pop + \epsilon
    """)
    
    st.markdown("""
    **Where:**
    - **Q**: Predicted demand (units)
    - **Q₀**: Base demand (reference level)
    - **β**: Price elasticity coefficient (controls price sensitivity)
    - **P**: Current product price
    - **P_base**: Base reference price
    - **θ**: Competitive penalty coefficient (penalizes pricing above competitor)
    - **P_c**: Competitor's price
    - **γ**: Popularity boost coefficient
    - **Pop**: Product popularity score (0-1 range)
    - **ε**: Random noise/stochastic component
    
    **Model Characteristics:**
    - **Price-elastic component**: β·(P/P_base) captures how demand decreases with higher prices
    - **Competitive component**: θ·max(0, P - P_c) penalizes prices above competitor
    - **Popularity boost**: γ·Pop amplifies demand for popular products
    """)
    
    st.markdown("### Step-by-Step Breakdown for Current Product")
    
    # Create input features for demand calculation
    breakdown_features = {
        "price": current_price,
        "competitor_price": competitor_price,
        "price_gap": current_price - competitor_price if pd.notna(current_price) and pd.notna(competitor_price) else np.nan,
        "popularity": popularity,
        "month": latest.get("month", 1),
        "day_of_week": latest.get("day_of_week", 0),
        "rolling_7d_sales": latest.get("rolling_7d_sales", 0),
        "is_black_friday": 1 if is_black_friday else 0,
        "is_new_year": 1 if is_new_year else 0,
        "is_festival_season": 1 if is_festival else 0,
    }
    
    # Display individual feature contributions
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Pricing Features:**")
        st.metric("Current Price", f"₹{current_price * USD_TO_INR:.2f}", 
                 delta=f"Gap: ₹{(current_price - competitor_price) * USD_TO_INR:.2f}" if pd.notna(current_price) and pd.notna(competitor_price) else None)
        st.metric("Competitor Price", f"₹{competitor_price * USD_TO_INR:.2f}")
        st.metric("Price Gap", f"₹{breakdown_features['price_gap'] * USD_TO_INR:.2f}" if pd.notna(breakdown_features['price_gap']) else "N/A")
    
    with col_b:
        st.markdown("**Market Factors:**")
        st.metric("Popularity Score", f"{popularity:.3f}")
        st.metric("7-Day Rolling Sales", f"{int(breakdown_features['rolling_7d_sales'])}")
        st.metric("Seasonality", f"Month {int(breakdown_features['month'])}, Day {int(breakdown_features['day_of_week'])}")
    
    # ===== STEP-BY-STEP DEMAND CALCULATION =====
    st.markdown("---")
    st.markdown("### Numerical Example: Step-by-Step Calculation")
    
    # Extract values from latest record
    P = current_price  # Current price
    P_c = competitor_price  # Competitor price
    Pop = popularity  # Popularity score
    Cost = cost  # Cost
    
    # Define reasonable formula constants (based on product category)
    Q0 = 250.0  # Base demand (units)
    beta = 1.5  # Price elasticity coefficient (how sensitive demand is to price)
    theta = 2.5  # Competitive penalty coefficient
    gamma = 50.0  # Popularity boost coefficient
    P_base = 500.0  # Base reference price
    
    # Display the constants used
    st.markdown("**Formula Constants:**")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric("Base Demand (Q₀)", f"{Q0:.0f} units")
    col_c2.metric("Price Elasticity (β)", f"{beta:.2f}")
    col_c3.metric("Competitive Penalty (θ)", f"{theta:.2f}")
    col_c4.metric("Popularity Boost (γ)", f"{gamma:.1f}")
    
    # Step 1: Normalized Price
    normalized_price = P / P_base
    st.markdown(f"**Step 1: Normalized Price**")
    st.latex(rf"P_{{norm}} = \frac{{P}}{{P_{{base}}}} = \frac{{{P:.2f}}}{{{P_base:.2f}}} = {normalized_price:.4f}")
    
    # Step 2: Price Effect
    price_effect = beta * normalized_price
    st.markdown(f"**Step 2: Price Effect**")
    st.latex(rf"\text{{Price Effect}} = \beta \cdot P_{{norm}} = {beta:.2f} \times {normalized_price:.4f} = {price_effect:.4f}")
    
    # Step 3: Competitive Penalty
    price_gap = P - P_c
    competitive_penalty = max(0, price_gap)
    competitive_penalty_effect = theta * competitive_penalty / 10.0  # Normalize by 10
    st.markdown(f"**Step 3: Competitive Penalty**")
    st.latex(rf"\text{{Penalty}} = \theta \cdot \max(0, P - P_c) = {theta:.2f} \times \max(0, {P:.2f} - {P_c:.2f})")
    st.latex(rf"= {theta:.2f} \times {competitive_penalty:.2f} = {competitive_penalty_effect:.4f}")
    
    # Step 4: Popularity Effect
    popularity_effect = gamma * Pop
    st.markdown(f"**Step 4: Popularity Effect**")
    st.latex(rf"\text{{Popularity Effect}} = \gamma \cdot Pop = {gamma:.1f} \times {Pop:.3f} = {popularity_effect:.4f}")
    
    # Step 5: Final Calculated Demand
    final_calculated_demand = Q0 - price_effect - competitive_penalty_effect + popularity_effect
    st.markdown(f"**Step 5: Final Calculated Demand**")
    st.latex(rf"""
    Q = Q_0 - \beta \cdot \frac{{P}}{{P_{{base}}}} - \theta \cdot \max(0, P - P_c) + \gamma \cdot Pop
    """)
    st.latex(rf"Q = {Q0:.0f} - {price_effect:.4f} - {competitive_penalty_effect:.4f} + {popularity_effect:.4f} = {final_calculated_demand:.2f} \text{{ units}}")
    
    st.success(f"**Calculated Demand at ₹{P * USD_TO_INR:.2f}: {final_calculated_demand:.0f} units**")
    
    # Calculate actual predicted demand from model
    try:
        from feature_reorder import reorder_features, get_feature_order
        
        # Prepare features for model prediction
        demand_input = pd.DataFrame([{
            "price": current_price,
            "competitor_price": competitor_price,
            "price_gap": breakdown_features['price_gap'],
            "popularity": popularity,
            "month": breakdown_features['month'],
            "day_of_week": breakdown_features['day_of_week'],
            "month_sin": np.sin(2 * np.pi * breakdown_features['month'] / 12),
            "month_cos": np.cos(2 * np.pi * breakdown_features['month'] / 12),
            "dow_sin": np.sin(2 * np.pi * breakdown_features['day_of_week'] / 7),
            "dow_cos": np.cos(2 * np.pi * breakdown_features['day_of_week'] / 7),
            "rolling_7d_sales": breakdown_features['rolling_7d_sales'],
            "is_black_friday": breakdown_features['is_black_friday'],
            "is_new_year": breakdown_features['is_new_year'],
            "is_festival_season": breakdown_features['is_festival_season'],
        }])
        
        demand_input = reorder_features(demand_input, get_feature_order(), drop_extra=True)
        predicted_demand_value = price_optimizer.model.predict(demand_input)[0]
        
        st.markdown("---")
        st.markdown("### XGBoost Model Prediction")
        st.info(f"**ML Model Predicted Demand: {predicted_demand_value:.0f} units**")
        
        # Show comparison
        if pd.notna(current_demand):
            demand_diff = predicted_demand_value - current_demand
            demand_pct = (demand_diff / current_demand * 100) if current_demand != 0 else 0
            st.metric("Historical vs Predicted", f"{int(current_demand)} → {predicted_demand_value:.0f} units", delta=f"{demand_diff:+.0f} ({demand_pct:+.1f}%)")
        
    except Exception as e:
        st.warning(f"Could not calculate ML predictions: {str(e)}")
    
    # ===== INTERACTIVE VISUALIZATION BUTTONS =====
    st.markdown("---")
    st.markdown("### 📊 Interactive Demand Analysis")
    
    # Initialize session state for buttons
    if "show_elasticity" not in st.session_state:
        st.session_state.show_elasticity = False
    if "show_components" not in st.session_state:
        st.session_state.show_components = False
    if "show_popularity" not in st.session_state:
        st.session_state.show_popularity = False
    if "show_competitor" not in st.session_state:
        st.session_state.show_competitor = False
    if "show_uncertainty" not in st.session_state:
        st.session_state.show_uncertainty = False
    
    # Create buttons in two rows
    st.markdown("**Click buttons to toggle analysis visualizations:**")
    
    # Row 1: First 3 buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("📉 Price Elasticity", key="btn_elasticity", use_container_width=True):
            st.session_state.show_elasticity = not st.session_state.show_elasticity
    with col_btn2:
        if st.button("🧩 Demand Components", key="btn_components", use_container_width=True):
            st.session_state.show_components = not st.session_state.show_components
    with col_btn3:
        if st.button("🔥 Popularity Impact", key="btn_popularity", use_container_width=True):
            st.session_state.show_popularity = not st.session_state.show_popularity
    
    # Row 2: Last 2 buttons
    col_btn4, col_btn5 = st.columns(2)
    with col_btn4:
        if st.button("🏷️ Competitor Impact", key="btn_competitor", use_container_width=True):
            st.session_state.show_competitor = not st.session_state.show_competitor
    with col_btn5:
        if st.button("🎲 Demand Uncertainty", key="btn_uncertainty", use_container_width=True):
            st.session_state.show_uncertainty = not st.session_state.show_uncertainty
    
    # Display visualizations based on button states
    
    # 1. PRICE ELASTICITY VISUALIZATION - PLOTLY INTERACTIVE CHART
    if st.session_state.show_elasticity:
        st.markdown("#### 📉 Price Elasticity Analysis")
        st.markdown(f"Interactive chart showing how demand changes as price varies")
        
        # Create price range around current price
        price_range_viz = np.linspace(P * 0.7, P * 1.3, 50)
        elasticity_demands = []
        
        for price_point in price_range_viz:
            normalized_p = price_point / P_base
            demand_at_price = Q0 - beta * normalized_p - theta * max(0, price_point - P_c) / 10.0 + gamma * Pop
            elasticity_demands.append(max(10, demand_at_price))  # Floor at 10
        
        # Create Plotly interactive chart
        fig = go.Figure()
        
        # Add demand curve
        fig.add_trace(go.Scatter(
            x=price_range_viz * USD_TO_INR,
            y=elasticity_demands,
            mode='lines',
            name='Demand Curve',
            line=dict(color='#1f77b4', width=3),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
            hovertemplate='<b>Price: ₹%{x:.2f}</b><br>Demand: %{y:.0f} units<extra></extra>'
        ))
        
        # Add vertical line for current price
        fig.add_vline(
            x=P * USD_TO_INR,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Current Price: ₹{P * USD_TO_INR:.2f}",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="green"
        )
        
        # Add current demand point
        current_idx = np.argmin(np.abs(price_range_viz - P))
        fig.add_trace(go.Scatter(
            x=[P * USD_TO_INR],
            y=[elasticity_demands[current_idx]],
            mode='markers',
            name='Current Point',
            marker=dict(size=12, color='green', symbol='star'),
            hovertemplate='<b>Current Price: ₹%{x:.2f}</b><br>Demand: %{y:.0f} units<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='Price Elasticity: Demand vs Price',
            xaxis_title='Price (₹)',
            yaxis_title='Predicted Demand (units)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            font=dict(size=12),
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. DEMAND COMPONENTS VISUALIZATION - PLOTLY BAR CHART
    if st.session_state.show_components:
        st.markdown("#### 🧩 Demand Components Breakdown")
        st.markdown(f"Shows the contribution of each component to total demand")
        
        components = {
            'Base Demand': Q0,
            'Price Effect': -price_effect,
            'Competitive Penalty': -competitive_penalty_effect,
            'Popularity Boost': popularity_effect
        }
        
        # Create Plotly bar chart
        fig_comp = go.Figure()
        
        colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db']
        
        fig_comp.add_trace(go.Bar(
            x=list(components.keys()),
            y=list(components.values()),
            marker=dict(color=colors, line=dict(color='black', width=1.5)),
            text=[f'{v:.1f}' for v in components.values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Value: %{y:.2f} units<extra></extra>',
            showlegend=False
        ))
        
        # Add zero line
        fig_comp.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        
        # Update layout
        fig_comp.update_layout(
            title='Demand Components Breakdown',
            xaxis_title='Components',
            yaxis_title='Contribution to Demand (units)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            font=dict(size=12),
            showlegend=False,
            xaxis=dict(tickangle=-15)
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
    
    # 3. POPULARITY IMPACT VISUALIZATION - PLOTLY LINE CHART
    if st.session_state.show_popularity:
        st.markdown("#### 🔥 Popularity Impact on Demand")
        st.markdown(f"Interactive chart showing how different popularity scores affect demand at current price")
        
        # Compute demand across 50 popularity values
        popularity_range = np.linspace(0, 1, 50)
        popularity_demands = []
        
        for pop_score in popularity_range:
            demand_at_pop = Q0 - price_effect - competitive_penalty_effect + gamma * pop_score
            popularity_demands.append(max(10, demand_at_pop))
        
        # Create Plotly line chart
        fig_pop = go.Figure()
        
        # Add demand curve
        fig_pop.add_trace(go.Scatter(
            x=popularity_range,
            y=popularity_demands,
            mode='lines',
            name='Demand Curve',
            line=dict(color='#e74c3c', width=3),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)',
            hovertemplate='<b>Popularity: %{x:.2f}</b><br>Demand: %{y:.0f} units<extra></extra>'
        ))
        
        # Add current popularity point marker
        current_pop_idx = np.argmin(np.abs(popularity_range - Pop))
        fig_pop.add_trace(go.Scatter(
            x=[Pop],
            y=[popularity_demands[current_pop_idx]],
            mode='markers',
            name='Current Popularity',
            marker=dict(size=14, color='green', symbol='star', line=dict(color='darkgreen', width=2)),
            hovertemplate='<b>Current Popularity: %{x:.3f}</b><br>Demand: %{y:.0f} units<extra></extra>'
        ))
        
        # Add vertical line at current popularity
        fig_pop.add_vline(
            x=Pop,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Current: {Pop:.3f}",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="green"
        )
        
        # Update layout
        fig_pop.update_layout(
            title='Popularity Impact on Demand',
            xaxis_title='Popularity Score (0-1)',
            yaxis_title='Predicted Demand (units)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            font=dict(size=12),
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_pop, use_container_width=True)
    
    # 4. COMPETITOR IMPACT VISUALIZATION - PLOTLY LINE CHART
    if st.session_state.show_competitor:
        st.markdown("#### 🏷️ Competitor Price Impact")
        st.markdown(f"Interactive chart showing how competitor prices affect our demand (±20% range)")
        
        # Create competitor price range ±20% around current competitor price
        competitor_range = np.linspace(P_c * 0.8, P_c * 1.2, 50)
        competitor_demands = []
        
        for comp_price in competitor_range:
            comp_gap = max(0, P - comp_price)
            demand_at_comp = Q0 - price_effect - theta * comp_gap / 10.0 + popularity_effect
            competitor_demands.append(max(10, demand_at_comp))
        
        # Create Plotly line chart
        fig_comp_impact = go.Figure()
        
        # Add demand curve
        fig_comp_impact.add_trace(go.Scatter(
            x=competitor_range * USD_TO_INR,
            y=competitor_demands,
            mode='lines',
            name='Demand Curve',
            line=dict(color='#f39c12', width=3),
            fill='tozeroy',
            fillcolor='rgba(243, 156, 18, 0.2)',
            hovertemplate='<b>Competitor Price: ₹%{x:.2f}</b><br>Our Demand: %{y:.0f} units<extra></extra>'
        ))
        
        # Add current competitor price marker
        current_comp_idx = np.argmin(np.abs(competitor_range - P_c))
        fig_comp_impact.add_trace(go.Scatter(
            x=[P_c * USD_TO_INR],
            y=[competitor_demands[current_comp_idx]],
            mode='markers',
            name='Current Competitor Price',
            marker=dict(size=14, color='red', symbol='diamond', line=dict(color='darkred', width=2)),
            hovertemplate='<b>Current Competitor Price: ₹%{x:.2f}</b><br>Our Demand: %{y:.0f} units<extra></extra>'
        ))
        
        # Add vertical line at current competitor price
        fig_comp_impact.add_vline(
            x=P_c * USD_TO_INR,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Current: ₹{P_c * USD_TO_INR:.2f}",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="red"
        )
        
        # Update layout
        fig_comp_impact.update_layout(
            title='Competitor Price Impact on Our Demand',
            xaxis_title='Competitor Price (₹)',
            yaxis_title='Our Predicted Demand (units)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            font=dict(size=12),
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_comp_impact, use_container_width=True)
    
    # 5. DEMAND UNCERTAINTY VISUALIZATION - PLOTLY HISTOGRAM
    if st.session_state.show_uncertainty:
        st.markdown("#### 🎲 Demand Uncertainty Distribution")
        st.markdown(f"Interactive histogram showing Monte Carlo simulated demand distribution")
        
        # Simulate demand with uncertainty using Monte Carlo (10,000 simulations)
        noise_std = final_calculated_demand * 0.15  # 15% std dev
        simulated_demands = np.random.normal(final_calculated_demand, noise_std, 10000)
        simulated_demands = np.clip(simulated_demands, 10, None)  # Floor at 10
        
        # Calculate statistics
        mean_demand = np.mean(simulated_demands)
        percentile_5 = np.percentile(simulated_demands, 5)
        percentile_95 = np.percentile(simulated_demands, 95)
        std_dev = np.std(simulated_demands)
        
        # Create Plotly histogram
        fig_uncert = go.Figure()
        
        # Add histogram with 20 bins
        fig_uncert.add_trace(go.Histogram(
            x=simulated_demands,
            nbinsx=20,
            name='Demand Distribution',
            marker=dict(color='#9b59b6', line=dict(color='black', width=1)),
            hovertemplate='<b>Demand Range: %{x:.0f} units</b><br>Frequency: %{y}<extra></extra>'
        ))
        
        # Add mean line
        fig_uncert.add_vline(
            x=mean_demand,
            line_dash="solid",
            line_color="green",
            line_width=3,
            annotation_text=f"Mean: {mean_demand:.0f}",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="green"
        )
        
        # Add 5th percentile line
        fig_uncert.add_vline(
            x=percentile_5,
            line_dash="dot",
            line_color="orange",
            line_width=2,
            annotation_text=f"5th %ile: {percentile_5:.0f}",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="orange"
        )
        
        # Add 95th percentile line
        fig_uncert.add_vline(
            x=percentile_95,
            line_dash="dot",
            line_color="red",
            line_width=2,
            annotation_text=f"95th %ile: {percentile_95:.0f}",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="red"
        )
        
        # Update layout
        fig_uncert.update_layout(
            title='Demand Uncertainty: Monte Carlo Simulation (10,000 iterations)',
            xaxis_title='Demand (units)',
            yaxis_title='Frequency',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            font=dict(size=12),
            showlegend=False,
            bargap=0.1
        )
        
        st.plotly_chart(fig_uncert, use_container_width=True)
        
        # Display uncertainty statistics
        st.markdown("**Demand Uncertainty Statistics:**")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Mean Demand", f"{mean_demand:.0f} units")
        stat_col2.metric("Std Deviation", f"{std_dev:.0f} units")
        stat_col3.metric("5th Percentile", f"{percentile_5:.0f} units")
        stat_col4.metric("95th Percentile", f"{percentile_95:.0f} units")

st.markdown("---")

# -------------------------
# Optimal Price Recommendation (Calculate & Display First)
# -------------------------
try:
    rec_price, rec_profit = recommend_price(latest)
    status = price_status(latest.get("price", 0.0), rec_price)
except Exception as e:
    rec_price, rec_profit = None, None
    status = None
    st.warning(f"Recommendation could not be calculated: {e}")

st.subheader("Optimal Price Recommendation")
if rec_price is None:
    st.error("Could not compute recommendation.")
else:
    if status == "Price is Optimal":
        st.success(f"Recommended Price: ₹{rec_price * USD_TO_INR:.2f} — {status}")
    elif status and status.startswith("Overpriced"):
        st.warning(f"Recommended Price: ₹{rec_price * USD_TO_INR:.2f} — {status}")
    else:
        # Use transparent markdown instead of st.info()
        st.markdown(f"**Recommended Price:** ₹{rec_price * USD_TO_INR:.2f} — {status}")

    st.write(f"Expected Profit at recommended price: ₹{rec_profit * USD_TO_INR:.2f}")

st.markdown("---")

# -------------------------
# Price Simulation Engine
# -------------------------
st.subheader("Price Simulation")

# Build candidate price range
if pd.isna(current_price):
    st.error("Current price missing for this product.")
    st.stop()

price_range = np.linspace(current_price * 0.8, current_price * 1.2, 50)

# Prepare constants from latest row
competitor_price_val = latest.get("competitor_price", 0.0)
popularity_val = latest.get("popularity", 0.0)
month_val = latest.get("month", pd.NaT)
day_of_week_val = latest.get("day_of_week", 0)
rolling_sales_val = latest.get("rolling_7d_sales", 0.0)
cost_val = latest.get("cost", 0.0)

# Cyclical encodings (model expects these)
month_num = int(month_val) if pd.notna(month_val) else 1
dow_num = int(day_of_week_val) if pd.notna(day_of_week_val) else 0
month_sin_val = np.sin(2 * np.pi * month_num / 12)
month_cos_val = np.cos(2 * np.pi * month_num / 12)
dow_sin_val = np.sin(2 * np.pi * dow_num / 7)
dow_cos_val = np.cos(2 * np.pi * dow_num / 7)

# Calendar flags (use latest date if available)
date_val = latest.get("date", pd.Timestamp.now())
if hasattr(date_val, "month") and hasattr(date_val, "day"):
    is_black_friday_val = 1 if (date_val.month == 11 and date_val.day >= 21) else 0
    is_new_year_val = 1 if (date_val.month == 1 and date_val.day <= 5) else 0
    is_festival_season_val = 1 if date_val.month in (10, 11, 12) else 0
else:
    is_black_friday_val = int(latest.get("is_black_friday", 0))
    is_new_year_val = int(latest.get("is_new_year", 0))
    is_festival_season_val = int(latest.get("is_festival_season", 0))

# Base feature values (will be aligned to model's expected feature names)
base_features = {
    "price": current_price,
    "competitor_price": competitor_price_val,
    "price_gap": current_price - competitor_price_val,
    "popularity": popularity_val,
    "month": month_num,
    "day_of_week": dow_num,
    "month_sin": month_sin_val,
    "month_cos": month_cos_val,
    "dow_sin": dow_sin_val,
    "dow_cos": dow_cos_val,
    "rolling_7d_sales": rolling_sales_val,
    "is_black_friday": is_black_friday_val,
    "is_new_year": is_new_year_val,
    "is_festival_season": is_festival_season_val,
}

# Predict demand using the loaded model (price_optimizer loaded model on import)
model = getattr(price_optimizer, "model", None)
if model is None:
    st.error("Demand model not loaded. Ensure demand_model.pkl exists and price_optimizer loads it.")
    st.stop()

# Align simulation inputs to the model's actual feature set and order
model_features = None
try:
    booster = model.get_booster()
    model_features = getattr(booster, "feature_names", None)
except Exception:
    model_features = getattr(model, "feature_names_in_", None)

if not model_features:
    # Fallback to the training-time feature list (kept in sync with train_model.py)
    model_features = [
        "price",
        "competitor_price",
        "price_gap",
        "popularity",
        "month",
        "day_of_week",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "rolling_7d_sales",
        "is_black_friday",
        "is_new_year",
        "is_festival_season",
    ]

data = {}
for name in model_features:
    if name == "price":
        data[name] = price_range
    elif name == "price_gap":
        data[name] = price_range - competitor_price_val
    else:
        # Broadcast scalar feature across the simulation grid
        value = base_features.get(name, 0.0)
        data[name] = np.full_like(price_range, value, dtype=float)

sim_inputs = pd.DataFrame(data)

forecasted_demand = model.predict(sim_inputs)
predicted_profit = (price_range - cost_val) * forecasted_demand

# Store results in DataFrame
sim_results = pd.DataFrame({
    "price": price_range,
    "forecasted_demand": forecasted_demand,
    "predicted_profit": predicted_profit,
})

# -------------------------
# Interactive Price Slider
# -------------------------
st.subheader("Interactive Price Adjustment")

# Initialize session state for slider if not present
if "selected_price" not in st.session_state:
    st.session_state.selected_price = float(current_price)

# Create price slider
selected_price_usd = st.slider(
    "Adjust Price (₹)",
    min_value=float(price_range.min()),
    max_value=float(price_range.max()),
    value=float(st.session_state.selected_price),
    step=(float(price_range.max()) - float(price_range.min())) / 49,  # Match the 50-point granularity
    format="%.2f"
)

# Update session state
st.session_state.selected_price = selected_price_usd

# Calculate predicted demand and profit for selected price
selected_features = {}
for name in model_features:
    if name == "price":
        selected_features[name] = selected_price_usd
    elif name == "price_gap":
        selected_features[name] = selected_price_usd - competitor_price_val
    else:
        selected_features[name] = base_features.get(name, 0.0)

selected_input_df = pd.DataFrame([selected_features])
selected_demand = model.predict(selected_input_df)[0]
selected_profit = (selected_price_usd - cost_val) * selected_demand

# Display live metrics for selected price
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric(
        "Selected Price",
        f"₹{selected_price_usd * USD_TO_INR:.2f}",
        f"Current: ₹{current_price * USD_TO_INR:.2f}"
    )
with col_m2:
    st.metric(
        "Predicted Demand",
        f"{selected_demand:.0f} units"
    )
with col_m3:
    st.metric(
        "Predicted Profit",
        f"₹{selected_profit * USD_TO_INR:.2f}"
    )

# -------------------------
# Visualization
# -------------------------
st.subheader("Visualizations")

# Create Plotly figures for interactive visualizations
col_a, col_b = st.columns(2)

# 1. DEMAND VS PRICE - PLOTLY LINE CHART
with col_a:
    st.markdown("#### Demand vs Price")
    
    fig_demand = go.Figure()
    
    # Add demand curve
    fig_demand.add_trace(go.Scatter(
        x=sim_results["price"] * USD_TO_INR,
        y=sim_results["forecasted_demand"],
        mode='lines',
        name='Demand Curve',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='<b>Price: ₹%{x:.2f}</b><br>Demand: %{y:.0f} units<extra></extra>'
    ))
    
    # Add selected price marker
    fig_demand.add_trace(go.Scatter(
        x=[selected_price_usd * USD_TO_INR],
        y=[selected_demand],
        mode='markers',
        name='Selected Price',
        marker=dict(size=16, color='#ff1744', symbol='diamond', line=dict(color='darkred', width=2)),
        hovertemplate='<b>Selected: ₹%{x:.2f}</b><br>Demand: %{y:.0f} units<extra></extra>'
    ))
    
    # Add vertical line at selected price
    fig_demand.add_vline(
        x=selected_price_usd * USD_TO_INR,
        line_dash="solid",
        line_color="#ff1744",
        line_width=2,
        annotation_text=f"₹{selected_price_usd * USD_TO_INR:.2f}",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color="#ff1744"
    )
    
    # Add optimal price marker if available
    if rec_price is not None:
        optimal_idx = np.argmin(np.abs(sim_results["price"] - rec_price))
        optimal_demand = sim_results["forecasted_demand"].iloc[optimal_idx]
        
        fig_demand.add_trace(go.Scatter(
            x=[rec_price * USD_TO_INR],
            y=[optimal_demand],
            mode='markers',
            name='Optimal Price',
            marker=dict(size=14, color='green', symbol='star', line=dict(color='darkgreen', width=2)),
            hovertemplate='<b>Optimal Price: ₹%{x:.2f}</b><br>Demand: %{y:.0f} units<extra></extra>'
        ))
        
        # Add vertical line at optimal price
        fig_demand.add_vline(
            x=rec_price * USD_TO_INR,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Optimal: ₹{rec_price * USD_TO_INR:.2f}",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="green"
        )
    
    # Update layout
    fig_demand.update_layout(
        title='Demand vs Price',
        xaxis_title='Price (₹)',
        yaxis_title='Predicted Demand (units)',
        hovermode='x unified',
        template='plotly_white',
        height=450,
        font=dict(size=11),
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig_demand, use_container_width=True)

# 2. PROFIT VS PRICE - PLOTLY LINE CHART
with col_b:
    st.markdown("#### Profit vs Price")
    
    fig_profit = go.Figure()
    
    # Add profit curve
    fig_profit.add_trace(go.Scatter(
        x=sim_results["price"] * USD_TO_INR,
        y=sim_results["predicted_profit"] * USD_TO_INR,
        mode='lines',
        name='Profit Curve',
        line=dict(color='#ff7f0e', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.2)',
        hovertemplate='<b>Price: ₹%{x:.2f}</b><br>Profit: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Add selected price marker
    fig_profit.add_trace(go.Scatter(
        x=[selected_price_usd * USD_TO_INR],
        y=[selected_profit * USD_TO_INR],
        mode='markers',
        name='Selected Price',
        marker=dict(size=16, color='#ff1744', symbol='diamond', line=dict(color='darkred', width=2)),
        hovertemplate='<b>Selected: ₹%{x:.2f}</b><br>Profit: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Add vertical line at selected price
    fig_profit.add_vline(
        x=selected_price_usd * USD_TO_INR,
        line_dash="solid",
        line_color="#ff1744",
        line_width=2,
        annotation_text=f"₹{selected_price_usd * USD_TO_INR:.2f}",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color="#ff1744"
    )
    
    # Add optimal price marker if available
    if rec_price is not None:
        optimal_idx = np.argmin(np.abs(sim_results["price"] - rec_price))
        optimal_profit = sim_results["predicted_profit"].iloc[optimal_idx]
        
        fig_profit.add_trace(go.Scatter(
            x=[rec_price * USD_TO_INR],
            y=[optimal_profit * USD_TO_INR],
            mode='markers',
            name='Optimal Price',
            marker=dict(size=14, color='green', symbol='star', line=dict(color='darkgreen', width=2)),
            hovertemplate='<b>Optimal Price: ₹%{x:.2f}</b><br>Profit: ₹%{y:,.0f}<extra></extra>'
        ))
        
        # Add vertical line at optimal price
        fig_profit.add_vline(
            x=rec_price * USD_TO_INR,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Optimal: ₹{rec_price * USD_TO_INR:.2f}",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="green"
        )
    
    # Update layout
    fig_profit.update_layout(
        title='Profit vs Price',
        xaxis_title='Price (₹)',
        yaxis_title='Predicted Profit (₹)',
        hovermode='x unified',
        template='plotly_white',
        height=450,
        font=dict(size=11),
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig_profit, use_container_width=True)

st.markdown("---")

# -------------------------
# Adaptive Learning Simulation
# -------------------------
if rec_price is not None:
    st.subheader("Adaptive Learning Simulation")
    
    # Simulate demand at optimal price
    optimal_price_input = pd.DataFrame([{
        name: (rec_price if name == "price" else 
                rec_price - competitor_price_val if name == "price_gap" else 
                base_features.get(name, 0.0))
        for name in model_features
    }])
    predicted_demand_at_optimal = model.predict(optimal_price_input)[0]
    
    # Simulate rolling sales update with exponential smoothing
    rolling_7d_sales = latest.get("rolling_7d_sales", 0.0)
    new_rolling_sales = 0.7 * rolling_7d_sales + 0.3 * predicted_demand_at_optimal
    
    # Display the update
    col1_update, col2_update = st.columns(2)
    with col1_update:
        st.metric("Previous Rolling 7-day Sales", f"{rolling_7d_sales:.2f} units")
    with col2_update:
        st.metric("Simulated Updated Rolling Sales", f"{new_rolling_sales:.2f} units")
    
    st.info("📊 Simulated adaptive update: rolling sales adjusted based on predicted demand at optimal price.")

# -------------------------
# Notes and footer
# -------------------------
st.markdown("---")
st.caption("Model predictions are based on a trained XGBoost demand model. Use recommendations as decision support, not absolute rules.")
