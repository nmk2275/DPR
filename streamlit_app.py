import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Import optimizer helpers (these will load the trained model once on import)
from price_optimizer import recommend_price, price_status, compare_pricing_strategies
import price_optimizer

# Import dataset validation
from validate_dataset import validate_dataset_streamlit

# Import preprocessing utilities
from preprocess_inference import preprocess_for_inference

# Import prediction validation
from prediction_validator import validate_prediction_features
from feature_engineering import EXPECTED_FEATURES

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

# Extract key values (available throughout the app)
inventory_level = latest.get("inventory_level", 250)
max_inventory = latest.get("max_inventory", 500)
cost = latest.get("cost", np.nan)

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
        
        # ========================================================
        # VALIDATION BEFORE PREDICTION
        # Assert features match training exactly
        # ========================================================
        is_valid, validation_msg = validate_prediction_features(
            demand_input,
            raise_error=True,
            verbose=True,
            exclude_columns=[]  # No metadata columns here
        )
        
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
        
        # PIE CHART: DEMAND COMPONENTS WEIGHTAGE
        st.markdown("---")
        st.markdown("#### 📊 Demand Components Weightage Breakdown")
        
        # Calculate positive and negative contributions
        positive_components = {'Base Demand': Q0}
        if popularity_effect > 0:
            positive_components['Popularity Boost'] = popularity_effect
        
        negative_components = {}
        if price_effect > 0:
            negative_components['Price Effect'] = price_effect
        if competitive_penalty_effect > 0:
            negative_components['Competitor Penalty'] = competitive_penalty_effect
        
        # Create two-column layout for pie charts
        col_pie1, col_pie2 = st.columns(2)
        
        # POSITIVE COMPONENTS PIE CHART
        with col_pie1:
            st.markdown("**✅ Positive Contributors:**")
            
            pos_labels = list(positive_components.keys())
            pos_values = list(positive_components.values())
            pos_total = sum(pos_values)
            
            fig_pie_pos = go.Figure(data=[go.Pie(
                labels=pos_labels,
                values=pos_values,
                marker=dict(colors=['#2ecc71', '#27ae60'], line=dict(color='black', width=2)),
                textinfo='label+percent+value',
                textposition='inside',
                texttemplate='<b>%{label}</b><br>%{percent}<br>%{value:.1f}',
                hovertemplate='<b>%{label}</b><br>Value: %{value:.1f}<br>Percentage: %{percent}<extra></extra>',
                hole=0.3
            )])
            
            fig_pie_pos.update_layout(
                title='Positive Demand Factors',
                height=450,
                font=dict(size=10),
                showlegend=True,
                legend=dict(x=0.0, y=-0.2)
            )
            
            st.plotly_chart(fig_pie_pos, use_container_width=True)
        
        # NEGATIVE COMPONENTS PIE CHART
        with col_pie2:
            st.markdown("**❌ Negative Contributors:**")
            
            if negative_components:
                neg_labels = list(negative_components.keys())
                neg_values = list(negative_components.values())
                neg_total = sum(neg_values)
                
                fig_pie_neg = go.Figure(data=[go.Pie(
                    labels=neg_labels,
                    values=neg_values,
                    marker=dict(colors=['#e74c3c', '#c0392b'], line=dict(color='black', width=2)),
                    textinfo='label+percent+value',
                    textposition='inside',
                    texttemplate='<b>%{label}</b><br>%{percent}<br>%{value:.1f}',
                    hovertemplate='<b>%{label}</b><br>Value: %{value:.1f}<br>Percentage: %{percent}<extra></extra>',
                    hole=0.3
                )])
                
                fig_pie_neg.update_layout(
                    title='Negative Demand Factors',
                    height=450,
                    font=dict(size=10),
                    showlegend=True,
                    legend=dict(x=0.0, y=-0.2)
                )
                
                st.plotly_chart(fig_pie_neg, use_container_width=True)
            else:
                st.info("No negative factors affecting demand")
        
        # PIE CHART: POPULARITY COMPOSITION
        st.markdown("---")
        st.markdown("#### 🔥 Popularity Component Weightages")
        
        # Extract popularity components
        search_trend_normalized = latest.get('search_trend', 50) / 100.0
        review_velocity_normalized = latest.get('review_velocity', 15) / 30.0
        social_buzz_normalized = latest.get('social_buzz', 50) / 100.0
        
        # Calculate weighted components
        search_contribution = 0.40 * search_trend_normalized
        review_contribution = 0.30 * review_velocity_normalized
        social_contribution = 0.30 * social_buzz_normalized
        
        popularity_components = {
            'Search Trend (40%)': search_contribution,
            'Review Velocity (30%)': review_contribution,
            'Social Buzz (30%)': social_contribution
        }
        
        fig_pie_pop = go.Figure(data=[go.Pie(
            labels=list(popularity_components.keys()),
            values=list(popularity_components.values()),
            marker=dict(
                colors=['#3498db', '#9b59b6', '#f39c12'],
                line=dict(color='black', width=2)
            ),
            textinfo='label+percent+value',
            textposition='auto',
            texttemplate='<b>%{label}</b><br>%{percent}<br>%{value:.3f}',
            hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_pie_pop.update_layout(
            title='Popularity Score Composition (Total: {:.3f})'.format(Pop),
            height=500,
            font=dict(size=11),
            showlegend=True,
            legend=dict(x=0.7, y=0.5)
        )
        
        st.plotly_chart(fig_pie_pop, use_container_width=True)
        
        # Add interpretation
        st.info(f"""
        **📊 Popularity Breakdown:**
        - **Search Trend (40%):** {search_contribution:.3f} - How often customers search for this product
        - **Review Velocity (30%):** {review_contribution:.3f} - How many reviews are being posted
        - **Social Buzz (30%):** {social_contribution:.3f} - Social media mentions and engagement
        
        **Total Popularity Score:** {Pop:.3f} (out of 1.0)
        """)
    
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

    # Display profit with better formatting and clarity
    col_profit1, col_profit2 = st.columns(2)
    with col_profit1:
        st.metric(
            "💰 Expected Daily Profit",
            f"₹{rec_profit * USD_TO_INR:,.0f}",
            f"@ ₹{rec_price * USD_TO_INR:.0f}/unit"
        )
    with col_profit2:
        # Calculate and show margin percentage
        margin_per_unit = rec_price - latest.get("cost", 0)
        margin_pct = (margin_per_unit / rec_price * 100) if rec_price > 0 else 0
        st.metric(
            "📈 Profit Margin",
            f"{margin_pct:.1f}%",
            f"${margin_per_unit:.2f}/unit"
        )
    
    # --------- Inventory-Aware Recommendation Context ---------
    # (inventory_level and max_inventory already defined at app startup)
    inventory_ratio = inventory_level / max_inventory
    
    st.divider()
    
    # Display inventory metrics
    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        st.metric(
            "📦 Inventory Level",
            f"{inventory_level:.0f} units",
            f"Max: {max_inventory:.0f} units"
        )
    with col_inv2:
        st.metric(
            "📊 Inventory Ratio",
            f"{inventory_ratio:.1%}",
            delta=None
        )
    
    # Contextual warnings based on inventory levels
    if inventory_ratio < 0.2:
        st.warning(
            f"""🚨 **Low Stock Alert!**
            - Inventory is critically low ({inventory_ratio:.1%} of max capacity)
            - Price has been increased by 8% to maximize revenue per unit
            - Consider reordering stock soon to avoid stockouts"""
        )
    elif inventory_ratio > 0.8:
        st.info(
            f"""📈 **Overstock Situation**
            - Inventory is excessive ({inventory_ratio:.1%} of max capacity)
            - Price has been reduced by 8% to accelerate turnover
            - Prioritize selling existing stock to reduce holding costs"""
        )
    else:
        st.success(
            f"""✓ **Optimal Inventory Level**
            - Inventory at healthy level ({inventory_ratio:.1%} of capacity)
            - Pricing recommendation balances profit and market competition
            """
        )

st.markdown("---")

# =========================================================
# PRICING STRATEGY COMPARISON
# =========================================================

st.subheader("🎯 Advanced Pricing Strategy Comparison")

# Strategy configuration with buttons
st.markdown("**Compare pricing strategies:**")
button_col1, button_col2, button_col3 = st.columns([1, 1, 2])

with button_col1:
    if st.button(
        "📊 Compare All Strategies",
        key="compare_strategies",
        help="Run comprehensive strategy comparison",
        use_container_width=True
    ):
        st.session_state.run_strategy_comparison = True

with button_col2:
    if st.button(
        "🔄 Reset",
        key="reset_strategies",
        help="Clear strategy comparison results",
        use_container_width=True
    ):
        st.session_state.run_strategy_comparison = False

with button_col3:
    st.info("💡 Click 'Compare All Strategies' to see pricing analysis")

# Run strategy comparison if button was clicked
if st.session_state.get('run_strategy_comparison', False):
    try:
        strategy_result = compare_pricing_strategies(
            product_row=latest,
            compare_bundle_with_product_id=10,  # Always enable bundle comparison
            bundle_discount_pct=0.10,
            bundle_demand_boost_pct=0.20
        )
        
        # Display best strategy recommendation
        st.divider()
        
        best_strat = strategy_result['best_strategy_name']
        best_price = strategy_result['recommended_price']
        best_demand = strategy_result['expected_demand']
        best_profit = strategy_result['expected_profit']
        best_margin = strategy_result['profit_margin']
        
        # Highlight the best strategy
        st.success(f"✅ **BEST STRATEGY: {best_strat}**")
        
        # Display metrics for best strategy
        col_bs1, col_bs2, col_bs3, col_bs4 = st.columns(4)
        
        with col_bs1:
            st.metric(
                "Recommended Price",
                f"₹{best_price * USD_TO_INR:.2f}"
            )
        
        with col_bs2:
            st.metric(
                "Expected Demand",
                f"{best_demand:.0f} units"
            )
        
        with col_bs3:
            st.metric(
                "Expected Profit",
                f"₹{best_profit * USD_TO_INR:,.0f}"
            )
        
        with col_bs4:
            st.metric(
                "Profit Margin",
                f"{best_margin:.1f}%"
            )
        
        # Display explanation
        st.markdown(f"**Why this strategy?** {strategy_result['explanation']}")
        
        # Create comparison table
        st.divider()
        st.subheader("📊 Strategy Comparison Table")
        
        # Build comparison dataframe
        comparison_data = []
        
        for strategy_name, strategy_data in strategy_result['all_strategies'].items():
            if strategy_name == 'base':
                display_name = "BASE (ML Optimized)"
                discount_info = "—"
            elif strategy_name == 'discount':
                discount_rate = strategy_data.get('discount_rate', 0)
                display_name = f"DISCOUNT ({int(discount_rate*100)}% off)"
                discount_info = f"{int(discount_rate*100)}%"
            elif strategy_name == 'bundle':
                display_name = "BUNDLE"
                discount_info = "Bundle"
            else:
                display_name = strategy_name.upper()
                discount_info = "—"
            
            is_best = "✅ BEST" if strategy_name == best_strat.lower() else ""
            
            comparison_data.append({
                "Strategy": f"{display_name} {is_best}",
                "Price (₹)": f"{strategy_data['price'] * USD_TO_INR:.2f}",
                "Demand (units)": f"{strategy_data['demand']:.0f}",
                "Profit (₹)": f"{strategy_data['profit'] * USD_TO_INR:,.0f}",
                "Margin (%)": f"{strategy_data['margin']:.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Detailed insights
        st.divider()
        st.subheader("💡 Strategy Insights")
        
        # Calculate insights
        base_strategy = strategy_result['all_strategies'].get('base', {})
        discount_strategy = strategy_result['all_strategies'].get('discount', {})
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.write("**Profit Comparison vs Base:**")
            if discount_strategy:
                discount_vs_base = discount_strategy['profit'] - base_strategy.get('profit', 0)
                discount_pct = (discount_vs_base / base_strategy.get('profit', 1) * 100) if base_strategy.get('profit', 0) > 0 else 0
                
                if discount_vs_base > 0:
                    st.success(f"Discount: +₹{discount_vs_base * USD_TO_INR:,.0f} ({discount_pct:+.1f}%)")
                else:
                    st.warning(f"Discount: ₹{discount_vs_base * USD_TO_INR:,.0f} ({discount_pct:+.1f}%)")
        
        with insights_col2:
            st.write("**Volume Impact:**")
            if discount_strategy:
                discount_demand = discount_strategy.get('demand', 0)
                base_demand = base_strategy.get('demand', 0)
                volume_increase = discount_demand - base_demand
                volume_pct = (volume_increase / base_demand * 100) if base_demand > 0 else 0
                
                st.info(f"Discount increases volume by {volume_increase:.0f} units ({volume_pct:+.1f}%)")
    
    except Exception as e:
        st.error(f"Strategy comparison failed: {str(e)}")
        import traceback
        st.write(traceback.format_exc())

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

# ========================================================
# VALIDATION BEFORE PREDICTION
# Assert features match training exactly
# ========================================================
is_valid, validation_msg = validate_prediction_features(
    sim_inputs,
    raise_error=True,
    verbose=False,  # Suppress for simulation (many predictions)
    exclude_columns=[]  # No metadata columns here
)

forecasted_demand = model.predict(sim_inputs)

# Apply inventory constraint to profit calculation
# Actual sales cannot exceed available inventory
inventory_level = latest.get("inventory_level", float('inf'))
actual_sales = np.minimum(forecasted_demand, inventory_level)

# Profit uses actual sales (constrained by inventory)
predicted_profit = (price_range - cost_val) * actual_sales

# Store results in DataFrame
sim_results = pd.DataFrame({
    "price": price_range,
    "forecasted_demand": forecasted_demand,
    "actual_sales": actual_sales,
    "predicted_profit": predicted_profit,
    "inventory_constraint": inventory_level,
})

# -------------------------
# Interactive Price Slider
# -------------------------
st.subheader("Interactive Price Adjustment")

# --------- Inventory Scenario Selector ---------
st.markdown("**📦 Inventory Scenario Analysis:**")
col_scenario1, col_scenario2 = st.columns([2, 1])

with col_scenario1:
    inventory_scenario = st.selectbox(
        "Select inventory scenario:",
        options=["Normal Inventory", "Low Inventory (Scarce)", "High Inventory (Overstock)"],
        help="Analyze how optimal pricing changes under different inventory levels"
    )

# Map scenario to inventory ratio
scenario_to_ratio = {
    "Normal Inventory": inventory_level / max_inventory,
    "Low Inventory (Scarce)": 0.1,
    "High Inventory (Overstock)": 0.9,
}
scenario_inventory_ratio = scenario_to_ratio[inventory_scenario]
scenario_inventory_level = scenario_inventory_ratio * max_inventory

# Display scenario inventory info
with col_scenario2:
    st.metric(
        "Scenario Inventory",
        f"{scenario_inventory_ratio:.0%}",
        f"({scenario_inventory_level:.0f} units)"
    )

# Recalculate simulation with scenario inventory level
scenario_actual_sales = np.minimum(forecasted_demand, scenario_inventory_level)
scenario_predicted_profit = (price_range - cost_val) * scenario_actual_sales

# Store scenario results
sim_results_scenario = pd.DataFrame({
    "price": price_range,
    "forecasted_demand": forecasted_demand,
    "actual_sales": scenario_actual_sales,
    "predicted_profit": scenario_predicted_profit,
})

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

# ========================================================
# VALIDATION BEFORE PREDICTION
# Assert features match training exactly
# ========================================================
is_valid, validation_msg = validate_prediction_features(
    selected_input_df,
    raise_error=True,
    verbose=False,
    exclude_columns=[]  # No metadata columns here
)

selected_demand = model.predict(selected_input_df)[0]

# Apply inventory constraint to selected price profit
selected_actual_sales = min(selected_demand, inventory_level)
selected_profit = (selected_price_usd - cost_val) * selected_actual_sales

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
        "Actual Profit (Inventory-Constrained)",
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
    
    # Add base profit curve (current inventory scenario)
    fig_profit.add_trace(go.Scatter(
        x=sim_results["price"] * USD_TO_INR,
        y=sim_results["predicted_profit"] * USD_TO_INR,
        mode='lines',
        name='Current Inventory',
        line=dict(color='#ff7f0e', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.15)',
        hovertemplate='<b>Price: ₹%{x:.2f}</b><br>Profit: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Add scenario profit curve (only if different from current)
    if inventory_scenario != "Normal Inventory":
        fig_profit.add_trace(go.Scatter(
            x=sim_results_scenario["price"] * USD_TO_INR,
            y=sim_results_scenario["predicted_profit"] * USD_TO_INR,
            mode='lines',
            name=inventory_scenario,
            line=dict(color='#7cb342', width=2, dash='dash'),
            fill=None,
            hovertemplate='<b>Price: ₹%{x:.2f}</b><br>Profit (Scenario): ₹%{y:,.0f}<extra></extra>'
        ))
        
        # Highlight the optimal profit point for the scenario
        scenario_optimal_idx = np.argmax(sim_results_scenario["predicted_profit"])
        scenario_optimal_price = sim_results_scenario["price"].iloc[scenario_optimal_idx]
        scenario_optimal_profit = sim_results_scenario["predicted_profit"].iloc[scenario_optimal_idx]
        
        fig_profit.add_trace(go.Scatter(
            x=[scenario_optimal_price * USD_TO_INR],
            y=[scenario_optimal_profit * USD_TO_INR],
            mode='markers',
            name='Scenario Optimal',
            marker=dict(size=12, color='#7cb342', symbol='diamond', line=dict(color='#558b2f', width=2)),
            hovertemplate='<b>Scenario Optimal: ₹%{x:.2f}</b><br>Profit: ₹%{y:,.0f}<extra></extra>'
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

# Display inventory constraint information
st.markdown("---")

# Show scenario comparison if analyzing different inventory levels
if inventory_scenario != "Normal Inventory":
    st.subheader("📊 Inventory Scenario Comparison")
    
    # Calculate optimal prices for current and scenario
    current_optimal_idx = np.argmax(sim_results["predicted_profit"])
    current_optimal_price = sim_results["price"].iloc[current_optimal_idx]
    current_max_profit = sim_results["predicted_profit"].iloc[current_optimal_idx]
    
    scenario_optimal_idx = np.argmax(sim_results_scenario["predicted_profit"])
    scenario_optimal_price = sim_results_scenario["price"].iloc[scenario_optimal_idx]
    scenario_max_profit = sim_results_scenario["predicted_profit"].iloc[scenario_optimal_idx]
    
    # Display comparison metrics
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    
    with col_comp1:
        st.metric(
            "Current Scenario Price",
            f"₹{current_optimal_price * USD_TO_INR:.2f}",
            f"Profit: ₹{current_max_profit * USD_TO_INR:,.0f}"
        )
    
    with col_comp2:
        st.metric(
            f"{inventory_scenario} Price",
            f"₹{scenario_optimal_price * USD_TO_INR:.2f}",
            f"Profit: ₹{scenario_max_profit * USD_TO_INR:,.0f}"
        )
    
    with col_comp3:
        price_diff_pct = ((scenario_optimal_price - current_optimal_price) / current_optimal_price * 100)
        profit_diff_pct = ((scenario_max_profit - current_max_profit) / current_max_profit * 100)
        
        st.metric(
            "Price Change",
            f"{price_diff_pct:+.1f}%",
            f"Profit Impact: {profit_diff_pct:+.1f}%"
        )
    
    # Add insights
    if inventory_scenario == "Low Inventory (Scarce)":
        st.success("""
        **💡 Insights for Low Inventory:**
        - Scarcity allows for higher pricing
        - Optimal price increased to maximize per-unit revenue
        - Lower volume but higher margins compensate
        - This pricing also triggers the 8% scarcity markup in recommendations
        """)
    elif inventory_scenario == "High Inventory (Overstock)":
        st.warning("""
        **� Insights for High Inventory:**
        - Excess stock requires aggressive discounting
        - Lower prices accelerate turnover and reduce holding costs
        - Higher volume partially offsets lower margins
        - This pricing aligns with the 8% clearance discount in recommendations
        """)

st.info(f"""
**�📦 Inventory Constraint Applied:**
- Current Inventory Level: {inventory_level:.0f} units ({inventory_level/max_inventory:.0%} of capacity)
- Profit calculations are constrained by available inventory
- Actual sales = min(forecasted_demand, inventory_level)
- Demand shows predicted demand without constraint
- Profit shows revenue achievable with current inventory
""")

st.markdown("---")

# =========================================================
# PRICE CALCULATION BREAKDOWN
# =========================================================

with st.expander("💰 Price Calculation & Optimization Formula"):
    st.markdown("### Price Optimization Formula")
    
    st.markdown("""
    The optimal price is computed using a **profit maximization approach** with Monte Carlo simulation.
    The algorithm tests multiple price points and selects the one that maximizes expected profit.
    """)
    
    # Display the profit formula
    st.latex(r"""
    \text{Profit}(P) = (P - C) \cdot Q(P) - \text{Penalties}
    """)
    
    st.markdown("""
    **Where:**
    - **P**: Price (what we're optimizing)
    - **C**: Unit cost (fixed input)
    - **Q(P)**: Demand as a function of price (from demand model)
    - **Penalties**: Various adjustments for competition, volatility, etc.
    
    **Optimization Process:**
    1. Generate 100+ candidate prices (typically ±20% around current price)
    2. For each candidate price, run Monte Carlo simulations (500+ iterations)
    3. Simulate demand including stochastic noise/uncertainty
    4. Calculate profit = (price - cost) × simulated_demand
    5. Apply competitive penalties if price > competitor price
    6. Apply inventory adjustments based on stock levels
    7. Select price with maximum expected profit
    """)
    
    st.markdown("### Step-by-Step Price Optimization Calculation")
    
    # Use actual values from the product
    current_price = latest.get("price", 0.0)
    cost = latest.get("cost", 0.0)
    competitor_price = latest.get("competitor_price", 0.0)
    inventory_level = latest.get("inventory_level", 250)
    max_inventory = latest.get("max_inventory", 500)
    current_demand = latest.get("historical_demand", 100)
    
    st.markdown(f"**For Product:** {selected_product}")
    
    # Display input parameters
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("Current Price", f"₹{current_price * USD_TO_INR:.2f}")
    col_p2.metric("Unit Cost", f"₹{cost * USD_TO_INR:.2f}")
    col_p3.metric("Competitor Price", f"₹{competitor_price * USD_TO_INR:.2f}")
    col_p4.metric("Current Demand", f"{current_demand:.0f} units")
    
    st.markdown("---")
    st.markdown("### Profit Calculation at Different Price Points")
    
    # Calculate profit at several key price points
    price_points = [
        ("Current Price", current_price),
        ("10% Discount", current_price * 0.9),
        ("5% Discount", current_price * 0.95),
        ("Competitor Price", competitor_price),
        ("5% Premium", current_price * 1.05),
        ("10% Premium", current_price * 1.10),
    ]
    
    profit_analysis = []
    
    for price_label, test_price in price_points:
        # Prepare features for demand prediction
        test_features = {
            name: (test_price if name == "price" else 
                   test_price - competitor_price if name == "price_gap" else 
                   base_features.get(name, 0.0))
            for name in model_features
        }
        
        test_df = pd.DataFrame([test_features])
        
        try:
            # Predict demand at this price
            predicted_demand = model.predict(test_df)[0]
            
            # Calculate profit
            margin = test_price - cost
            margin_pct = (margin / test_price * 100) if test_price > 0 else 0
            profit = margin * predicted_demand
            
            profit_analysis.append({
                'Price Point': price_label,
                'Price (₹)': f"{test_price * USD_TO_INR:.2f}",
                'Unit Margin (₹)': f"{margin * USD_TO_INR:.2f}",
                'Margin %': f"{margin_pct:.1f}%",
                'Predicted Demand': f"{predicted_demand:.0f} units",
                'Total Profit (₹)': f"{profit * USD_TO_INR:,.0f}"
            })
        except:
            pass
    
    # Display profit analysis table
    profit_df = pd.DataFrame(profit_analysis)
    st.markdown("**Profit at Different Price Points:**")
    st.dataframe(profit_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Competitive Pricing Analysis")
    
    # Calculate price gaps and competitive metrics
    price_gap = current_price - competitor_price
    is_premium = price_gap > 0
    gap_pct = (price_gap / competitor_price * 100) if competitor_price > 0 else 0
    
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    
    with col_comp1:
        st.metric(
            "Price Gap (vs Competitor)",
            f"₹{price_gap * USD_TO_INR:.2f}",
            f"{gap_pct:+.1f}%"
        )
    
    with col_comp2:
        position = "Premium" if is_premium else "Discount"
        st.metric(
            "Competitive Position",
            position,
            f"{abs(gap_pct):.1f}% {'above' if is_premium else 'below'} market"
        )
    
    with col_comp3:
        # Calculate elasticity effect
        elasticity_factor = 1.3 if "mid" in selected_product.lower() else 0.7 if "Apple" in selected_product or "Sony" in selected_product else 1.0
        st.metric(
            "Price Elasticity Factor",
            f"{elasticity_factor:.1f}",
            "High sensitivity" if elasticity_factor > 1.0 else "Low sensitivity"
        )
    
    st.markdown("---")
    st.markdown("### Inventory Impact on Price")
    
    inventory_ratio = inventory_level / max_inventory
    
    # Determine inventory adjustment
    if inventory_ratio < 0.2:
        inventory_adjustment = 0.08  # 8% markup for scarcity
        inventory_status = "🚨 Low Stock - Increase Price"
        adjustment_reason = "Scarcity allows premium pricing to maximize revenue"
    elif inventory_ratio > 0.8:
        inventory_adjustment = -0.08  # 8% discount for overstock
        inventory_status = "📈 Overstock - Decrease Price"
        adjustment_reason = "Excess inventory requires aggressive discounting for faster turnover"
    else:
        inventory_adjustment = 0.0
        inventory_status = "✓ Balanced Inventory - Standard Pricing"
        adjustment_reason = "Inventory at healthy level, no special adjustments needed"
    
    col_inv1, col_inv2, col_inv3 = st.columns(3)
    
    with col_inv1:
        st.metric(
            "Inventory Level",
            f"{inventory_ratio:.0%}",
            f"{inventory_level:.0f} / {max_inventory:.0f} units"
        )
    
    with col_inv2:
        st.metric(
            "Inventory Status",
            inventory_status
        )
    
    with col_inv3:
        st.metric(
            "Price Adjustment",
            f"{inventory_adjustment:+.1%}",
            adjustment_reason
        )
    
    st.markdown("---")
    st.markdown("### Margin vs Volume Trade-off")
    
    # Show margin and volume at different prices
    st.markdown("""
    **Key Insight:** Higher prices increase per-unit margin but reduce demand volume.
    The optimizer finds the sweet spot that maximizes total profit.
    """)
    
    # Create interactive visualization
    test_prices = np.linspace(current_price * 0.8, current_price * 1.2, 30)
    margin_analysis = []
    
    for test_p in test_prices:
        test_feat = {
            name: (test_p if name == "price" else 
                   test_p - competitor_price if name == "price_gap" else 
                   base_features.get(name, 0.0))
            for name in model_features
        }
        
        try:
            test_d = model.predict(pd.DataFrame([test_feat]))[0]
            test_margin = test_p - cost
            margin_analysis.append({
                'price': test_p,
                'margin': test_margin,
                'demand': test_d,
                'profit': test_margin * test_d
            })
        except:
            pass
    
    if margin_analysis:
        margin_df = pd.DataFrame(margin_analysis)
        
        # Create Plotly figure with dual y-axes
        fig_margin = go.Figure()
        
        # Add margin curve (left y-axis)
        fig_margin.add_trace(go.Scatter(
            x=margin_df['price'] * USD_TO_INR,
            y=margin_df['margin'] * USD_TO_INR,
            mode='lines',
            name='Unit Margin',
            line=dict(color='#2ecc71', width=3),
            yaxis='y1',
            hovertemplate='<b>Price: ₹%{x:.2f}</b><br>Unit Margin: ₹%{y:.2f}<extra></extra>'
        ))
        
        # Add demand curve (right y-axis)
        fig_margin.add_trace(go.Scatter(
            x=margin_df['price'] * USD_TO_INR,
            y=margin_df['demand'],
            mode='lines',
            name='Demand',
            line=dict(color='#3498db', width=3),
            yaxis='y2',
            hovertemplate='<b>Price: ₹%{x:.2f}</b><br>Demand: %{y:.0f} units<extra></extra>'
        ))
        
        # Add current price line
        fig_margin.add_vline(
            x=current_price * USD_TO_INR,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            annotation_text=f"Current: ₹{current_price * USD_TO_INR:.2f}",
            annotation_position="top left",
            annotation_font_size=10
        )
        
        # Update layout with dual y-axes
        fig_margin.update_layout(
            title='Margin vs Volume Trade-off',
            xaxis_title='Price (₹)',
            yaxis=dict(
                title='<b>Unit Margin (₹)</b>',
                title_font=dict(color='#2ecc71'),
                tickfont=dict(color='#2ecc71')
            ),
            yaxis2=dict(
                title='<b>Predicted Demand (units)</b>',
                title_font=dict(color='#3498db'),
                tickfont=dict(color='#3498db'),
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=500,
            font=dict(size=11),
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_margin, use_container_width=True)
        
        # PIE CHART: PRICE OPTIMIZATION REWARD FUNCTION WEIGHTAGES
        st.markdown("---")
        st.markdown("#### 🎯 Price Optimization Reward Function Weightages")
        
        st.markdown("""
        The optimizer uses a **reward function** that balances multiple objectives:
        - Maximize profit while considering risk, competition, and customer trust
        - Each component has a specific weightage in the final optimization score
        """)
        
        # Calculate reward function components
        # Simulated values for current price
        test_feat_current = {
            name: (current_price if name == "price" else 
                   current_price - competitor_price if name == "price_gap" else 
                   base_features.get(name, 0.0))
            for name in model_features
        }
        
        try:
            demand_current = model.predict(pd.DataFrame([test_feat_current]))[0]
            base_profit = (current_price - cost) * demand_current
            
            # Define reward components with weightages
            volatility_penalty = 0.1 * (demand_current * 0.15)  # 10% volatility penalty
            competitive_penalty = 0.25 * max(0, current_price - competitor_price)  # 25% competitive
            uncertainty_penalty = 0.1 * (demand_current * 0.15)  # 10% uncertainty
            trust_penalty = 0.15 * abs(current_price - latest.get("price", current_price)) / current_price  # 15% trust
            anchor_penalty = 0.15 * abs(current_price - competitor_price) / current_price  # 15% anchor
            inventory_bonus = 0.2 * (inventory_level / max_inventory) * base_profit  # 20% inventory
            turnover_bonus = 0.1 * (demand_current / 100)  # 10% turnover
            
            reward_components = {
                'Base Profit': max(0.1, base_profit),  # Avoid zero for visualization
                'Volatility Penalty': volatility_penalty,
                'Competitive Penalty': competitive_penalty,
                'Uncertainty Penalty': uncertainty_penalty,
                'Trust Penalty': trust_penalty,
                'Anchor Penalty': anchor_penalty,
                'Inventory Bonus': inventory_bonus,
                'Turnover Bonus': turnover_bonus
            }
            
            # Create two pie charts: Positive and Negative
            col_reward1, col_reward2 = st.columns(2)
            
            # POSITIVE REWARDS
            with col_reward1:
                st.markdown("**✅ Positive Reward Factors:**")
                
                positive_rewards = {
                    'Base Profit (Primary)': reward_components['Base Profit'],
                    'Inventory Bonus (+20%)': reward_components['Inventory Bonus'],
                    'Turnover Bonus (+10%)': reward_components['Turnover Bonus']
                }
                
                fig_pie_reward_pos = go.Figure(data=[go.Pie(
                    labels=list(positive_rewards.keys()),
                    values=list(positive_rewards.values()),
                    marker=dict(
                        colors=['#2ecc71', '#27ae60', '#229954'],
                        line=dict(color='black', width=2)
                    ),
                    textinfo='label+percent',
                    textposition='inside',
                    texttemplate='<b>%{label}</b><br>%{percent}',
                    hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<br>Percentage: %{percent}<extra></extra>',
                    hole=0.3
                )])
                
                fig_pie_reward_pos.update_layout(
                    title='Positive Reward Components',
                    height=450,
                    font=dict(size=9),
                    showlegend=True,
                    legend=dict(x=0.0, y=-0.15, font=dict(size=9))
                )
                
                st.plotly_chart(fig_pie_reward_pos, use_container_width=True)
            
            # NEGATIVE PENALTIES
            with col_reward2:
                st.markdown("**❌ Negative Penalty Factors:**")
                
                negative_penalties = {
                    'Competitive Penalty (-25%)': reward_components['Competitive Penalty'],
                    'Trust Penalty (-15%)': reward_components['Trust Penalty'],
                    'Anchor Penalty (-15%)': reward_components['Anchor Penalty'],
                    'Volatility Penalty (-10%)': reward_components['Volatility Penalty'],
                    'Uncertainty Penalty (-10%)': reward_components['Uncertainty Penalty']
                }
                
                fig_pie_reward_neg = go.Figure(data=[go.Pie(
                    labels=list(negative_penalties.keys()),
                    values=list(negative_penalties.values()),
                    marker=dict(
                        colors=['#e74c3c', '#c0392b', '#a93226', '#922b21', '#78281f'],
                        line=dict(color='black', width=2)
                    ),
                    textinfo='label+percent',
                    textposition='inside',
                    texttemplate='<b>%{label}</b><br>%{percent}',
                    hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<br>Percentage: %{percent}<extra></extra>',
                    hole=0.3
                )])
                
                fig_pie_reward_neg.update_layout(
                    title='Negative Penalty Components',
                    height=450,
                    font=dict(size=9),
                    showlegend=True,
                    legend=dict(x=0.0, y=-0.15, font=dict(size=9))
                )
                
                st.plotly_chart(fig_pie_reward_neg, use_container_width=True)
            
            # OVERALL WEIGHTAGE SUMMARY TABLE
            st.markdown("---")
            st.markdown("#### 📊 Reward Function Component Weightages")
            
            weightage_summary = pd.DataFrame({
                'Component': [
                    'Base Profit (1.0x)',
                    'Inventory Bonus',
                    'Turnover Bonus',
                    'Competitive Penalty',
                    'Trust Penalty',
                    'Anchor Penalty',
                    'Volatility Penalty',
                    'Uncertainty Penalty'
                ],
                'Weightage': [
                    '100% (Primary)',
                    '+20%',
                    '+10%',
                    '-25%',
                    '-15%',
                    '-15%',
                    '-10%',
                    '-10%'
                ],
                'Direction': [
                    '⬆️ Increase',
                    '⬆️ Increase',
                    '⬆️ Increase',
                    '⬇️ Decrease',
                    '⬇️ Decrease',
                    '⬇️ Decrease',
                    '⬇️ Decrease',
                    '⬇️ Decrease'
                ],
                'Purpose': [
                    'Maximize revenue',
                    'Incentivize inventory clearance',
                    'Encourage volume sales',
                    'Avoid competitor pricing wars',
                    'Minimize price volatility',
                    'Stay anchored to market',
                    'Avoid demand uncertainty',
                    'Reduce risk'
                ]
            })
            
            st.dataframe(weightage_summary, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.warning(f"Could not calculate reward components: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Monte Carlo Simulation for Profit Uncertainty")
    
    st.markdown("""
    The optimizer uses Monte Carlo simulation to account for demand uncertainty:
    - Generates 500+ random demand scenarios based on predicted demand ± uncertainty
    - Calculates profit for each scenario
    - Uses mean profit as the optimization criterion
    - This reduces the risk of extreme outcomes
    """)
    
    # Simulate profit distribution at current and optimal prices
    if rec_price is not None:
        st.markdown(f"**Comparing Profit Distributions:**")
        
        # Simulate at current price
        current_feat = {
            name: (current_price if name == "price" else 
                   current_price - competitor_price if name == "price_gap" else 
                   base_features.get(name, 0.0))
            for name in model_features
        }
        current_base_demand = model.predict(pd.DataFrame([current_feat]))[0]
        current_profits = (current_price - cost) * np.random.normal(current_base_demand, current_base_demand * 0.15, 5000)
        
        # Simulate at optimal price
        optimal_feat = {
            name: (rec_price if name == "price" else 
                   rec_price - competitor_price if name == "price_gap" else 
                   base_features.get(name, 0.0))
            for name in model_features
        }
        optimal_base_demand = model.predict(pd.DataFrame([optimal_feat]))[0]
        optimal_profits = (rec_price - cost) * np.random.normal(optimal_base_demand, optimal_base_demand * 0.15, 5000)
        
        # Create comparison boxplot
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(
            y=current_profits * USD_TO_INR,
            name=f'Current Price<br>₹{current_price * USD_TO_INR:.2f}',
            boxmean='sd',
            marker_color='#ff9800'
        ))
        
        fig_box.add_trace(go.Box(
            y=optimal_profits * USD_TO_INR,
            name=f'Optimal Price<br>₹{rec_price * USD_TO_INR:.2f}',
            boxmean='sd',
            marker_color='#2ecc71'
        ))
        
        fig_box.update_layout(
            title='Profit Distribution: Current vs Optimal Price',
            yaxis_title='Profit (₹)',
            template='plotly_white',
            height=450,
            font=dict(size=11),
            hovermode='y unified'
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Display statistics
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.write("**Current Price Profit Statistics:**")
            st.metric("Mean Profit", f"₹{np.mean(current_profits) * USD_TO_INR:,.0f}")
            st.metric("Std Dev", f"₹{np.std(current_profits) * USD_TO_INR:,.0f}")
            st.metric("5th Percentile", f"₹{np.percentile(current_profits, 5) * USD_TO_INR:,.0f}")
        
        with col_stat2:
            st.write("**Optimal Price Profit Statistics:**")
            st.metric("Mean Profit", f"₹{np.mean(optimal_profits) * USD_TO_INR:,.0f}")
            st.metric("Std Dev", f"₹{np.std(optimal_profits) * USD_TO_INR:,.0f}")
            st.metric("5th Percentile", f"₹{np.percentile(optimal_profits, 5) * USD_TO_INR:,.0f}")

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
    
    # ========================================================
    # VALIDATION BEFORE PREDICTION
    # Assert features match training exactly
    # ========================================================
    is_valid, validation_msg = validate_prediction_features(
        optimal_price_input,
        raise_error=True,
        verbose=False,
        exclude_columns=[]  # No metadata columns here
    )
    
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
