#!/usr/bin/env python3
"""
📉 Price Elasticity Analysis Script
Calculates price elasticity of demand and provides visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

def calculate_elasticity(df, price_col='price', quantity_col='quantity'):
    """Calculate price elasticity of demand"""
    if price_col not in df.columns or quantity_col not in df.columns:
        st.error(f"Required columns {price_col} and {quantity_col} not found")
        return None
    
    # Remove any rows with missing values
    clean_df = df[[price_col, quantity_col]].dropna()
    
    if len(clean_df) < 2:
        st.error("Insufficient data for elasticity calculation")
        return None
    
    # Calculate elasticity using log-log regression
    log_price = np.log(clean_df[price_col])
    log_quantity = np.log(clean_df[quantity_col])
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(log_price.values.reshape(-1, 1), log_quantity.values)
    
    # Elasticity is the coefficient
    elasticity = model.coef_[0]
    
    # Calculate R-squared
    r2 = model.score(log_price.values.reshape(-1, 1), log_quantity.values)
    
    return {
        'elasticity': elasticity,
        'r2': r2,
        'model': model,
        'log_price': log_price,
        'log_quantity': log_quantity,
        'clean_df': clean_df
    }

def create_elasticity_visualizations(results):
    """Create visualizations for elasticity analysis"""
    if not results:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Log-log plot
        fig = px.scatter(
            x=results['log_price'], 
            y=results['log_quantity'],
            title="Log-Log Price vs Quantity",
            labels={'x': 'Log(Price)', 'y': 'Log(Quantity)'}
        )
        
        # Add regression line
        log_price_range = np.linspace(results['log_price'].min(), results['log_price'].max(), 100)
        predicted_log_quantity = results['model'].predict(log_price_range.reshape(-1, 1))
        
        fig.add_trace(go.Scatter(
            x=log_price_range,
            y=predicted_log_quantity,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', dash='dash')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Original price vs quantity
        fig = px.scatter(
            x=results['clean_df']['price'], 
            y=results['clean_df']['quantity'],
            title="Price vs Quantity (Original Scale)",
            labels={'x': 'Price', 'y': 'Quantity'}
        )
        st.plotly_chart(fig, use_container_width=True)

def elasticity_interpretation(elasticity):
    """Provide interpretation of elasticity value"""
    abs_elasticity = abs(elasticity)
    
    if abs_elasticity < 0.1:
        interpretation = "Perfectly Inelastic"
        color = "green"
    elif abs_elasticity < 0.5:
        interpretation = "Inelastic"
        color = "blue"
    elif abs_elasticity < 1.0:
        interpretation = "Relatively Inelastic"
        color = "orange"
    elif abs_elasticity < 2.0:
        interpretation = "Elastic"
        color = "red"
    else:
        interpretation = "Highly Elastic"
        color = "darkred"
    
    return interpretation, color

def main():
    st.set_page_config(
        page_title="Price Elasticity Analysis",
        page_icon="📉",
        layout="wide"
    )
    
    st.title("📉 Price Elasticity of Demand Analysis")
    st.markdown("---")
    
    # Load data
    st.header("📊 Data Loading")
    
    # Try to load from different possible locations
    data_paths = [
        "DATASET/raw/wfp_food_prices_nga.csv",
        "DATASET/raw/wfp_food_prices.csv",
        "data/processed/merged_input_dataset.csv"
    ]
    
    df = None
    for path in data_paths:
        try:
            df = pd.read_csv(path)
            st.success(f"✅ Loaded data from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        st.warning("⚠️ No data files found. Creating dummy data for demonstration.")
        # Create dummy data
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic price and quantity data with some elasticity
        base_price = 100
        price_variation = np.random.normal(0, 20, n_samples)
        prices = base_price + price_variation
        
        # Quantity inversely related to price (elasticity)
        elasticity_factor = -0.8  # Negative elasticity
        base_quantity = 500
        quantity_variation = elasticity_factor * (prices - base_price) / base_price * base_quantity
        quantities = base_quantity + quantity_variation + np.random.normal(0, 50, n_samples)
        
        df = pd.DataFrame({
            'price': prices,
            'quantity': quantities,
            'date': pd.date_range('2022-01-01', periods=n_samples, freq='D')
        })
        
        st.info("Dummy data created with realistic elasticity patterns")
    
    # Display data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column selection
    st.subheader("🔧 Column Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        price_col = st.selectbox("Select Price Column:", df.columns, index=0)
    with col2:
        quantity_col = st.selectbox("Select Quantity Column:", df.columns, index=1)
    
    # Calculate elasticity
    st.header("📈 Elasticity Calculation")
    
    if st.button("Calculate Price Elasticity", type="primary"):
        with st.spinner("Calculating elasticity..."):
            results = calculate_elasticity(df, price_col, quantity_col)
            
            if results:
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Price Elasticity", f"{results['elasticity']:.4f}")
                
                with col2:
                    st.metric("R² Score", f"{results['r2']:.4f}")
                
                with col3:
                    interpretation, color = elasticity_interpretation(results['elasticity'])
                    st.markdown(f"**Interpretation:** <span style='color:{color}'>{interpretation}</span>", 
                              unsafe_allow_html=True)
                
                # Detailed interpretation
                st.subheader("📝 Detailed Analysis")
                
                if results['elasticity'] < 0:
                    st.info("✅ **Negative elasticity** indicates normal demand behavior (quantity decreases as price increases)")
                else:
                    st.warning("⚠️ **Positive elasticity** indicates unusual demand behavior (quantity increases as price increases)")
                
                if abs(results['elasticity']) < 1:
                    st.info("📊 **Inelastic demand** - Price changes have relatively small impact on quantity demanded")
                else:
                    st.info("📊 **Elastic demand** - Price changes have significant impact on quantity demanded")
                
                # Create visualizations
                st.subheader("📊 Visualizations")
                create_elasticity_visualizations(results)
                
                # Statistical significance test
                st.subheader("🔬 Statistical Analysis")
                
                # Correlation test
                correlation, p_value = stats.pearsonr(results['log_price'], results['log_quantity'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Correlation Coefficient", f"{correlation:.4f}")
                with col2:
                    st.metric("P-value", f"{p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("✅ Statistically significant relationship (p < 0.05)")
                else:
                    st.warning("⚠️ Relationship not statistically significant (p ≥ 0.05)")
                
                # Business recommendations
                st.subheader("💡 Business Recommendations")
                
                if abs(results['elasticity']) < 1:
                    st.info("""
                    **Inelastic Demand Recommendations:**
                    - Consider price increases to boost revenue
                    - Focus on cost reduction rather than volume
                    - Price changes will have limited impact on sales volume
                    """)
                else:
                    st.info("""
                    **Elastic Demand Recommendations:**
                    - Be cautious with price increases
                    - Focus on volume strategies
                    - Consider promotional pricing to increase sales
                    - Monitor competitor pricing closely
                    """)
    
    # Additional analysis options
    st.header("🔍 Additional Analysis")
    
    if st.checkbox("Show Price Distribution Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=price_col, title="Price Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x=quantity_col, title="Quantity Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Show Time Series Analysis") and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df_sorted = df.sort_values('date')
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Over Time', 'Quantity Over Time'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=df_sorted['date'], y=df_sorted[price_col], name='Price'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_sorted['date'], y=df_sorted[quantity_col], name='Quantity'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
