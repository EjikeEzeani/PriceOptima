#!/usr/bin/env python3
"""
PriceOptima Streamlit Visualization Dashboard
A comprehensive dashboard for analyzing food price data, training ML models, and generating insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PriceOptima Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data functions
@st.cache_data
def load_food_prices_data():
    """Load the main food prices dataset."""
    try:
        df = pd.read_csv("DATASET/raw/wfp_food_prices_nga.csv")
        # Clean the data
        df = df[df['date'] != '#date']  # Remove header rows
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['usdprice'] = pd.to_numeric(df['usdprice'], errors='coerce')
        return df.dropna(subset=['date', 'price'])
    except Exception as e:
        st.error(f"Error loading food prices data: {e}")
        return None

@st.cache_data
def load_simple_food_prices():
    """Load the simplified food prices dataset."""
    try:
        df = pd.read_csv("DATASET/raw/wfp_food_prices.csv")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        return df.dropna(subset=['date', 'price'])
    except Exception as e:
        st.error(f"Error loading simple food prices data: {e}")
        return None

@st.cache_data
def load_rainfall_data():
    """Load rainfall data."""
    try:
        df = pd.read_csv("DATASET/raw/nga-rainfall-subnat-full.csv")
        return df
    except Exception as e:
        st.warning(f"Could not load rainfall data: {e}")
        return None

def load_trained_models():
    """Load available trained models."""
    models = {}
    model_dir = "models"
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.joblib'):
                model_name = filename.replace('.joblib', '')
                try:
                    model_path = os.path.join(model_dir, filename)
                    models[model_name] = joblib.load(model_path)
                except Exception as e:
                    st.warning(f"Could not load model {filename}: {e}")
    return models

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 PriceOptima Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Food Price Analysis & Machine Learning Platform")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "📈 Data Explorer", "🤖 Machine Learning", "📊 Insights", "🔮 Forecasting"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        food_data = load_food_prices_data()
        simple_data = load_simple_food_prices()
        rainfall_data = load_rainfall_data()
        models = load_trained_models()
    
    if food_data is None and simple_data is None:
        st.error("No data available. Please check your data files.")
        return
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(food_data, simple_data, rainfall_data, models)
    elif page == "📈 Data Explorer":
        show_data_explorer(food_data, simple_data, rainfall_data)
    elif page == "🤖 Machine Learning":
        show_ml_page(food_data, simple_data, models)
    elif page == "📊 Insights":
        show_insights_page(food_data, simple_data, rainfall_data)
    elif page == "🔮 Forecasting":
        show_forecasting_page(food_data, simple_data, models)

def show_overview(food_data, simple_data, rainfall_data, models):
    """Show the overview dashboard."""
    st.header("📊 Dashboard Overview")
    
    # Select dataset
    dataset_choice = st.selectbox("Select Dataset:", ["Nigeria Food Prices (Detailed)", "Simple Food Prices"])
    
    if dataset_choice == "Nigeria Food Prices (Detailed)" and food_data is not None:
        df = food_data
        st.success(f"✅ Loaded {len(df):,} records from Nigeria Food Prices dataset")
    elif simple_data is not None:
        df = simple_data
        st.success(f"✅ Loaded {len(df):,} records from Simple Food Prices dataset")
    else:
        st.error("No data available")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        unique_commodities = df['commodity'].nunique() if 'commodity' in df.columns else df['market'].nunique()
        st.metric("Unique Items", f"{unique_commodities:,}")
    
    with col3:
        if 'admin1' in df.columns:
            unique_states = df['admin1'].nunique()
            st.metric("States", f"{unique_states:,}")
        else:
            unique_markets = df['market'].nunique()
            st.metric("Markets", f"{unique_markets:,}")
    
    with col4:
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        st.metric("Date Range", date_range)
    
    # Price statistics
    st.subheader("💰 Price Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        price_col = 'price' if 'price' in df.columns else 'usdprice'
        if price_col in df.columns:
            st.metric("Average Price", f"₦{df[price_col].mean():.2f}")
            st.metric("Median Price", f"₦{df[price_col].median():.2f}")
            st.metric("Max Price", f"₦{df[price_col].max():.2f}")
    
    with col2:
        if price_col in df.columns:
            st.metric("Min Price", f"₦{df[price_col].min():.2f}")
            st.metric("Price Std Dev", f"₦{df[price_col].std():.2f}")
            st.metric("Price Range", f"₦{df[price_col].max() - df[price_col].min():.2f}")
    
    # Quick visualizations
    st.subheader("📈 Quick Visualizations")
    
    # Price trends over time
    if 'date' in df.columns and price_col in df.columns:
        fig = px.line(df.groupby('date')[price_col].mean().reset_index(), 
                     x='date', y=price_col,
                     title="Average Price Trends Over Time")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top commodities by average price
    if 'commodity' in df.columns and price_col in df.columns:
        top_commodities = df.groupby('commodity')[price_col].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_commodities.values, y=top_commodities.index,
                    orientation='h', title="Top 10 Commodities by Average Price")
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Model status
    if models:
        st.subheader("🤖 Trained Models")
        model_cols = st.columns(len(models))
        for i, (model_name, model) in enumerate(models.items()):
            with model_cols[i]:
                st.info(f"**{model_name}**\nTrained and ready")

def show_data_explorer(food_data, simple_data, rainfall_data):
    """Show the data exploration page."""
    st.header("📈 Data Explorer")
    
    # Dataset selection
    dataset_choice = st.selectbox("Select Dataset:", ["Nigeria Food Prices (Detailed)", "Simple Food Prices", "Rainfall Data"])
    
    if dataset_choice == "Nigeria Food Prices (Detailed)" and food_data is not None:
        df = food_data
    elif dataset_choice == "Simple Food Prices" and simple_data is not None:
        df = simple_data
    elif dataset_choice == "Rainfall Data" and rainfall_data is not None:
        df = rainfall_data
    else:
        st.error("Selected dataset not available")
        return
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    # Basic statistics
    st.subheader("📊 Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Column information
    st.subheader("ℹ️ Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Interactive filters
    st.subheader("🔍 Interactive Filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'commodity' in df.columns:
            selected_commodities = st.multiselect("Select Commodities:", df['commodity'].unique())
        if 'admin1' in df.columns:
            selected_states = st.multiselect("Select States:", df['admin1'].unique())
        if 'market' in df.columns:
            selected_markets = st.multiselect("Select Markets:", df['market'].unique())
    
    with col2:
        if 'date' in df.columns:
            date_range = st.date_input("Date Range:", 
                                     value=(df['date'].min().date(), df['date'].max().date()),
                                     min_value=df['date'].min().date(),
                                     max_value=df['date'].max().date())
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'commodity' in df.columns and selected_commodities:
        filtered_df = filtered_df[filtered_df['commodity'].isin(selected_commodities)]
    
    if 'admin1' in df.columns and selected_states:
        filtered_df = filtered_df[filtered_df['admin1'].isin(selected_states)]
    
    if 'market' in df.columns and selected_markets:
        filtered_df = filtered_df[filtered_df['market'].isin(selected_markets)]
    
    if 'date' in df.columns and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) & 
            (filtered_df['date'].dt.date <= date_range[1])
        ]
    
    st.write(f"Filtered data: {len(filtered_df):,} records")
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    # Price distribution
    if 'price' in filtered_df.columns:
        fig = px.histogram(filtered_df, x='price', nbins=50, title="Price Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    if 'date' in filtered_df.columns and 'price' in filtered_df.columns:
        time_series = filtered_df.groupby('date')['price'].mean().reset_index()
        fig = px.line(time_series, x='date', y='price', title="Price Trends Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic analysis (if available)
    if 'admin1' in filtered_df.columns and 'price' in filtered_df.columns:
        state_prices = filtered_df.groupby('admin1')['price'].mean().sort_values(ascending=False)
        fig = px.bar(x=state_prices.values, y=state_prices.index,
                    orientation='h', title="Average Prices by State")
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = filtered_df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

def show_ml_page(food_data, simple_data, models):
    """Show the machine learning page."""
    st.header("🤖 Machine Learning")
    
    # Dataset selection
    dataset_choice = st.selectbox("Select Dataset:", ["Nigeria Food Prices (Detailed)", "Simple Food Prices"])
    
    if dataset_choice == "Nigeria Food Prices (Detailed)" and food_data is not None:
        df = food_data
    elif simple_data is not None:
        df = simple_data
    else:
        st.error("No data available")
        return
    
    # Model selection
    st.subheader("🎯 Model Selection")
    
    if models:
        selected_model = st.selectbox("Select Trained Model:", list(models.keys()))
        model = models[selected_model]
        
        st.success(f"✅ Loaded {selected_model} model")
        
        # Model information
        st.subheader("📋 Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {type(model).__name__}")
            if hasattr(model, 'n_estimators'):
                st.write(f"**Number of Estimators:** {model.n_estimators}")
            if hasattr(model, 'max_depth'):
                st.write(f"**Max Depth:** {model.max_depth}")
        
        with col2:
            if hasattr(model, 'feature_importances_'):
                st.write(f"**Number of Features:** {len(model.feature_importances_)}")
                st.write(f"**Feature Importance Available:** ✅")
            else:
                st.write("**Feature Importance:** Not available")
    
    else:
        st.warning("No trained models found. Please train models first using the backend API.")
        return
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        st.subheader("🔍 Feature Importance")
        
        # Get feature names (this would need to be stored with the model)
        feature_names = [f"Feature_{i}" for i in range(len(model.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df.head(20), x='Importance', y='Feature',
                    orientation='h', title="Top 20 Feature Importances")
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Model predictions
    st.subheader("🔮 Model Predictions")
    
    # Create sample data for prediction
    if 'price' in df.columns:
        sample_data = df.sample(min(100, len(df)))
        
        # Simple feature engineering for demonstration
        features = []
        if 'commodity' in sample_data.columns:
            features.append(pd.get_dummies(sample_data['commodity']))
        if 'admin1' in sample_data.columns:
            features.append(pd.get_dummies(sample_data['admin1']))
        
        if features:
            X_sample = pd.concat(features, axis=1).fillna(0)
            
            # Ensure we have the right number of features
            if X_sample.shape[1] == len(model.feature_importances_):
                predictions = model.predict(X_sample)
                
                # Create comparison plot
                actual = sample_data['price'].values[:len(predictions)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual, y=predictions, mode='markers',
                                       name='Predictions vs Actual'))
                fig.add_trace(go.Scatter(x=[actual.min(), actual.max()], 
                                       y=[actual.min(), actual.max()],
                                       mode='lines', name='Perfect Prediction'))
                fig.update_layout(title="Model Predictions vs Actual Values",
                                xaxis_title="Actual Price",
                                yaxis_title="Predicted Price")
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate metrics
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                
                r2 = r2_score(actual, predictions)
                mse = mean_squared_error(actual, predictions)
                mae = mean_absolute_error(actual, predictions)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col2:
                    st.metric("MSE", f"{mse:.2f}")
                with col3:
                    st.metric("MAE", f"{mae:.2f}")

def show_insights_page(food_data, simple_data, rainfall_data):
    """Show the insights page."""
    st.header("📊 Insights & Analytics")
    
    # Dataset selection
    dataset_choice = st.selectbox("Select Dataset:", ["Nigeria Food Prices (Detailed)", "Simple Food Prices"])
    
    if dataset_choice == "Nigeria Food Prices (Detailed)" and food_data is not None:
        df = food_data
    elif simple_data is not None:
        df = simple_data
    else:
        st.error("No data available")
        return
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    # Price volatility analysis
    if 'price' in df.columns and 'date' in df.columns:
        price_volatility = df.groupby('date')['price'].std().mean()
        st.metric("Average Price Volatility", f"₦{price_volatility:.2f}")
        
        # Price trends by commodity
        if 'commodity' in df.columns:
            commodity_trends = df.groupby(['commodity', 'date'])['price'].mean().reset_index()
            
            # Calculate trend for each commodity
            trends = {}
            for commodity in commodity_trends['commodity'].unique():
                comm_data = commodity_trends[commodity_trends['commodity'] == commodity]
                if len(comm_data) > 1:
                    x = np.arange(len(comm_data))
                    y = comm_data['price'].values
                    slope = np.polyfit(x, y, 1)[0]
                    trends[commodity] = slope
            
            # Sort by trend
            sorted_trends = sorted(trends.items(), key=lambda x: x[1], reverse=True)
            
            st.write("**Price Trends by Commodity (Top 10):**")
            for commodity, trend in sorted_trends[:10]:
                direction = "📈" if trend > 0 else "📉"
                st.write(f"{direction} {commodity}: {trend:.4f} ₦/day")
    
    # Seasonal analysis
    if 'date' in df.columns and 'price' in df.columns:
        st.subheader("📅 Seasonal Analysis")
        
        df['month'] = df['date'].dt.month
        monthly_prices = df.groupby('month')['price'].mean()
        
        fig = px.bar(x=monthly_prices.index, y=monthly_prices.values,
                    title="Average Prices by Month",
                    labels={'x': 'Month', 'y': 'Average Price (₦)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic insights
    if 'admin1' in df.columns and 'price' in df.columns:
        st.subheader("🗺️ Geographic Insights")
        
        state_analysis = df.groupby('admin1').agg({
            'price': ['mean', 'std', 'count']
        }).round(2)
        
        state_analysis.columns = ['Average Price', 'Price Std Dev', 'Record Count']
        state_analysis = state_analysis.sort_values('Average Price', ascending=False)
        
        st.dataframe(state_analysis.head(20), use_container_width=True)
        
        # Map visualization (if coordinates available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            map_data = df.groupby(['admin1', 'latitude', 'longitude'])['price'].mean().reset_index()
            
            fig = px.scatter_mapbox(map_data, lat="latitude", lon="longitude", 
                                  size="price", color="price",
                                  hover_name="admin1", hover_data=["price"],
                                  mapbox_style="open-street-map",
                                  title="Price Distribution by Location")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # Market analysis
    if 'market' in df.columns and 'price' in df.columns:
        st.subheader("🏪 Market Analysis")
        
        market_analysis = df.groupby('market').agg({
            'price': ['mean', 'std', 'count']
        }).round(2)
        
        market_analysis.columns = ['Average Price', 'Price Std Dev', 'Record Count']
        market_analysis = market_analysis.sort_values('Average Price', ascending=False)
        
        st.dataframe(market_analysis.head(20), use_container_width=True)

def show_forecasting_page(food_data, simple_data, models):
    """Show the forecasting page."""
    st.header("🔮 Price Forecasting")
    
    # Dataset selection
    dataset_choice = st.selectbox("Select Dataset:", ["Nigeria Food Prices (Detailed)", "Simple Food Prices"])
    
    if dataset_choice == "Nigeria Food Prices (Detailed)" and food_data is not None:
        df = food_data
    elif simple_data is not None:
        df = simple_data
    else:
        st.error("No data available")
        return
    
    # Forecasting parameters
    st.subheader("⚙️ Forecasting Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
        selected_commodity = st.selectbox("Select Commodity:", df['commodity'].unique() if 'commodity' in df.columns else ['All'])
    
    with col2:
        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95)
        model_choice = st.selectbox("Select Model:", list(models.keys()) if models else ['Simple Trend'])
    
    # Filter data
    if selected_commodity != 'All' and 'commodity' in df.columns:
        forecast_df = df[df['commodity'] == selected_commodity].copy()
    else:
        forecast_df = df.copy()
    
    if 'date' in forecast_df.columns and 'price' in forecast_df.columns:
        # Simple forecasting using trend analysis
        st.subheader("📈 Forecast Results")
        
        # Prepare time series data
        ts_data = forecast_df.groupby('date')['price'].mean().reset_index()
        ts_data = ts_data.sort_values('date')
        
        if len(ts_data) > 1:
            # Calculate trend
            x = np.arange(len(ts_data))
            y = ts_data['price'].values
            
            # Linear trend
            coeffs = np.polyfit(x, y, 1)
            trend_line = np.poly1d(coeffs)
            
            # Generate forecast
            future_dates = pd.date_range(start=ts_data['date'].max(), periods=forecast_days+1, freq='D')[1:]
            future_x = np.arange(len(ts_data), len(ts_data) + forecast_days)
            future_prices = trend_line(future_x)
            
            # Calculate confidence intervals
            residuals = y - trend_line(x)
            std_error = np.std(residuals)
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # Approximate for 95% and 99%
            
            upper_bound = future_prices + z_score * std_error
            lower_bound = future_prices - z_score * std_error
            
            # Create forecast dataframe
            forecast_data = pd.DataFrame({
                'date': future_dates,
                'predicted_price': future_prices,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound
            })
            
            # Combine historical and forecast data
            historical_data = ts_data.copy()
            historical_data['type'] = 'Historical'
            
            forecast_plot_data = forecast_data.copy()
            forecast_plot_data['type'] = 'Forecast'
            forecast_plot_data = forecast_plot_data.rename(columns={'predicted_price': 'price'})
            
            # Create visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['price'],
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue')
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['predicted_price'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['lower_bound'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name=f'{confidence_level*100}% Confidence Interval',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f"Price Forecast for {selected_commodity}",
                xaxis_title="Date",
                yaxis_title="Price (₦)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            st.subheader("📊 Forecast Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"₦{ts_data['price'].iloc[-1]:.2f}")
            
            with col2:
                st.metric("Forecast Price", f"₦{future_prices[-1]:.2f}")
            
            with col3:
                price_change = ((future_prices[-1] - ts_data['price'].iloc[-1]) / ts_data['price'].iloc[-1]) * 100
                st.metric("Price Change", f"{price_change:+.2f}%")
            
            # Forecast table
            st.subheader("📋 Detailed Forecast")
            forecast_display = forecast_data.copy()
            forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
            forecast_display = forecast_display.round(2)
            st.dataframe(forecast_display, use_container_width=True)
            
        else:
            st.warning("Insufficient data for forecasting")

if __name__ == "__main__":
    main()

