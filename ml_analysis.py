#!/usr/bin/env python3
"""
🤖 Machine Learning Analysis Script
Comprehensive ML analysis for price prediction and revenue optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

def load_data():
    """Load data from various possible locations"""
    data_paths = [
        "DATASET/raw/wfp_food_prices_nga.csv",
        "DATASET/raw/wfp_food_prices.csv",
        "data/processed/merged_input_dataset.csv"
    ]
    
    for path in data_paths:
        try:
            df = pd.read_csv(path)
            st.success(f"✅ Loaded data from: {path}")
            return df
        except FileNotFoundError:
            continue
    
    # Create dummy data if no files found
    st.warning("⚠️ No data files found. Creating dummy data for demonstration.")
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic dummy data
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    commodities = np.random.choice(['Rice', 'Wheat', 'Maize', 'Beans', 'Tomatoes'], n_samples)
    states = np.random.choice(['Lagos', 'Abuja', 'Kano', 'Rivers', 'Ogun'], n_samples)
    
    # Create price with some seasonality and commodity effects
    base_prices = {'Rice': 200, 'Wheat': 150, 'Maize': 100, 'Beans': 300, 'Tomatoes': 250}
    prices = []
    quantities = []
    
    for commodity in commodities:
        base_price = base_prices[commodity]
        # Add seasonality
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
        # Add random variation
        price_variation = np.random.normal(0, 0.1)
        price = base_price * seasonal_factor + price_variation
        prices.append(price)
        
        # Quantity inversely related to price
        quantity = 1000 - (price - base_price) * 2 + np.random.normal(0, 50)
        quantities.append(max(0, quantity))
    
    df = pd.DataFrame({
        'date': dates,
        'commodity': commodities,
        'state': states,
        'price': prices,
        'quantity': quantities,
        'revenue': np.array(prices) * np.array(quantities)
    })
    
    return df

def preprocess_data(df, target_column='price'):
    """Preprocess data for machine learning"""
    df_processed = df.copy()
    
    # Handle date features
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed['year'] = df_processed['date'].dt.year
        df_processed['month'] = df_processed['date'].dt.month
        df_processed['day'] = df_processed['date'].dt.day
        df_processed['dayofweek'] = df_processed['date'].dt.dayofweek
        df_processed['is_weekend'] = df_processed['dayofweek'].isin([5, 6]).astype(int)
    
    # Encode categorical variables
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in categorical_columns:
        if col != 'date':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
    
    # Remove any remaining non-numeric columns except target
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    if target_column in numeric_columns:
        feature_columns = [col for col in numeric_columns if col != target_column]
    else:
        feature_columns = numeric_columns.tolist()
    
    return df_processed, feature_columns, le_dict

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models and return results"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred_test': y_pred_test
        }
    
    return results

def create_model_comparison_plot(results):
    """Create comparison plot for different models"""
    model_names = list(results.keys())
    test_r2_scores = [results[name]['test_r2'] for name in model_names]
    test_rmse_scores = [results[name]['test_rmse'] for name in model_names]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('R² Score Comparison', 'RMSE Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=model_names, y=test_r2_scores, name='R² Score', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=model_names, y=test_rmse_scores, name='RMSE', marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_prediction_plot(y_test, y_pred, model_name):
    """Create prediction vs actual plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name} - Predictions vs Actual',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=400
    )
    
    return fig

def feature_importance_analysis(model, feature_names):
    """Analyze feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(
            importance_df.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance (Top 15)'
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        
        return fig, importance_df
    else:
        return None, None

def main():
    st.set_page_config(
        page_title="Machine Learning Analysis",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Machine Learning Analysis for Price Prediction")
    st.markdown("---")
    
    # Load data
    st.header("📊 Data Loading and Preprocessing")
    df = load_data()
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Target selection
    st.subheader("🎯 Target Variable Selection")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    target_column = st.selectbox("Select Target Variable:", numeric_columns, index=0)
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        df_processed, feature_columns, le_dict = preprocess_data(df, target_column)
    
    st.success(f"✅ Preprocessing complete. Using {len(feature_columns)} features")
    
    # Display feature information
    st.subheader("🔧 Feature Information")
    feature_info = pd.DataFrame({
        'Feature': feature_columns,
        'Type': [df_processed[col].dtype for col in feature_columns],
        'Missing Values': [df_processed[col].isnull().sum() for col in feature_columns]
    })
    st.dataframe(feature_info, use_container_width=True)
    
    # Train-test split
    st.header("🔄 Model Training")
    
    X = df_processed[feature_columns].fillna(0)
    y = df_processed[target_column].fillna(0)
    
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.slider("Random State", 0, 100, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    st.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Train models
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            results = train_models(X_train, X_test, y_train, y_test)
        
        st.success("✅ Model training completed!")
        
        # Display results table
        st.subheader("📊 Model Performance Comparison")
        
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Test R²': [f"{results[name]['test_r2']:.4f}" for name in results.keys()],
            'Test RMSE': [f"{results[name]['test_rmse']:.2f}" for name in results.keys()],
            'Test MAE': [f"{results[name]['test_mae']:.2f}" for name in results.keys()],
            'CV R² (Mean)': [f"{results[name]['cv_mean']:.4f}" for name in results.keys()],
            'CV R² (Std)': [f"{results[name]['cv_std']:.4f}" for name in results.keys()]
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Model comparison plots
        st.subheader("📈 Model Comparison")
        comparison_fig = create_model_comparison_plot(results)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Best model selection
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        st.success(f"🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['test_r2']:.4f})")
        
        # Detailed analysis of best model
        st.subheader(f"🔍 Detailed Analysis - {best_model_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction plot
            pred_fig = create_prediction_plot(
                y_test, 
                results[best_model_name]['y_pred_test'], 
                best_model_name
            )
            st.plotly_chart(pred_fig, use_container_width=True)
        
        with col2:
            # Feature importance
            if hasattr(results[best_model_name]['model'], 'feature_importances_'):
                importance_fig, importance_df = feature_importance_analysis(
                    results[best_model_name]['model'], 
                    feature_columns
                )
                if importance_fig:
                    st.plotly_chart(importance_fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type")
        
        # Model saving
        st.subheader("💾 Save Model")
        if st.button("Save Best Model"):
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{best_model_name.lower().replace(' ', '_')}_{timestamp}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            
            joblib.dump(results[best_model_name]['model'], model_path)
            st.success(f"✅ Model saved to: {model_path}")
        
        # Predictions on new data
        st.subheader("🔮 Make Predictions")
        
        st.write("Enter values for prediction (leave empty for random values):")
        
        input_data = {}
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(feature_columns[:6]):  # Show first 6 features
            col = col1 if i % 2 == 0 else col2
            with col:
                if df_processed[feature].dtype in ['int64', 'float64']:
                    min_val = float(df_processed[feature].min())
                    max_val = float(df_processed[feature].max())
                    default_val = float(df_processed[feature].mean())
                    input_data[feature] = st.number_input(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=(max_val - min_val) / 100
                    )
                else:
                    input_data[feature] = st.number_input(f"{feature}:", value=0.0)
        
        if st.button("Predict"):
            # Create prediction input
            prediction_input = np.array([input_data[feature] for feature in feature_columns]).reshape(1, -1)
            
            # Make prediction
            prediction = results[best_model_name]['model'].predict(prediction_input)[0]
            
            st.success(f"🎯 Predicted {target_column}: {prediction:.2f}")
            
            # Show confidence interval (rough estimate)
            rmse = results[best_model_name]['test_rmse']
            st.info(f"📊 Confidence Interval: {prediction - 1.96*rmse:.2f} to {prediction + 1.96*rmse:.2f}")

if __name__ == "__main__":
    main()
