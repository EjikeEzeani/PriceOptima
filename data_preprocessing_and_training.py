#!/usr/bin/env python3
"""
Data Preprocessing and Model Training Script for PriceOptima
This script extracts the preprocessing and training functions from api_backend.py
for standalone use in data science workflows.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Column mapping configuration
REQUIRED_COLUMNS_SYNONYMS = {
    "Date": ["date"],
    "Product": ["product", "product name", "product_name", "item", "item name"],
    "Category": ["category", "type"],
    "Price": ["price", "unit price", "unit_price"],
    "Quantity": ["quantity", "quantity sold", "qty", "qty sold"],
    "Revenue": ["revenue", "sales", "total", "total revenue"]
}

OPTIONAL_COLUMNS_SYNONYMS = {
    "Waste": ["waste", "waste amount", "waste_amount"],
    "Cost": ["cost", "unit cost", "unit_cost"],
    "Supplier": ["supplier", "vendor"]
}

def _normalize_column_name(name: str) -> str:
    """Normalize column names for matching."""
    return re.sub(r"\s+", " ", str(name).strip().lower())

def standardize_columns(original_df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names using synonyms. Returns a copy."""
    df = original_df.copy()
    lower_map = {i: _normalize_column_name(i) for i in df.columns}

    rename_map = {}
    # Required columns
    for standard, synonyms in REQUIRED_COLUMNS_SYNONYMS.items():
        candidates = [standard.lower(), *synonyms]
        match = next((col for col, low in lower_map.items() if low in candidates), None)
        if match is not None:
            rename_map[match] = standard

    # Optional columns
    for standard, synonyms in OPTIONAL_COLUMNS_SYNONYMS.items():
        candidates = [standard.lower(), *synonyms]
        match = next((col for col, low in lower_map.items() if low in candidates), None)
        if match is not None:
            rename_map[match] = standard

    df = df.rename(columns=rename_map)

    # Validate required columns present (after rename)
    missing = [col for col in REQUIRED_COLUMNS_SYNONYMS.keys() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    return df

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Date column supporting YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY."""
    df = df.copy()
    if "Date" in df.columns:
        # Try multiple formats
        date_series = pd.to_datetime(df["Date"], errors="coerce", format=None)
        df["Date"] = date_series
    return df

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns to numeric type, handling errors gracefully."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def preprocess_data(original_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data: standardize columns, parse dates, coerce numerics, compute Revenue when needed, handle missing."""
    df = standardize_columns(original_df)
    df = parse_dates(df)
    df = coerce_numeric(df, ["Price", "Quantity", "Revenue", "Waste", "Cost"])

    # Compute Revenue if missing
    if "Revenue" not in df.columns or df["Revenue"].isna().all():
        if "Price" in df.columns and "Quantity" in df.columns:
            df["Revenue"] = df["Price"].fillna(0) * df["Quantity"].fillna(0)
        else:
            raise ValueError("Cannot compute Revenue: missing Price or Quantity")

    # Fill missing values
    for col in ["Price", "Quantity", "Revenue", "Waste", "Cost"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Drop rows with missing critical categorical fields; keep rows even if Date is NaT
    df = df.dropna(subset=["Product", "Category"])

    return df

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Prepare features for machine learning."""
    df = df.copy()
    
    # Create target variable (Price)
    if "Price" not in df.columns:
        raise ValueError("Price column is required for training")
    
    target = "Price"
    
    # Create features
    features_df = pd.DataFrame()
    
    # Categorical features
    if "Category" in df.columns:
        features_df["Category"] = df["Category"]
    
    if "Product" in df.columns:
        features_df["Product"] = df["Product"]
    
    # Numerical features
    if "Quantity" in df.columns:
        features_df["Quantity"] = df["Quantity"]
    
    if "Revenue" in df.columns:
        features_df["Revenue"] = df["Revenue"]
    
    if "Waste" in df.columns:
        features_df["Waste"] = df["Waste"]
    
    if "Cost" in df.columns:
        features_df["Cost"] = df["Cost"]
    
    # Date features
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        features_df["Year"] = df["Date"].dt.year
        features_df["Month"] = df["Date"].dt.month
        features_df["Day"] = df["Date"].dt.day
        features_df["DayOfWeek"] = df["Date"].dt.dayofweek
    
    # One-hot encode categorical variables
    categorical_cols = ["Category", "Product"]
    for col in categorical_cols:
        if col in features_df.columns:
            dummies = pd.get_dummies(features_df[col], prefix=col)
            features_df = pd.concat([features_df, dummies], axis=1)
            features_df = features_df.drop(columns=[col])
    
    # Fill any remaining NaN values
    features_df = features_df.fillna(0)
    
    return features_df, target

def train_ml_model(df: pd.DataFrame, model_type: str = "random_forest") -> dict:
    """Train a machine learning model on the preprocessed data."""
    try:
        # Prepare features
        X, target = prepare_features(df)
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize model
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "linear_regression":
            model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_type}_{timestamp}.joblib"
        model_path = os.path.join("models", model_filename)
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        joblib.dump(model, model_path)
        
        return {
            "model_type": model_type,
            "r2_score": r2,
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "model_path": model_path,
            "feature_importance": dict(zip(X.columns, model.feature_importances_)) if hasattr(model, 'feature_importances_') else None
        }
        
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main function to demonstrate usage."""
    print("PriceOptima Data Preprocessing and Training Script")
    print("=" * 50)
    
    # Example usage
    try:
        # Load your data
        print("Loading data...")
        # Replace 'your_data.csv' with your actual data file
        df = pd.read_csv("your_data.csv")
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Preprocess data
        print("Preprocessing data...")
        processed_df = preprocess_data(df)
        print(f"After preprocessing: {len(processed_df)} rows and {len(processed_df.columns)} columns")
        
        # Train models
        models = ["random_forest", "gradient_boosting", "linear_regression"]
        
        for model_type in models:
            print(f"\nTraining {model_type}...")
            results = train_ml_model(processed_df, model_type)
            
            if "error" in results:
                print(f"Error training {model_type}: {results['error']}")
            else:
                print(f"R² Score: {results['r2_score']:.4f}")
                print(f"MAE: {results['mae']:.4f}")
                print(f"Model saved to: {results['model_path']}")
        
        print("\nTraining completed!")
        
    except FileNotFoundError:
        print("Error: 'your_data.csv' not found. Please provide your data file.")
        print("Expected CSV format with columns: Date, Product, Category, Price, Quantity, Revenue")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
