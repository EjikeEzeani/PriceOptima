#!/usr/bin/env python3
"""
Data Preprocessing Script for PriceOptima
Standalone preprocessing functions extracted from api_backend.py
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
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
    print("Starting data preprocessing...")
    
    # Step 1: Standardize columns
    print("1. Standardizing column names...")
    df = standardize_columns(original_df)
    print(f"   Columns after standardization: {list(df.columns)}")
    
    # Step 2: Parse dates
    print("2. Parsing date columns...")
    df = parse_dates(df)
    if "Date" in df.columns:
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Step 3: Convert numeric columns
    print("3. Converting numeric columns...")
    df = coerce_numeric(df, ["Price", "Quantity", "Revenue", "Waste", "Cost"])
    
    # Step 4: Compute Revenue if missing
    print("4. Computing Revenue if missing...")
    if "Revenue" not in df.columns or df["Revenue"].isna().all():
        if "Price" in df.columns and "Quantity" in df.columns:
            df["Revenue"] = df["Price"].fillna(0) * df["Quantity"].fillna(0)
            print("   Revenue computed as Price × Quantity")
        else:
            raise ValueError("Cannot compute Revenue: missing Price or Quantity")
    
    # Step 5: Fill missing values
    print("5. Filling missing values...")
    for col in ["Price", "Quantity", "Revenue", "Waste", "Cost"]:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(0)
                print(f"   Filled {missing_count} missing values in {col} with 0")
    
    # Step 6: Drop rows with missing critical categorical fields
    print("6. Removing rows with missing critical data...")
    initial_rows = len(df)
    df = df.dropna(subset=["Product", "Category"])
    final_rows = len(df)
    removed_rows = initial_rows - final_rows
    if removed_rows > 0:
        print(f"   Removed {removed_rows} rows with missing Product or Category")
    
    print(f"Preprocessing completed! Final dataset: {len(df)} rows, {len(df.columns)} columns")
    return df

def analyze_data(df: pd.DataFrame) -> dict:
    """Analyze the preprocessed data and return insights."""
    analysis = {
        "basic_info": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns)
        },
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": {},
        "categorical_summary": {}
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in df.columns:
            analysis["categorical_summary"][col] = {
                "unique_values": df[col].nunique(),
                "most_common": df[col].value_counts().head(5).to_dict()
            }
    
    return analysis

def main():
    """Main function to demonstrate usage."""
    print("PriceOptima Data Preprocessing Script")
    print("=" * 40)
    
    try:
        # Load your data
        print("Loading data...")
        # Replace 'your_data.csv' with your actual data file
        df = pd.read_csv("your_data.csv")
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"Original columns: {list(df.columns)}")
        
        # Preprocess data
        processed_df = preprocess_data(df)
        
        # Analyze processed data
        print("\nAnalyzing processed data...")
        analysis = analyze_data(processed_df)
        
        print(f"\nData Analysis Results:")
        print(f"- Total rows: {analysis['basic_info']['total_rows']:,}")
        print(f"- Total columns: {analysis['basic_info']['total_columns']}")
        print(f"- Missing values: {sum(analysis['missing_values'].values())}")
        
        # Show numeric summary
        if analysis['numeric_summary']:
            print(f"\nNumeric columns summary:")
            for col, stats in analysis['numeric_summary'].items():
                print(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
        # Show categorical summary
        if analysis['categorical_summary']:
            print(f"\nCategorical columns summary:")
            for col, info in analysis['categorical_summary'].items():
                print(f"  {col}: {info['unique_values']} unique values")
        
        # Save processed data
        output_file = "processed_data.csv"
        processed_df.to_csv(output_file, index=False)
        print(f"\nProcessed data saved to: {output_file}")
        
    except FileNotFoundError:
        print("Error: 'your_data.csv' not found. Please provide your data file.")
        print("Expected CSV format with columns: Date, Product, Category, Price, Quantity, Revenue")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
