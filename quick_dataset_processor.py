#!/usr/bin/env python3
"""
Quick Dataset Processor for PriceOptima
Simplified script for processing individual datasets
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class QuickDatasetProcessor:
    """Quick dataset processing for individual files"""
    
    def __init__(self):
        self.required_columns = {
            "Date": ["date", "datetime", "timestamp"],
            "Product": ["product", "product name", "product_name", "item"],
            "Category": ["category", "type", "product_category"],
            "Price": ["price", "unit price", "unit_price"],
            "Quantity": ["quantity", "quantity sold", "qty", "qty sold"],
            "Revenue": ["revenue", "sales", "total", "total revenue"]
        }
        
        self.optional_columns = {
            "Waste": ["waste", "waste amount", "waste_amount"],
            "Cost": ["cost", "unit cost", "unit_cost"],
            "Supplier": ["supplier", "vendor", "supplier_name"]
        }
    
    def normalize_column_name(self, name: str) -> str:
        """Normalize column names for matching"""
        return re.sub(r"\s+", " ", str(name).strip().lower())
    
    def find_matching_column(self, target_name: str, available_columns: list) -> str:
        """Find matching column name"""
        target_lower = self.normalize_column_name(target_name)
        synonyms = self.required_columns.get(target_name, []) + self.optional_columns.get(target_name, [])
        candidates = [target_lower] + [self.normalize_column_name(syn) for syn in synonyms]
        
        for col in available_columns:
            if self.normalize_column_name(col) in candidates:
                return col
        return None
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        df = df.copy()
        available_columns = list(df.columns)
        rename_map = {}
        
        # Map all columns
        all_columns = {**self.required_columns, **self.optional_columns}
        for standard_name, synonyms in all_columns.items():
            matching_col = self.find_matching_column(standard_name, available_columns)
            if matching_col:
                rename_map[matching_col] = standard_name
        
        df = df.rename(columns=rename_map)
        
        # Check required columns
        missing_required = [col for col in self.required_columns.keys() if col not in df.columns]
        if missing_required:
            print(f"Warning: Missing required columns: {missing_required}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing function"""
        print("Starting data preprocessing...")
        
        # 1. Standardize columns
        print("1. Standardizing columns...")
        df = self.standardize_columns(df)
        
        # 2. Parse dates
        print("2. Parsing dates...")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        
        # 3. Convert numeric columns
        print("3. Converting numeric columns...")
        numeric_cols = ["Price", "Quantity", "Revenue", "Waste", "Cost"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. Compute Revenue if missing
        print("4. Computing Revenue...")
        if "Revenue" not in df.columns or df["Revenue"].isna().all():
            if "Price" in df.columns and "Quantity" in df.columns:
                df["Revenue"] = df["Price"] * df["Quantity"]
                print("   Revenue computed as Price × Quantity")
        
        # 5. Fill missing values
        print("5. Filling missing values...")
        for col in df.columns:
            if df[col].dtype in ['object'] and col != 'Date':
                df[col] = df[col].fillna('Unknown')
            elif df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(0)
        
        # 6. Add date features
        print("6. Adding date features...")
        if "Date" in df.columns:
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["Day"] = df["Date"].dt.day
            df["DayOfWeek"] = df["Date"].dt.dayofweek
        
        # 7. Remove duplicates
        print("7. Removing duplicates...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        if removed > 0:
            print(f"   Removed {removed} duplicate rows")
        
        print(f"Preprocessing completed! Final dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def analyze_data(self, df: pd.DataFrame) -> dict:
        """Analyze the processed data"""
        analysis = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns)
            },
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns:
                analysis['categorical_summary'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].value_counts().head(3).to_dict()
                }
        
        return analysis
    
    def process_file(self, input_path: str, output_path: str = None) -> dict:
        """Process a single file"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Determine output path
        if output_path is None:
            filename = os.path.basename(input_path)
            name, ext = os.path.splitext(filename)
            output_path = f"DATASET/processed/processed_{filename}"
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load data
        print(f"Loading data from: {input_path}")
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(input_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(input_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Analyze data
        analysis = self.analyze_data(processed_df)
        
        # Save processed data
        if file_ext == '.parquet':
            processed_df.to_parquet(output_path, index=False)
        else:
            processed_df.to_csv(output_path, index=False)
        
        print(f"Processed data saved to: {output_path}")
        
        return {
            'input_path': input_path,
            'output_path': output_path,
            'rows_before': len(df),
            'rows_after': len(processed_df),
            'analysis': analysis
        }

def main():
    """Main function for quick processing"""
    print("PriceOptima Quick Dataset Processor")
    print("=" * 40)
    
    processor = QuickDatasetProcessor()
    
    # Example usage
    try:
        # Process a single file
        input_file = "DATASET/raw/your_data.csv"  # Replace with your file
        result = processor.process_file(input_file)
        
        print(f"\nProcessing Results:")
        print(f"- Input: {result['input_path']}")
        print(f"- Output: {result['output_path']}")
        print(f"- Rows: {result['rows_before']} -> {result['rows_after']}")
        print(f"- Columns: {result['analysis']['basic_info']['total_columns']}")
        
        # Show data summary
        analysis = result['analysis']
        print(f"\nData Summary:")
        print(f"- Missing values: {sum(analysis['missing_values'].values())}")
        
        if analysis['numeric_summary']:
            print(f"- Numeric columns: {len(analysis['numeric_summary'])}")
        
        if analysis['categorical_summary']:
            print(f"- Categorical columns: {len(analysis['categorical_summary'])}")
        
    except FileNotFoundError:
        print("Error: Input file not found!")
        print("Please update the input_file path in the script")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
