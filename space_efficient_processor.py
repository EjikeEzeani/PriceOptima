#!/usr/bin/env python3
"""
Space-Efficient Dataset Processor for PriceOptima
Processes datasets in smaller batches to avoid disk space issues
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def process_single_file_efficient(file_path, output_dir="DATASET/processed"):
    """Process a single file with space-efficient approach"""
    
    print(f"Processing: {os.path.basename(file_path)}")
    
    try:
        # Load data
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            print(f"  ❌ Unsupported format: {file_ext}")
            return False
        
        print(f"  📊 Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Process data
        processed_df = preprocess_data_efficient(df)
        
        # Save with compression
        output_filename = f"processed_{os.path.basename(file_path)}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Use compression to save space
        if file_ext == '.parquet':
            processed_df.to_parquet(output_path, index=False, compression='gzip')
        else:
            processed_df.to_csv(output_path, index=False, compression='gzip')
        
        print(f"  ✅ Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

def preprocess_data_efficient(df):
    """Efficient preprocessing with minimal memory usage"""
    df = df.copy()
    
    # 1. Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    # 2. Map columns to standard format
    column_mapping = {
        'timestamp': 'date',
        'store_id': 'location',
        'sku': 'product',
        'unit_price': 'price',
        'qty': 'quantity',
        'cost_price': 'cost',
        'waste_units': 'waste',
        'commodity': 'product',
        'admin1': 'region',
        'admin2': 'state',
        'market': 'location',
        'price': 'price',
        'usdprice': 'price_usd'
    }
    
    df = df.rename(columns=column_mapping)
    
    # 3. Compute Revenue if missing
    if 'revenue' not in df.columns and 'price' in df.columns and 'quantity' in df.columns:
        df['revenue'] = df['price'] * df['quantity']
    
    # 4. Handle missing values efficiently
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(0)
    
    # 5. Convert numeric columns
    numeric_cols = ['price', 'quantity', 'revenue', 'cost', 'waste', 'price_usd']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 6. Parse dates efficiently
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
    
    # 7. Create category from product
    if 'category' not in df.columns and 'product' in df.columns:
        df['category'] = df['product'].str.split().str[0]
    
    # 8. Remove duplicates
    df = df.drop_duplicates()
    
    # 9. Remove rows with missing critical data
    df = df.dropna(subset=['product', 'price'])
    
    return df

def process_datasets_in_batches(batch_size=5):
    """Process datasets in small batches to avoid disk space issues"""
    
    print("Space-Efficient Dataset Processing")
    print("=" * 40)
    
    # Find all data files
    RAW_DIR = "DATASET/raw"
    PROCESSED_DIR = "DATASET/processed"
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    file_patterns = ["*.csv", "*.parquet", "*.xlsx", "*.xls"]
    all_files = []
    
    for pattern in file_patterns:
        files = glob.glob(os.path.join(RAW_DIR, "**", pattern), recursive=True)
        all_files.extend(files)
    
    print(f"Found {len(all_files)} files to process")
    print(f"Processing in batches of {batch_size} files...")
    
    successful = 0
    failed = 0
    
    # Process in batches
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"\n--- Batch {batch_num} ({len(batch_files)} files) ---")
        
        for file_path in batch_files:
            if process_single_file_efficient(file_path, PROCESSED_DIR):
                successful += 1
            else:
                failed += 1
        
        # Check disk space after each batch
        print(f"Batch {batch_num} completed. Success: {successful}, Failed: {failed}")
        
        # Ask user if they want to continue
        if i + batch_size < len(all_files):
            response = input(f"Continue with next batch? (y/n): ").lower()
            if response != 'y':
                break
    
    print(f"\nFinal Results:")
    print(f"Total files: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%")

def process_specific_files(file_list):
    """Process only specific files from a list"""
    
    print("Processing Specific Files")
    print("=" * 30)
    
    PROCESSED_DIR = "DATASET/processed"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for file_path in file_list:
        if os.path.exists(file_path):
            if process_single_file_efficient(file_path, PROCESSED_DIR):
                successful += 1
            else:
                failed += 1
        else:
            print(f"File not found: {file_path}")
            failed += 1
    
    print(f"\nResults: {successful} successful, {failed} failed")

if __name__ == "__main__":
    print("Choose processing option:")
    print("1. Process all files in batches")
    print("2. Process specific files")
    print("3. Process one file")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        batch_size = int(input("Enter batch size (default 5): ") or "5")
        process_datasets_in_batches(batch_size)
    
    elif choice == "2":
        print("Enter file paths (one per line, empty line to finish):")
        files = []
        while True:
            file_path = input().strip()
            if not file_path:
                break
            files.append(file_path)
        
        if files:
            process_specific_files(files)
        else:
            print("No files specified")
    
    elif choice == "3":
        file_path = input("Enter file path: ").strip()
        if file_path and os.path.exists(file_path):
            process_single_file_efficient(file_path)
        else:
            print("File not found")
    
    else:
        print("Invalid choice")
