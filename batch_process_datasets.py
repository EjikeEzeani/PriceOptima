#!/usr/bin/env python3
"""
Batch Dataset Processing for PriceOptima DATASET folder
Processes all files in DATASET/raw and outputs to DATASET/processed
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

def process_datasets_batch():
    """Process all datasets in the DATASET/raw folder"""
    
    print("PriceOptima Batch Dataset Processing")
    print("=" * 50)
    
    # Configuration
    RAW_DIR = "DATASET/raw"
    PROCESSED_DIR = "DATASET/processed"
    REPORTS_DIR = "DATASET/processed/reports"
    
    # Create output directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Find all data files (including subdirectories)
    file_patterns = ["*.csv", "*.parquet", "*.xlsx", "*.xls"]
    all_files = []
    
    for pattern in file_patterns:
        # Search in main directory and subdirectories
        files = glob.glob(os.path.join(RAW_DIR, "**", pattern), recursive=True)
        all_files.extend(files)
    
    print(f"Found {len(all_files)} files to process:")
    for file in all_files:
        print(f"  - {os.path.basename(file)}")
    
    if not all_files:
        print("No files found in DATASET/raw folder!")
        return
    
    # Processing results
    results = []
    successful = 0
    failed = 0
    
    # Process each file
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Processing: {os.path.basename(file_path)}")
        
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
                failed += 1
                continue
            
            print(f"  📊 Loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Basic preprocessing
            processed_df = basic_preprocess(df)
            
            # Save processed data
            output_filename = f"processed_{os.path.basename(file_path)}"
            output_path = os.path.join(PROCESSED_DIR, output_filename)
            
            if file_ext == '.parquet':
                processed_df.to_parquet(output_path, index=False)
            else:
                processed_df.to_csv(output_path, index=False)
            
            print(f"  ✅ Saved: {output_path}")
            
            # Record results
            results.append({
                'file': os.path.basename(file_path),
                'status': 'success',
                'rows_before': len(df),
                'rows_after': len(processed_df),
                'columns': len(processed_df.columns),
                'output_file': output_filename
            })
            
            successful += 1
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                'file': os.path.basename(file_path),
                'status': 'failed',
                'error': str(e)
            })
            failed += 1
    
    # Generate summary report
    print(f"\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files: {len(all_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(all_files)*100:.1f}%")
    
    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_files': len(all_files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful/len(all_files) if all_files else 0
        },
        'results': results
    }
    
    report_path = os.path.join(REPORTS_DIR, f"batch_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    print(f"Processed files saved to: {PROCESSED_DIR}")

def basic_preprocess(df):
    """Enhanced preprocessing steps for PriceOptima datasets"""
    df = df.copy()
    
    print(f"    Original shape: {df.shape}")
    print(f"    Original columns: {list(df.columns)}")
    
    # 1. Clean column names and standardize
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    # 2. Map columns to standard PriceOptima format
    column_mapping = {
        'timestamp': 'date',
        'store_id': 'location',
        'sku': 'product',
        'unit_price': 'price',
        'qty': 'quantity',
        'cost_price': 'cost',
        'waste_units': 'waste',
        'commodity': 'product',
        'category': 'category',
        'admin1': 'region',
        'admin2': 'state',
        'market': 'location',
        'price': 'price',
        'usdprice': 'price_usd'
    }
    
    # Apply column mapping
    df = df.rename(columns=column_mapping)
    
    # 3. Compute Revenue if missing
    if 'revenue' not in df.columns and 'price' in df.columns and 'quantity' in df.columns:
        df['revenue'] = df['price'] * df['quantity']
        print("    ✓ Computed Revenue as Price × Quantity")
    
    # 4. Handle missing values intelligently
    for col in df.columns:
        if df[col].dtype == 'object':
            # For categorical columns, fill with 'Unknown' or most common value
            if df[col].isnull().sum() > 0:
                most_common = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(most_common)
        else:
            # For numeric columns, fill with 0 or median
            if df[col].isnull().sum() > 0:
                median_val = df[col].median() if not df[col].isnull().all() else 0
                df[col] = df[col].fillna(median_val)
    
    # 5. Convert numeric columns safely
    numeric_cols = ['price', 'quantity', 'revenue', 'cost', 'waste', 'price_usd']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill any remaining NaN values with 0
            df[col] = df[col].fillna(0)
    
    # 6. Parse dates with multiple format support
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Add date features
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            print(f"    ✓ Parsed dates and added date features for {col}")
    
    # 7. Create category if missing but product exists
    if 'category' not in df.columns and 'product' in df.columns:
        # Extract category from product name (first word)
        df['category'] = df['product'].str.split().str[0]
        print("    ✓ Created category from product names")
    
    # 8. Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - len(df)
    if removed_duplicates > 0:
        print(f"    ✓ Removed {removed_duplicates} duplicate rows")
    
    # 9. Filter out rows with missing critical data
    critical_cols = ['product', 'price']
    for col in critical_cols:
        if col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            after = len(df)
            if before != after:
                print(f"    ✓ Removed {before - after} rows with missing {col}")
    
    print(f"    Final shape: {df.shape}")
    print(f"    Final columns: {list(df.columns)}")
    
    return df

def analyze_processed_datasets():
    """Analyze all processed datasets"""
    PROCESSED_DIR = "DATASET/processed"
    
    print("\nAnalyzing processed datasets...")
    
    # Find all processed files
    processed_files = glob.glob(os.path.join(PROCESSED_DIR, "processed_*"))
    
    if not processed_files:
        print("No processed files found!")
        return
    
    analysis_results = []
    
    for file_path in processed_files:
        try:
            # Load processed data
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Basic analysis
            analysis = {
                'file': os.path.basename(file_path),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
            }
            
            analysis_results.append(analysis)
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    # Save analysis report
    analysis_report = {
        'timestamp': datetime.now().isoformat(),
        'total_processed_files': len(analysis_results),
        'analysis': analysis_results
    }
    
    report_path = os.path.join("DATASET/processed/reports", f"dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(analysis_report, f, indent=2, default=str)
    
    print(f"Analysis report saved to: {report_path}")
    
    # Print summary
    print(f"\nDataset Analysis Summary:")
    print(f"Total processed files: {len(analysis_results)}")
    
    if analysis_results:
        total_rows = sum(a['rows'] for a in analysis_results)
        avg_columns = sum(a['columns'] for a in analysis_results) / len(analysis_results)
        print(f"Total rows across all files: {total_rows:,}")
        print(f"Average columns per file: {avg_columns:.1f}")

if __name__ == "__main__":
    # Check if DATASET/raw exists
    if not os.path.exists("DATASET/raw"):
        print("Error: DATASET/raw directory not found!")
        print("Please ensure your datasets are in the DATASET/raw folder")
    else:
        # Process all datasets
        process_datasets_batch()
        
        # Analyze processed datasets
        analyze_processed_datasets()
        
        print("\nBatch processing completed!")
