#!/usr/bin/env python3
"""
Quick Preprocessing Runner
Simplified interface to run the comprehensive preprocessing pipeline
"""

import os
import sys
from comprehensive_preprocessing import ComprehensivePreprocessor

def main():
    print("PRICEOPTIMA COMPREHENSIVE PREPROCESSING")
    print("=" * 50)
    print("This script implements the complete preprocessing pipeline")
    print("as specified in your detailed requirements (Parts A-G)")
    print("=" * 50)
    
    # Check if data directory exists
    data_dir = "DATASET/raw"
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory '{data_dir}' not found.")
        print("Creating dummy data for demonstration...")
        os.makedirs(data_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = ComprehensivePreprocessor(data_dir=data_dir)
    
    # Run the complete pipeline
    try:
        print("\nStarting preprocessing pipeline...")
        merged_df, train_df, val_df, test_df = preprocessor.run_complete_pipeline()
        
        print("\nPREPROCESSING SUCCESSFUL!")
        print("=" * 50)
        print("RESULTS SUMMARY:")
        print(f"   - Merged dataset: {len(merged_df):,} rows")
        print(f"   - Training data: {len(train_df):,} rows")
        print(f"   - Validation data: {len(val_df):,} rows")
        print(f"   - Test data: {len(test_df):,} rows")
        print(f"   - Total features: {len(merged_df.columns)}")
        
        print("\nFILES CREATED:")
        output_files = [
            "data/processed/merged_input_dataset.csv",
            "data/processed/train_dataset.csv", 
            "data/processed/val_dataset.csv",
            "data/processed/test_dataset.csv"
        ]
        
        for file in output_files:
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024  # KB
                print(f"   SUCCESS: {file} ({size:.1f} KB)")
            else:
                print(f"   ERROR: {file} (not found)")
        
        print("\nNEXT STEPS:")
        print("   1. Review the processed data files")
        print("   2. Run ML training: python data_preprocessing_and_training.py")
        print("   3. Launch dashboard: python -m streamlit run streamlit_app.py")
        print("   4. Or use the main menu: python main.py")
        
    except Exception as e:
        print(f"\nERROR during preprocessing: {e}")
        print("Please check your data files and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
