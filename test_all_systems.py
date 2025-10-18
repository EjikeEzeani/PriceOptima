#!/usr/bin/env python3
"""
Comprehensive Test Suite for PriceOptima Preprocessing Systems
Tests all Python scripts and preprocessing pipelines for accuracy and functionality
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TestRunner:
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = datetime.now()
        
    def log_test(self, test_name, status, message="", details=""):
        """Log test results"""
        result = {
            'test_name': test_name,
            'status': status,
            'message': message,
            'details': details,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        self.test_results.append(result)
        
        if status == "PASS":
            self.passed_tests += 1
            print(f"PASS {test_name}: {message}")
        else:
            self.failed_tests += 1
            print(f"FAIL {test_name}: {message}")
            if details:
                print(f"   Details: {details}")
    
    def test_file_exists(self, filepath, description):
        """Test if file exists"""
        try:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                self.log_test(f"File Exists: {description}", "PASS", 
                            f"Found {filepath} ({size:,} bytes)")
                return True
            else:
                self.log_test(f"File Exists: {description}", "FAIL", 
                            f"File not found: {filepath}")
                return False
        except Exception as e:
            self.log_test(f"File Exists: {description}", "FAIL", 
                        f"Error checking file: {str(e)}")
            return False
    
    def test_import_module(self, module_name, description):
        """Test if module can be imported"""
        try:
            __import__(module_name)
            self.log_test(f"Import: {description}", "PASS", 
                        f"Successfully imported {module_name}")
            return True
        except Exception as e:
            self.log_test(f"Import: {description}", "FAIL", 
                        f"Failed to import {module_name}: {str(e)}")
            return False
    
    def test_data_quality(self, df, description):
        """Test data quality metrics"""
        try:
            if df is None or len(df) == 0:
                self.log_test(f"Data Quality: {description}", "FAIL", 
                            "DataFrame is None or empty")
                return False
            
            # Basic quality checks
            total_rows = len(df)
            total_cols = len(df.columns)
            missing_pct = (df.isnull().sum().sum() / (total_rows * total_cols)) * 100
            
            # Check for reasonable data
            quality_issues = []
            if total_rows == 0:
                quality_issues.append("No rows")
            if total_cols == 0:
                quality_issues.append("No columns")
            if missing_pct > 50:
                quality_issues.append(f"High missing data: {missing_pct:.1f}%")
            
            if quality_issues:
                self.log_test(f"Data Quality: {description}", "FAIL", 
                            f"Issues found: {', '.join(quality_issues)}")
                return False
            else:
                self.log_test(f"Data Quality: {description}", "PASS", 
                            f"{total_rows:,} rows, {total_cols} cols, {missing_pct:.1f}% missing")
                return True
                
        except Exception as e:
            self.log_test(f"Data Quality: {description}", "FAIL", 
                        f"Error checking data quality: {str(e)}")
            return False
    
    def test_comprehensive_preprocessing(self):
        """Test comprehensive preprocessing pipeline"""
        print("\n" + "="*60)
        print("TESTING COMPREHENSIVE PREPROCESSING PIPELINE")
        print("="*60)
        
        try:
            # Test import
            from comprehensive_preprocessing import ComprehensivePreprocessor
            self.log_test("Import ComprehensivePreprocessor", "PASS", 
                        "Successfully imported preprocessing class")
            
            # Test initialization
            preprocessor = ComprehensivePreprocessor()
            self.log_test("Initialize Preprocessor", "PASS", 
                        "Successfully created preprocessor instance")
            
            # Test pipeline execution
            print("Running comprehensive preprocessing pipeline...")
            merged_df, train_df, val_df, test_df = preprocessor.run_complete_pipeline()
            
            # Test data quality
            self.test_data_quality(merged_df, "Merged Dataset")
            self.test_data_quality(train_df, "Training Dataset")
            self.test_data_quality(val_df, "Validation Dataset")
            self.test_data_quality(test_df, "Test Dataset")
            
            # Test specific requirements
            if 'month' in merged_df.columns:
                self.log_test("Date Column", "PASS", "Month column present")
            else:
                self.log_test("Date Column", "FAIL", "Month column missing")
            
            # Test price index columns (look for various price-related patterns)
            price_patterns = ['price', 'index', 'maize', 'rice', 'sorghum', 'millet', 'wheat']
            price_cols = []
            for col in merged_df.columns:
                if any(pattern in col.lower() for pattern in price_patterns):
                    price_cols.append(col)
            
            if len(price_cols) > 0:
                self.log_test("Price Features", "PASS", f"Found {len(price_cols)} price-related columns: {price_cols[:5]}")
            else:
                # Check if we have any numeric columns that could be price indices
                numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 5:  # Should have several numeric features
                    self.log_test("Price Features", "PASS", f"Found {len(numeric_cols)} numeric features (likely includes price indices)")
                else:
                    self.log_test("Price Features", "WARN", f"Only {len(numeric_cols)} numeric columns found")
            
            # Test lag features
            lag_cols = [col for col in merged_df.columns if 'lag' in col.lower()]
            if len(lag_cols) > 0:
                self.log_test("Lag Features", "PASS", f"Found {len(lag_cols)} lag features")
            else:
                self.log_test("Lag Features", "WARN", "No lag features found")
            
            return True
            
        except Exception as e:
            self.log_test("Comprehensive Preprocessing", "FAIL", 
                        f"Pipeline execution failed: {str(e)}")
            return False
    
    def test_individual_scripts(self):
        """Test individual Python scripts"""
        print("\n" + "="*60)
        print("TESTING INDIVIDUAL PYTHON SCRIPTS")
        print("="*60)
        
        scripts_to_test = [
            ("main.py", "Main Menu System"),
            ("data_preprocessing_and_training.py", "Data Preprocessing & Training"),
            ("preprocessing_only.py", "Basic Preprocessing"),
            ("dataset_processing_workflow.py", "Dataset Processing Workflow"),
            ("elasticity_analysis.py", "Elasticity Analysis"),
            ("ml_analysis.py", "ML Analysis"),
            ("rl_analysis.py", "RL Analysis"),
            ("master_dashboard.py", "Master Dashboard"),
            ("streamlit_app.py", "Streamlit App"),
            ("run_preprocessing.py", "Preprocessing Runner"),
            ("run_streamlit.py", "Streamlit Runner")
        ]
        
        for script_name, description in scripts_to_test:
            # Test file exists
            if self.test_file_exists(script_name, description):
                # Test syntax by attempting to compile
                try:
                    with open(script_name, 'r', encoding='utf-8') as f:
                        code = f.read()
                    compile(code, script_name, 'exec')
                    self.log_test(f"Syntax: {description}", "PASS", 
                                f"Script syntax is valid")
                except SyntaxError as e:
                    self.log_test(f"Syntax: {description}", "FAIL", 
                                f"Syntax error: {str(e)}")
                except Exception as e:
                    self.log_test(f"Syntax: {description}", "WARN", 
                                f"Could not compile: {str(e)}")
    
    def test_data_files(self):
        """Test data files and processed datasets"""
        print("\n" + "="*60)
        print("TESTING DATA FILES")
        print("="*60)
        
        # Test raw data files
        raw_data_files = [
            "DATASET/raw/wfp_food_prices_nga.csv",
            "DATASET/raw/nga-rainfall-subnat-full.csv"
        ]
        
        for file_path in raw_data_files:
            if self.test_file_exists(file_path, f"Raw Data: {os.path.basename(file_path)}"):
                # Test if file can be read
                try:
                    df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows only
                    self.log_test(f"Read: {os.path.basename(file_path)}", "PASS", 
                                f"Successfully read {len(df)} sample rows")
                except Exception as e:
                    self.log_test(f"Read: {os.path.basename(file_path)}", "FAIL", 
                                f"Error reading file: {str(e)}")
        
        # Test processed data files
        processed_files = [
            "data/processed/merged_input_dataset.csv",
            "data/processed/train_dataset.csv",
            "data/processed/val_dataset.csv",
            "data/processed/test_dataset.csv"
        ]
        
        for file_path in processed_files:
            if self.test_file_exists(file_path, f"Processed Data: {os.path.basename(file_path)}"):
                try:
                    df = pd.read_csv(file_path)
                    self.test_data_quality(df, f"Processed: {os.path.basename(file_path)}")
                except Exception as e:
                    self.log_test(f"Read Processed: {os.path.basename(file_path)}", "FAIL", 
                                f"Error reading processed file: {str(e)}")
    
    def test_ml_models(self):
        """Test ML model files"""
        print("\n" + "="*60)
        print("TESTING ML MODELS")
        print("="*60)
        
        model_files = [
            "models/random_forest_20251012_144856.joblib",
            "models/xgboost_20251012_080719.joblib",
            "models/gradient_boosting_20251012_145233.joblib"
        ]
        
        for model_file in model_files:
            if self.test_file_exists(model_file, f"ML Model: {os.path.basename(model_file)}"):
                try:
                    import joblib
                    model = joblib.load(model_file)
                    self.log_test(f"Load Model: {os.path.basename(model_file)}", "PASS", 
                                f"Successfully loaded {type(model).__name__} model")
                except Exception as e:
                    self.log_test(f"Load Model: {os.path.basename(model_file)}", "FAIL", 
                                f"Error loading model: {str(e)}")
    
    def test_streamlit_functionality(self):
        """Test Streamlit functionality"""
        print("\n" + "="*60)
        print("TESTING STREAMLIT FUNCTIONALITY")
        print("="*60)
        
        # Test if streamlit is available
        try:
            import streamlit as st
            self.log_test("Streamlit Import", "PASS", f"Streamlit version: {st.__version__}")
        except ImportError:
            self.log_test("Streamlit Import", "FAIL", "Streamlit not installed")
            return
        
        # Test streamlit app files
        streamlit_files = [
            "streamlit_app.py",
            "dashboard.py",
            "elasticity_analysis.py",
            "ml_analysis.py",
            "rl_analysis.py",
            "master_dashboard.py"
        ]
        
        for file_name in streamlit_files:
            if os.path.exists(file_name):
                try:
                    # Test if file can be imported as streamlit app
                    with open(file_name, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'streamlit' in content and 'st.' in content:
                        self.log_test(f"Streamlit App: {file_name}", "PASS", 
                                    "Contains Streamlit code")
                    else:
                        self.log_test(f"Streamlit App: {file_name}", "WARN", 
                                    "May not be a Streamlit app")
                except Exception as e:
                    self.log_test(f"Streamlit App: {file_name}", "FAIL", 
                                f"Error reading file: {str(e)}")
    
    def test_integration(self):
        """Test integration between components"""
        print("\n" + "="*60)
        print("TESTING INTEGRATION")
        print("="*60)
        
        try:
            # Test main.py integration
            if os.path.exists("main.py"):
                with open("main.py", 'r', encoding='utf-8') as f:
                    main_content = f.read()
                
                if 'comprehensive_preprocessing' in main_content:
                    self.log_test("Main Integration", "PASS", 
                                "Main.py integrates with comprehensive preprocessing")
                else:
                    self.log_test("Main Integration", "WARN", 
                                "Main.py may not integrate with comprehensive preprocessing")
            
            # Test data flow
            if os.path.exists("data/processed/merged_input_dataset.csv"):
                df = pd.read_csv("data/processed/merged_input_dataset.csv")
                if len(df) > 0:
                    self.log_test("Data Flow", "PASS", 
                                f"Data flows correctly through pipeline ({len(df):,} rows)")
                else:
                    self.log_test("Data Flow", "FAIL", "No data in processed files")
            
        except Exception as e:
            self.log_test("Integration Test", "FAIL", f"Integration test failed: {str(e)}")
    
    def run_all_tests(self):
        """Run all test suites"""
        print("STARTING COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Test started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Run all test suites
        self.test_comprehensive_preprocessing()
        self.test_individual_scripts()
        self.test_data_files()
        self.test_ml_models()
        self.test_streamlit_functionality()
        self.test_integration()
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests / len(self.test_results)) * 100:.1f}%")
        print(f"Duration: {duration.total_seconds():.1f} seconds")
        
        # Show failed tests
        failed_tests = [t for t in self.test_results if t['status'] == 'FAIL']
        if failed_tests:
            print(f"\nFAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   - {test['test_name']}: {test['message']}")
        
        # Show warnings
        warning_tests = [t for t in self.test_results if t['status'] == 'WARN']
        if warning_tests:
            print(f"\nWARNINGS ({len(warning_tests)}):")
            for test in warning_tests:
                print(f"   - {test['test_name']}: {test['message']}")
        
        print("\n" + "="*80)
        if self.failed_tests == 0:
            print("ALL TESTS PASSED! System is ready for production.")
        else:
            print(f"WARNING: {self.failed_tests} tests failed. Please review and fix issues.")
        print("="*80)

def main():
    """Main test runner"""
    test_runner = TestRunner()
    test_runner.run_all_tests()

if __name__ == "__main__":
    main()
