#!/usr/bin/env python3
"""
🎯 PriceOptima Main Application
Interactive menu-driven system for data processing, ML training, and dashboard launch
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print application banner"""
    print("=" * 80)
    print("🎯 PRICEOPTIMA - DYNAMIC PRICING OPTIMIZATION PLATFORM")
    print("=" * 80)
    print("📊 MSc Project - Advanced Analytics for Food Price Optimization")
    print("🤖 Machine Learning | 🎮 Reinforcement Learning | 📈 Elasticity Analysis")
    print("=" * 80)

def print_menu():
    """Print main menu options"""
    print("\n📌 Select Stage to Run:")
    print("1. Merge Data")
    print("2. Preprocess Data") 
    print("3. Train ML / DQN")
    print("4. Evaluate Model")
    print("5. Launch Dashboard (Streamlit)")
    print("6. Run All")
    print("0. Exit")
    print("-" * 40)

def merge_data():
    """Merge raw datasets"""
    print("🔄 Merging raw datasets...")
    
    try:
        # Check for raw data files
        raw_data_paths = [
            "DATASET/raw/wfp_food_prices_nga.csv",
            "DATASET/raw/wfp_food_prices.csv",
            "DATASET/raw/nga-rainfall-subnat-full.csv"
        ]
        
        available_files = []
        for path in raw_data_paths:
            if os.path.exists(path):
                available_files.append(path)
                print(f"   ✅ Found: {path}")
        
        if not available_files:
            print("   ⚠️ No raw data files found. Creating dummy dataset...")
            create_dummy_dataset()
            return True
        
        # Create processed directory
        os.makedirs("data/processed", exist_ok=True)
        
        # Load and merge datasets
        merged_df = None
        
        for file_path in available_files:
            try:
                df = pd.read_csv(file_path)
                print(f"   📊 Loading {file_path}: {len(df)} rows")
                
                if merged_df is None:
                    merged_df = df
                else:
                    # Simple merge - add rows
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
                    
            except Exception as e:
                print(f"   ⚠️ Error loading {file_path}: {e}")
                continue
        
        if merged_df is not None:
            # Save merged dataset
            output_path = "data/processed/merged_input_dataset.csv"
            merged_df.to_csv(output_path, index=False)
            print(f"✅ Data merged → {output_path}")
            print(f"   📊 Total records: {len(merged_df):,}")
            print(f"   📋 Columns: {list(merged_df.columns)}")
            return True
        else:
            print("❌ Failed to merge data")
            return False
            
    except Exception as e:
        print(f"❌ Error during data merging: {e}")
        return False

def create_dummy_dataset():
    """Create dummy dataset for demonstration"""
    print("   🔧 Creating dummy dataset...")
    
    # Create processed directory
    os.makedirs("data/processed", exist_ok=True)
    
    # Generate realistic dummy data
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    commodities = np.random.choice(['Rice', 'Wheat', 'Maize', 'Beans', 'Tomatoes', 'Onions'], n_samples)
    states = np.random.choice(['Lagos', 'Abuja', 'Kano', 'Rivers', 'Ogun', 'Kaduna'], n_samples)
    
    base_prices = {
        'Rice': 200, 'Wheat': 150, 'Maize': 100, 
        'Beans': 300, 'Tomatoes': 250, 'Onions': 180
    }
    
    prices = []
    quantities = []
    
    for commodity in commodities:
        base_price = base_prices[commodity]
        # Add seasonality and trend
        seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
        trend = 0.02 * np.arange(n_samples) / n_samples
        noise = np.random.normal(0, 0.05)
        
        price = base_price * (1 + seasonal + trend + noise)
        quantity = 1000 - (price - base_price) * 2 + np.random.normal(0, 50)
        quantity = max(0, quantity)
        
        prices.append(price)
        quantities.append(quantity)
    
    # Create DataFrame
    dummy_df = pd.DataFrame({
        'date': dates,
        'commodity': commodities,
        'state': states,
        'price': prices,
        'quantity': quantities,
        'revenue': np.array(prices) * np.array(quantities)
    })
    
    # Save dummy dataset
    output_path = "data/processed/merged_input_dataset.csv"
    dummy_df.to_csv(output_path, index=False)
    print(f"   ✅ Dummy dataset created → {output_path}")
    print(f"   📊 Records: {len(dummy_df):,}")

def preprocess_data():
    """Preprocess the merged dataset"""
    print("🔄 Preprocessing dataset...")
    
    try:
        # Load merged data
        input_path = "data/processed/merged_input_dataset.csv"
        if not os.path.exists(input_path):
            print("   ❌ Merged dataset not found. Please run 'Merge Data' first.")
            return False
        
        df = pd.read_csv(input_path)
        print(f"   📊 Loading data: {len(df)} rows, {len(df.columns)} columns")
        
        # Basic preprocessing
        df_processed = df.copy()
        
        # Handle date column
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
            df_processed['year'] = df_processed['date'].dt.year
            df_processed['month'] = df_processed['date'].dt.month
            df_processed['day'] = df_processed['date'].dt.day
            df_processed['dayofweek'] = df_processed['date'].dt.dayofweek
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(0)
        
        # Handle categorical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'date':
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        # Save processed data
        output_path = "data/processed/processed_dataset.csv"
        df_processed.to_csv(output_path, index=False)
        
        print(f"✅ Preprocessing complete → {output_path}")
        print(f"   📊 Processed records: {len(df_processed):,}")
        print(f"   📋 New columns: {len(df_processed.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return False

def train_models():
    """Train ML and RL models"""
    print("🔄 Training ML / DQN models...")
    
    try:
        # Import training functions
        from data_preprocessing_and_training import train_ml_model, preprocess_data as prep_data
        
        # Load processed data
        input_path = "data/processed/processed_dataset.csv"
        if not os.path.exists(input_path):
            input_path = "data/processed/merged_input_dataset.csv"
        
        if not os.path.exists(input_path):
            print("   ❌ Processed dataset not found. Please run preprocessing first.")
            return False
        
        df = pd.read_csv(input_path)
        print(f"   📊 Training on {len(df)} records")
        
        # Preprocess for ML
        df_ml = prep_data(df)
        
        # Train different ML models
        models_to_train = ['random_forest', 'gradient_boosting', 'linear_regression']
        trained_models = {}
        
        for model_type in models_to_train:
            print(f"   🤖 Training {model_type}...")
            result = train_ml_model(df_ml, model_type)
            
            if 'error' not in result:
                trained_models[model_type] = result
                print(f"      ✅ {model_type}: R² = {result['r2_score']:.4f}")
            else:
                print(f"      ❌ {model_type}: {result['error']}")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Save model results
        results_path = "data/processed/training_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(trained_models, f, indent=2, default=str)
        
        print(f"✅ Models trained → {len(trained_models)} models saved")
        print(f"   📊 Results saved → {results_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

def evaluate_models():
    """Evaluate trained models"""
    print("🔄 Evaluating models...")
    
    try:
        # Load training results
        results_path = "data/processed/training_results.json"
        if not os.path.exists(results_path):
            print("   ❌ Training results not found. Please run 'Train ML / DQN' first.")
            return False
        
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print("   📊 Model Evaluation Results:")
        print("   " + "-" * 50)
        
        for model_name, result in results.items():
            if 'error' not in result:
                print(f"   🤖 {model_name.upper()}:")
                print(f"      R² Score: {result['r2_score']:.4f}")
                print(f"      MSE: {result['mse']:.4f}")
                print(f"      MAE: {result['mae']:.4f}")
                print(f"      MAPE: {result['mape']:.4f}%")
                print()
        
        # Find best model
        best_model = None
        best_r2 = -1
        
        for model_name, result in results.items():
            if 'error' not in result and result['r2_score'] > best_r2:
                best_r2 = result['r2_score']
                best_model = model_name
        
        if best_model:
            print(f"   🏆 Best Model: {best_model} (R² = {best_r2:.4f})")
        
        print("✅ Evaluation complete")
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return False

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("🚀 Launching Streamlit dashboard...")
    
    try:
        # Check for available dashboards
        dashboard_files = [
            "streamlit_app.py",  # Comprehensive dashboard
            "master_dashboard.py",  # Master dashboard
            "dashboard.py",  # Original dashboard
            "elasticity_analysis.py",  # Elasticity analysis
            "ml_analysis.py",  # ML analysis
            "rl_analysis.py"  # RL analysis
        ]
        
        available_dashboards = []
        for file in dashboard_files:
            if os.path.exists(file):
                available_dashboards.append(file)
        
        if not available_dashboards:
            print("   ❌ No dashboard files found!")
            return False
        
        print("   📊 Available dashboards:")
        for i, dashboard in enumerate(available_dashboards, 1):
            print(f"      {i}. {dashboard}")
        
        # Select dashboard
        if len(available_dashboards) == 1:
            selected_dashboard = available_dashboards[0]
            print(f"   🎯 Auto-selecting: {selected_dashboard}")
        else:
            try:
                choice = input("   👉 Select dashboard (1-{}): ".format(len(available_dashboards)))
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_dashboards):
                    selected_dashboard = available_dashboards[choice_idx]
                else:
                    selected_dashboard = available_dashboards[0]
            except:
                selected_dashboard = available_dashboards[0]
        
        print(f"   🚀 Launching: {selected_dashboard}")
        print(f"   🌐 Dashboard will open at: http://localhost:8501")
        print("   🛑 Press Ctrl+C to stop the dashboard")
        print("-" * 60)
        
        # Launch Streamlit
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                selected_dashboard, "--server.port", "8501"
            ], check=True)
        except KeyboardInterrupt:
            print("\n   🛑 Dashboard stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error launching dashboard: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False

def run_all():
    """Run all stages in sequence"""
    print("🚀 Running all stages...")
    print("=" * 60)
    
    stages = [
        ("Merge Data", merge_data),
        ("Preprocess Data", preprocess_data),
        ("Train ML / DQN", train_models),
        ("Evaluate Model", evaluate_models),
        ("Launch Dashboard", launch_dashboard)
    ]
    
    for stage_name, stage_func in stages:
        print(f"\n📌 Stage: {stage_name}")
        print("-" * 40)
        
        success = stage_func()
        
        if success:
            print(f"✅ {stage_name} completed successfully")
        else:
            print(f"❌ {stage_name} failed")
            print("🛑 Stopping execution due to failure")
            return False
        
        time.sleep(1)  # Brief pause between stages
    
    print("\n🎉 All stages completed successfully!")
    return True

def main():
    """Main application loop"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("👉 Enter choice: ").strip()
            
            if choice == "0":
                print("\n👋 Thank you for using PriceOptima!")
                print("🎯 Goodbye!")
                break
            
            elif choice == "1":
                print("\n📊 STAGE 1: MERGE DATA")
                print("-" * 40)
                merge_data()
            
            elif choice == "2":
                print("\n🔧 STAGE 2: PREPROCESS DATA")
                print("-" * 40)
                preprocess_data()
            
            elif choice == "3":
                print("\n🤖 STAGE 3: TRAIN ML / DQN")
                print("-" * 40)
                train_models()
            
            elif choice == "4":
                print("\n📊 STAGE 4: EVALUATE MODEL")
                print("-" * 40)
                evaluate_models()
            
            elif choice == "5":
                print("\n🚀 STAGE 5: LAUNCH DASHBOARD")
                print("-" * 40)
                launch_dashboard()
            
            elif choice == "6":
                print("\n🎯 STAGE 6: RUN ALL")
                print("-" * 40)
                run_all()
            
            else:
                print("❌ Invalid choice. Please enter 0-6.")
            
            print("\n" + "=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Application interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
