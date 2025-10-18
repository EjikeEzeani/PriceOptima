#!/usr/bin/env python3
"""
Comprehensive Data Preprocessing Pipeline for PriceOptima
Implements Parts A-G of the detailed preprocessing specification
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ComprehensivePreprocessor:
    """
    Comprehensive preprocessing pipeline for PriceOptima project
    Implements all steps from Parts A through G
    """
    
    def __init__(self, data_dir="DATASET/raw", output_dir="data/processed"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize scalers and encoders
        self.scalers = {}
        self.encoders = {}
        
    def part_a_common_steps(self, df, file_path):
        """
        Part A — Common first steps (applies to every dataset)
        """
        print("=" * 60)
        print("PART A: COMMON FIRST STEPS")
        print("=" * 60)
        
        # Step 1: Load file and peek at header
        print(f"1. Loading file: {file_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Data types:")
        print(df.dtypes)
        print(f"   First 5 rows:")
        print(df.head())
        
        # Step 2: Fix messed up header rows
        print("\n2. Checking for metadata in first row...")
        if df.iloc[0, 0].startswith('#') or 'date' in str(df.iloc[0, 0]).lower():
            print("   Found metadata in first row, skipping...")
            df = df.iloc[1:].reset_index(drop=True)
            print("   Header row removed")
        
        # Step 3: Parse dates properly
        print("\n3. Parsing dates...")
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            print(f"   Found date column: {date_col}")
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.sort_values(date_col)
            print(f"   Date range: {df[date_col].min()} to {df[date_col].max()}")
        else:
            print("   No date column found")
        
        # Step 4: Fix numeric columns that are strings
        print("\n4. Converting numeric columns...")
        numeric_columns = ['price', 'quantity', 'revenue', 'waste', 'cost', 'usdprice']
        for col in numeric_columns:
            if col in df.columns:
                print(f"   Converting {col}...")
                # Remove commas and convert to numeric
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                )
                print(f"   {col}: {df[col].isna().sum()} missing values")
        
        # Step 5: Handle missing values
        print("\n5. Handling missing values...")
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                print(f"   {col}: {missing_count} missing ({missing_pct:.1f}%)")
                
                if missing_pct > 50:
                    print(f"   Dropping {col} (mostly empty)")
                    df = df.drop(columns=[col])
                elif df[col].dtype in ['int64', 'float64']:
                    # Fill numeric with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"   Filled {col} with median: {median_val}")
                else:
                    # Fill categorical with 'Unknown'
                    df[col] = df[col].fillna('Unknown')
                    print(f"   Filled {col} with 'Unknown'")
        
        # Step 6: Remove obvious outliers
        print("\n6. Removing outliers...")
        initial_rows = len(df)
        
        # Remove negative prices and quantities
        if 'price' in df.columns:
            df = df[df['price'] > 0]
        if 'quantity' in df.columns:
            df = df[df['quantity'] > 0]
        
        # Remove extreme outliers (top 1%)
        for col in ['price', 'quantity', 'revenue']:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                df = df[df[col] <= q99]
        
        removed_rows = initial_rows - len(df)
        print(f"   Removed {removed_rows} outlier rows")
        
        return df
    
    def part_b_price_index(self, wfp_df):
        """
        Part B — Build the Market-Level Price Index (Feature 1)
        """
        print("\n" + "=" * 60)
        print("PART B: BUILDING MARKET-LEVEL PRICE INDEX")
        print("=" * 60)
        
        # Step 1: Filter to staples
        print("1. Filtering to staple commodities...")
        staples = ['Maize', 'Rice (imported)', 'Sorghum', 'Millet', 'Rice', 'Wheat']
        available_commodities = wfp_df['commodity'].unique() if 'commodity' in wfp_df.columns else []
        print(f"   Available commodities: {available_commodities}")
        
        # Find matching commodities
        matching_staples = [s for s in staples if any(s.lower() in c.lower() for c in available_commodities)]
        if not matching_staples:
            print("   No staple commodities found, using all available")
            w = wfp_df.copy()
        else:
            print(f"   Using staples: {matching_staples}")
            w = wfp_df[wfp_df['commodity'].isin(matching_staples)]
        
        print(f"   Filtered to {len(w)} rows")
        
        # Step 2: Convert daily to monthly
        print("\n2. Aggregating to monthly data...")
        if 'date' in w.columns:
            w['month'] = w['date'].dt.to_period('M')
            monthly = w.groupby(['month', 'admin1', 'market', 'commodity'])['price'].mean().reset_index()
            monthly['month'] = monthly['month'].dt.to_timestamp()
            print(f"   Monthly data: {len(monthly)} rows")
        else:
            print("   No date column found, skipping monthly aggregation")
            monthly = w.copy()
        
        # Step 3: Normalize to baseline year (2010)
        print("\n3. Creating price index with 2010 baseline...")
        if 'month' in monthly.columns and 'price' in monthly.columns:
            # Find 2010 data for baseline
            baseline_2010 = monthly[monthly['month'].dt.year == 2010]
            if len(baseline_2010) > 0:
                baseline = baseline_2010.groupby(['admin1', 'market', 'commodity'])['price'].mean()
                monthly = monthly.merge(
                    baseline.reset_index().rename(columns={'price': 'price_base'}),
                    on=['admin1', 'market', 'commodity'],
                    how='left'
                )
                monthly['price_index'] = monthly['price'] / monthly['price_base']
                monthly['price_index'] = monthly['price_index'].fillna(1.0)  # Default to 1.0 if no baseline
                print("   Price index created with 2010 baseline")
            else:
                print("   No 2010 data found, using raw prices as index")
                monthly['price_index'] = monthly['price']
        else:
            monthly['price_index'] = monthly['price'] if 'price' in monthly.columns else 1.0
        
        # Step 4: Pivot to wide format
        print("\n4. Creating wide format for ML models...")
        if 'commodity' in monthly.columns and 'price_index' in monthly.columns:
            price_wide = monthly.pivot_table(
                index=['month', 'admin1', 'market'],
                columns='commodity',
                values='price_index',
                fill_value=1.0
            ).reset_index()
            print(f"   Wide format: {len(price_wide)} rows, {len(price_wide.columns)} columns")
        else:
            price_wide = monthly.copy()
        
        return price_wide
    
    def part_c_demand_waste(self, pos_df):
        """
        Part C — Build Store-Level Demand & Waste (Feature 2)
        """
        print("\n" + "=" * 60)
        print("PART C: BUILDING STORE-LEVEL DEMAND & WASTE")
        print("=" * 60)
        
        if pos_df is None or len(pos_df) == 0:
            print("No POS data available, creating dummy data...")
            # Create dummy POS data
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            dummy_data = []
            for store_id in range(1, 6):
                for sku in ['SKU001', 'SKU002', 'SKU003']:
                    for date in dates[::7]:  # Weekly data
                        dummy_data.append({
                            'timestamp': date,
                            'store_id': f'store_{store_id}',
                            'sku': sku,
                            'qty': np.random.randint(10, 100),
                            'waste_units': np.random.randint(0, 10),
                            'inventory_open': np.random.randint(50, 200),
                            'inventory_close': np.random.randint(50, 200)
                        })
            pos_df = pd.DataFrame(dummy_data)
        
        # Step 1: Parse timestamps and aggregate monthly
        print("1. Parsing timestamps and aggregating monthly...")
        pos_df['date'] = pd.to_datetime(pos_df['timestamp'], errors='coerce')
        pos_df['month'] = pos_df['date'].dt.to_period('M').dt.to_timestamp()
        
        monthly_pos = pos_df.groupby(['month', 'store_id', 'sku']).agg({
            'qty': 'sum',
            'waste_units': 'sum',
            'inventory_open': 'mean',
            'inventory_close': 'mean'
        }).reset_index()
        
        print(f"   Monthly POS data: {len(monthly_pos)} rows")
        
        # Step 2: Compute waste rate and demand
        print("\n2. Computing waste rate and demand...")
        monthly_pos['waste_rate'] = monthly_pos['waste_units'] / (
            monthly_pos['qty'] + monthly_pos['waste_units']
        ).replace(0, np.nan)
        monthly_pos['demand'] = monthly_pos['qty']
        monthly_pos['waste_rate'] = monthly_pos['waste_rate'].fillna(0)
        
        print(f"   Average waste rate: {monthly_pos['waste_rate'].mean():.3f}")
        print(f"   Average demand: {monthly_pos['demand'].mean():.1f}")
        
        # Step 3: Create lag features
        print("\n3. Creating lag features...")
        monthly_pos = monthly_pos.sort_values(['store_id', 'sku', 'month'])
        
        for lag in [1, 2, 3]:
            monthly_pos[f'demand_lag_{lag}'] = monthly_pos.groupby(['store_id', 'sku'])['demand'].shift(lag)
        
        # Step 4: Fill gaps sensibly
        print("\n4. Filling gaps in lag features...")
        lag_columns = ['demand_lag_1', 'demand_lag_2', 'demand_lag_3']
        monthly_pos[lag_columns] = monthly_pos.groupby(['store_id', 'sku'])[lag_columns].fillna(method='ffill')
        
        return monthly_pos
    
    def part_d_enrich_external(self, rainfall_df, cpi_df):
        """
        Part D — Enrich with Rainfall and CPI / Exchange rate
        """
        print("\n" + "=" * 60)
        print("PART D: ENRICHING WITH EXTERNAL DATA")
        print("=" * 60)
        
        # Step 1: Process rainfall data
        print("1. Processing rainfall data...")
        if rainfall_df is not None and len(rainfall_df) > 0:
            if 'date' in rainfall_df.columns:
                rainfall_df['month'] = rainfall_df['date'].dt.to_period('M').dt.to_timestamp()
                rainfall_monthly = rainfall_df.groupby(['month', 'adm_id'])['rfh'].mean().reset_index()
                print(f"   Monthly rainfall: {len(rainfall_monthly)} rows")
            else:
                print("   No date column in rainfall data")
                rainfall_monthly = None
        else:
            print("   No rainfall data available")
            rainfall_monthly = None
        
        # Step 2: Process CPI data
        print("\n2. Processing CPI data...")
        if cpi_df is not None and len(cpi_df) > 0:
            # Convert yearly CPI to monthly
            cpi_long = cpi_df.melt(
                id_vars=['Country Name', 'Country Code', 'Indicator Name'],
                var_name='year',
                value_name='value'
            )
            cpi_long['month'] = pd.to_datetime(cpi_long['year'] + '-01-01')
            cpi_monthly = cpi_long[['month', 'value']].rename(columns={'value': 'cpi'})
            print(f"   Monthly CPI: {len(cpi_monthly)} rows")
        else:
            print("   No CPI data available")
            cpi_monthly = None
        
        return rainfall_monthly, cpi_monthly
    
    def part_e_ml_transforms(self, merged_df):
        """
        Part E — Final model-ready transforms for ML models
        """
        print("\n" + "=" * 60)
        print("PART E: ML-READY TRANSFORMS")
        print("=" * 60)
        
        # Step 1: Encode categorical variables
        print("1. Encoding categorical variables...")
        categorical_columns = ['sku', 'market', 'admin1', 'commodity']
        available_categorical = [col for col in categorical_columns if col in merged_df.columns]
        
        for col in available_categorical:
            if merged_df[col].nunique() < 20:  # One-hot for small cardinality
                dummies = pd.get_dummies(merged_df[col], prefix=col)
                merged_df = pd.concat([merged_df, dummies], axis=1)
                merged_df = merged_df.drop(columns=[col])
                print(f"   One-hot encoded {col}: {len(dummies.columns)} new columns")
            else:  # Label encoding for large cardinality
                le = LabelEncoder()
                merged_df[col] = le.fit_transform(merged_df[col].astype(str))
                self.encoders[col] = le
                print(f"   Label encoded {col}: {merged_df[col].nunique()} unique values")
        
        # Step 2: Scale numeric features
        print("\n2. Scaling numeric features...")
        numeric_columns = merged_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['month', 'year']  # Don't scale time columns
        numeric_columns = [col for col in numeric_columns if not any(exc in col.lower() for exc in exclude_columns)]
        
        if numeric_columns:
            scaler = StandardScaler()
            merged_df[numeric_columns] = scaler.fit_transform(merged_df[numeric_columns])
            self.scalers['numeric'] = scaler
            print(f"   Scaled {len(numeric_columns)} numeric columns")
        
        # Step 3: Train/validation split for time series
        print("\n3. Creating time-based train/validation split...")
        if 'month' in merged_df.columns:
            merged_df = merged_df.sort_values('month')
            train_cutoff = merged_df['month'].quantile(0.7)
            val_cutoff = merged_df['month'].quantile(0.85)
            
            train_df = merged_df[merged_df['month'] <= train_cutoff]
            val_df = merged_df[(merged_df['month'] > train_cutoff) & (merged_df['month'] <= val_cutoff)]
            test_df = merged_df[merged_df['month'] > val_cutoff]
            
            print(f"   Train: {len(train_df)} rows ({train_df['month'].min()} to {train_df['month'].max()})")
            print(f"   Validation: {len(val_df)} rows ({val_df['month'].min()} to {val_df['month'].max()})")
            print(f"   Test: {len(test_df)} rows ({test_df['month'].min()} to {test_df['month'].max()})")
        else:
            # Random split if no time column
            train_df, temp_df = train_test_split(merged_df, test_size=0.3, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            print(f"   Random split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df, merged_df
    
    def part_f_rl_preprocessing(self, merged_df):
        """
        Part F — Extra steps for Reinforcement Learning (RL)
        """
        print("\n" + "=" * 60)
        print("PART F: RL PREPROCESSING")
        print("=" * 60)
        
        # Step 1: Define environment state
        print("1. Defining RL environment state...")
        state_columns = []
        
        # Demand features
        if 'demand' in merged_df.columns:
            state_columns.append('demand')
        for lag in [1, 2, 3]:
            lag_col = f'demand_lag_{lag}'
            if lag_col in merged_df.columns:
                state_columns.append(lag_col)
        
        # Price features
        price_cols = [col for col in merged_df.columns if 'price' in col.lower()]
        state_columns.extend(price_cols[:3])  # Limit to 3 price features
        
        # Other features
        if 'waste_rate' in merged_df.columns:
            state_columns.append('waste_rate')
        if 'inventory_open' in merged_df.columns:
            state_columns.append('inventory_open')
        
        # Ensure we have state columns
        available_state_cols = [col for col in state_columns if col in merged_df.columns]
        if not available_state_cols:
            print("   No suitable state columns found, using all numeric columns")
            available_state_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()[:10]
        
        print(f"   State columns: {available_state_cols}")
        
        # Step 2: Normalize state for RL stability
        print("\n2. Normalizing state for RL stability...")
        if available_state_cols:
            rl_scaler = StandardScaler()
            merged_df[available_state_cols] = rl_scaler.fit_transform(merged_df[available_state_cols])
            self.scalers['rl_state'] = rl_scaler
            print(f"   Normalized {len(available_state_cols)} state columns")
        
        # Step 3: Define action space (price multiplier)
        print("\n3. Defining action space...")
        # Create discrete action space: [0.9x, 1.0x, 1.1x, 1.2x]
        action_space = [0.9, 1.0, 1.1, 1.2]
        print(f"   Action space: {action_space} (price multipliers)")
        
        # Step 4: Design reward function
        print("\n4. Designing reward function...")
        if 'demand' in merged_df.columns and 'price' in merged_df.columns:
            # Simple reward: revenue - waste penalty
            merged_df['revenue'] = merged_df['demand'] * merged_df['price']
            merged_df['waste_penalty'] = merged_df.get('waste_rate', 0) * merged_df['demand'] * 2
            merged_df['reward'] = merged_df['revenue'] - merged_df['waste_penalty']
            merged_df['reward'] = np.clip(merged_df['reward'], -1000, 1000)
            print("   Reward function: revenue - waste_penalty (clipped to [-1000, 1000])")
        else:
            merged_df['reward'] = 0
            print("   No suitable columns for reward, setting to 0")
        
        return merged_df, available_state_cols, action_space
    
    def part_g_validation(self, train_df, val_df, test_df):
        """
        Part G — Validation & backtesting
        """
        print("\n" + "=" * 60)
        print("PART G: VALIDATION & BACKTESTING")
        print("=" * 60)
        
        # Step 1: Check for data leakage
        print("1. Checking for data leakage...")
        if 'month' in train_df.columns and 'month' in val_df.columns:
            max_train_date = train_df['month'].max()
            min_val_date = val_df['month'].min()
            if max_train_date >= min_val_date:
                print("   WARNING: Potential data leakage detected!")
                print(f"   Max train date: {max_train_date}")
                print(f"   Min val date: {min_val_date}")
            else:
                print("   SUCCESS: No data leakage detected")
        
        # Step 2: Validate scalers
        print("\n2. Validating scalers...")
        for scaler_name, scaler in self.scalers.items():
            print(f"   {scaler_name}: {type(scaler).__name__}")
        
        # Step 3: Check data quality
        print("\n3. Checking data quality...")
        for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            print(f"   {name}: {len(df)} rows, {len(df.columns)} columns")
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            print(f"   Missing values: {missing_pct:.2f}%")
        
        return True
    
    def run_complete_pipeline(self):
        """
        Run the complete preprocessing pipeline
        """
        print("STARTING COMPREHENSIVE PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # Load data files
        print("Loading data files...")
        wfp_df = None
        pos_df = None
        rainfall_df = None
        cpi_df = None
        
        # Try to load WFP food prices
        wfp_paths = [
            os.path.join(self.data_dir, "wfp_food_prices_nga.csv"),
            os.path.join(self.data_dir, "wfp_food_prices.csv")
        ]
        
        for path in wfp_paths:
            if os.path.exists(path):
                print(f"Loading WFP data from: {path}")
                wfp_df = pd.read_csv(path, low_memory=False)
                break
        
        if wfp_df is None:
            print("No WFP data found, creating dummy data...")
            # Create dummy WFP data
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            dummy_wfp = []
            for date in dates[::7]:  # Weekly data
                for commodity in ['Maize', 'Rice', 'Wheat']:
                    for market in ['Lagos', 'Abuja', 'Kano']:
                        dummy_wfp.append({
                            'date': date,
                            'admin1': 'Lagos',
                            'market': market,
                            'commodity': commodity,
                            'price': np.random.uniform(100, 500),
                            'quantity': np.random.randint(50, 200)
                        })
            wfp_df = pd.DataFrame(dummy_wfp)
        
        # Apply Part A to WFP data
        wfp_df = self.part_a_common_steps(wfp_df, "WFP data")
        
        # Apply Part B to create price index
        price_wide = self.part_b_price_index(wfp_df)
        
        # Apply Part C to create demand/waste features
        monthly_pos = self.part_c_demand_waste(pos_df)
        
        # Apply Part D to enrich with external data
        rainfall_monthly, cpi_monthly = self.part_d_enrich_external(rainfall_df, cpi_df)
        
        # Merge everything
        print("\n" + "=" * 60)
        print("MERGING ALL DATASETS")
        print("=" * 60)
        
        merged_df = price_wide.copy()
        
        if monthly_pos is not None and len(monthly_pos) > 0:
            merged_df = merged_df.merge(
                monthly_pos,
                left_on='month',
                right_on='month',
                how='left'
            )
            print(f"   Merged with POS data: {len(merged_df)} rows")
        
        if rainfall_monthly is not None:
            merged_df = merged_df.merge(
                rainfall_monthly,
                left_on=['month', 'admin1'],
                right_on=['month', 'adm_id'],
                how='left'
            )
            print(f"   Merged with rainfall data: {len(merged_df)} rows")
        
        if cpi_monthly is not None:
            merged_df = merged_df.merge(cpi_monthly, on='month', how='left')
            print(f"   Merged with CPI data: {len(merged_df)} rows")
        
        # Apply Part E for ML transforms
        train_df, val_df, test_df, merged_df = self.part_e_ml_transforms(merged_df)
        
        # Apply Part F for RL preprocessing
        merged_df, state_columns, action_space = self.part_f_rl_preprocessing(merged_df)
        
        # Apply Part G for validation
        self.part_g_validation(train_df, val_df, test_df)
        
        # Save processed data
        print("\n" + "=" * 60)
        print("SAVING PROCESSED DATA")
        print("=" * 60)
        
        merged_df.to_csv(os.path.join(self.output_dir, "merged_input_dataset.csv"), index=False)
        train_df.to_csv(os.path.join(self.output_dir, "train_dataset.csv"), index=False)
        val_df.to_csv(os.path.join(self.output_dir, "val_dataset.csv"), index=False)
        test_df.to_csv(os.path.join(self.output_dir, "test_dataset.csv"), index=False)
        
        print(f"Complete pipeline finished!")
        print(f"   Final merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        print(f"   Train: {len(train_df)} rows")
        print(f"   Validation: {len(val_df)} rows")
        print(f"   Test: {len(test_df)} rows")
        print(f"   State columns for RL: {state_columns}")
        print(f"   Action space: {action_space}")
        
        return merged_df, train_df, val_df, test_df

def main():
    """Main function to run the preprocessing pipeline"""
    preprocessor = ComprehensivePreprocessor()
    merged_df, train_df, val_df, test_df = preprocessor.run_complete_pipeline()
    
    print("\nPREPROCESSING COMPLETE!")
    print("Files saved in data/processed/ directory")

if __name__ == "__main__":
    main()
