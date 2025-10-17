#!/usr/bin/env python3
"""
Dataset Processing Workflow for PriceOptima
This script processes datasets from the DATASET/raw folder and outputs processed data to DATASET/processed
"""

import pandas as pd
import numpy as np
import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import glob

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class DatasetConfig:
    """Configuration for dataset processing"""
    
    # Input and output directories
    RAW_DATA_DIR = "DATASET/raw"
    PROCESSED_DATA_DIR = "DATASET/processed"
    MODELS_DIR = "models"
    REPORTS_DIR = "DATASET/processed/reports"
    
    # File patterns to process
    SUPPORTED_EXTENSIONS = ['.csv', '.parquet', '.xlsx', '.xls']
    
    # Column mapping configuration
    REQUIRED_COLUMNS_SYNONYMS = {
        "Date": ["date", "datetime", "timestamp", "time"],
        "Product": ["product", "product name", "product_name", "item", "item name", "product_id"],
        "Category": ["category", "type", "product_category", "cat"],
        "Price": ["price", "unit price", "unit_price", "cost_per_unit", "selling_price"],
        "Quantity": ["quantity", "quantity sold", "qty", "qty sold", "units", "volume"],
        "Revenue": ["revenue", "sales", "total", "total revenue", "total_sales", "sales_amount"]
    }
    
    OPTIONAL_COLUMNS_SYNONYMS = {
        "Waste": ["waste", "waste amount", "waste_amount", "spoilage", "loss"],
        "Cost": ["cost", "unit cost", "unit_cost", "purchase_cost", "cost_price"],
        "Supplier": ["supplier", "vendor", "supplier_name", "source"],
        "Location": ["location", "store", "branch", "outlet", "region"],
        "Season": ["season", "period", "quarter", "month_name"]
    }
    
    # Processing parameters
    MAX_FILE_SIZE_MB = 100
    MIN_ROWS = 10
    MAX_MISSING_PERCENTAGE = 0.5

class DatasetProcessor:
    """Main dataset processing class"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.processing_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_rows_processed': 0,
            'processing_errors': []
        }
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary output directories"""
        directories = [
            self.config.PROCESSED_DATA_DIR,
            self.config.REPORTS_DIR,
            self.config.MODELS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize column names for matching"""
        return re.sub(r"\s+", " ", str(name).strip().lower())
    
    def _find_matching_column(self, target_name: str, available_columns: List[str]) -> Optional[str]:
        """Find matching column name using synonyms"""
        target_lower = self._normalize_column_name(target_name)
        synonyms = self.config.REQUIRED_COLUMNS_SYNONYMS.get(target_name, []) + \
                  self.config.OPTIONAL_COLUMNS_SYNONYMS.get(target_name, [])
        
        candidates = [target_lower] + [self._normalize_column_name(syn) for syn in synonyms]
        
        for col in available_columns:
            col_normalized = self._normalize_column_name(col)
            if col_normalized in candidates:
                return col
        
        return None
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using mapping configuration"""
        df = df.copy()
        available_columns = list(df.columns)
        rename_map = {}
        
        # Map required columns
        for standard_name in self.config.REQUIRED_COLUMNS_SYNONYMS.keys():
            matching_col = self._find_matching_column(standard_name, available_columns)
            if matching_col:
                rename_map[matching_col] = standard_name
        
        # Map optional columns
        for standard_name in self.config.OPTIONAL_COLUMNS_SYNONYMS.keys():
            matching_col = self._find_matching_column(standard_name, available_columns)
            if matching_col:
                rename_map[matching_col] = standard_name
        
        df = df.rename(columns=rename_map)
        
        # Validate required columns
        missing_required = [col for col in self.config.REQUIRED_COLUMNS_SYNONYMS.keys() 
                           if col not in df.columns]
        
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        return df
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date columns with multiple format support"""
        df = df.copy()
        
        if "Date" in df.columns:
            # Try multiple date formats
            date_formats = [
                '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S',
                '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y %H:%M:%S'
            ]
            
            for fmt in date_formats:
                try:
                    df["Date"] = pd.to_datetime(df["Date"], format=fmt, errors='coerce')
                    if not df["Date"].isna().all():
                        break
                except:
                    continue
            
            # If all formats failed, use pandas' automatic parsing
            if df["Date"].isna().all():
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        
        return df
    
    def coerce_numeric(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Convert specified columns to numeric type"""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type"""
        df = df.copy()
        
        # Fill numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill categorical columns with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns and col not in ['Date']:
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived features like Revenue if missing"""
        df = df.copy()
        
        # Compute Revenue if missing or all NaN
        if "Revenue" not in df.columns or df["Revenue"].isna().all():
            if "Price" in df.columns and "Quantity" in df.columns:
                df["Revenue"] = df["Price"] * df["Quantity"]
                logger.info("Computed Revenue as Price × Quantity")
        
        # Add date-based features
        if "Date" in df.columns:
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["Day"] = df["Date"].dt.day
            df["DayOfWeek"] = df["Date"].dt.dayofweek
            df["Quarter"] = df["Date"].dt.quarter
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and return quality metrics"""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            quality_report['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns:
                quality_report['categorical_summary'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].value_counts().head(5).to_dict()
                }
        
        return quality_report
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        logger.info("Starting dataset preprocessing...")
        
        # Step 1: Standardize columns
        logger.info("1. Standardizing column names...")
        df = self.standardize_columns(df)
        
        # Step 2: Parse dates
        logger.info("2. Parsing date columns...")
        df = self.parse_dates(df)
        
        # Step 3: Convert numeric columns
        logger.info("3. Converting numeric columns...")
        numeric_cols = ["Price", "Quantity", "Revenue", "Waste", "Cost"]
        df = self.coerce_numeric(df, numeric_cols)
        
        # Step 4: Handle missing values
        logger.info("4. Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Step 5: Compute derived features
        logger.info("5. Computing derived features...")
        df = self.compute_derived_features(df)
        
        # Step 6: Remove duplicates
        logger.info("6. Removing duplicate rows...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            logger.info(f"   Removed {removed_duplicates} duplicate rows")
        
        logger.info(f"Preprocessing completed! Final dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def process_single_file(self, file_path: str) -> Dict:
        """Process a single dataset file"""
        file_info = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'status': 'failed',
            'rows_before': 0,
            'rows_after': 0,
            'processing_time': 0,
            'quality_report': {},
            'error': None
        }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Load data
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            file_info['rows_before'] = len(df)
            
            # Validate minimum rows
            if len(df) < self.config.MIN_ROWS:
                raise ValueError(f"Dataset too small: {len(df)} rows (minimum: {self.config.MIN_ROWS})")
            
            # Preprocess data
            processed_df = self.preprocess_dataset(df)
            file_info['rows_after'] = len(processed_df)
            
            # Validate data quality
            quality_report = self.validate_data_quality(processed_df)
            file_info['quality_report'] = quality_report
            
            # Save processed data
            output_filename = f"processed_{os.path.basename(file_path)}"
            output_path = os.path.join(self.config.PROCESSED_DATA_DIR, output_filename)
            
            if file_ext == '.parquet':
                processed_df.to_parquet(output_path, index=False)
            else:
                processed_df.to_csv(output_path, index=False)
            
            file_info['output_path'] = output_path
            file_info['status'] = 'success'
            
            # Update statistics
            self.processing_stats['files_processed'] += 1
            self.processing_stats['total_rows_processed'] += file_info['rows_after']
            
            logger.info(f"Successfully processed {file_path} -> {output_path}")
            
        except Exception as e:
            file_info['error'] = str(e)
            file_info['status'] = 'failed'
            self.processing_stats['files_failed'] += 1
            self.processing_stats['processing_errors'].append({
                'file': file_path,
                'error': str(e)
            })
            logger.error(f"Failed to process {file_path}: {e}")
        
        finally:
            file_info['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return file_info
    
    def process_all_datasets(self) -> Dict:
        """Process all datasets in the raw data directory"""
        logger.info("Starting batch processing of all datasets...")
        
        # Find all supported files
        raw_files = []
        for ext in self.config.SUPPORTED_EXTENSIONS:
            pattern = os.path.join(self.config.RAW_DATA_DIR, f"*{ext}")
            raw_files.extend(glob.glob(pattern))
        
        logger.info(f"Found {len(raw_files)} files to process")
        
        # Process each file
        processing_results = []
        for file_path in raw_files:
            result = self.process_single_file(file_path)
            processing_results.append(result)
        
        # Generate summary report
        summary_report = self._generate_summary_report(processing_results)
        
        # Save processing report
        report_path = os.path.join(self.config.REPORTS_DIR, f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump({
                'summary': summary_report,
                'file_results': processing_results,
                'processing_stats': self.processing_stats
            }, f, indent=2, default=str)
        
        logger.info(f"Processing completed! Report saved to: {report_path}")
        return summary_report
    
    def _generate_summary_report(self, processing_results: List[Dict]) -> Dict:
        """Generate summary report of processing results"""
        successful_files = [r for r in processing_results if r['status'] == 'success']
        failed_files = [r for r in processing_results if r['status'] == 'failed']
        
        total_rows_before = sum(r['rows_before'] for r in processing_results)
        total_rows_after = sum(r['rows_after'] for r in successful_files)
        
        return {
            'total_files': len(processing_results),
            'successful_files': len(successful_files),
            'failed_files': len(failed_files),
            'success_rate': len(successful_files) / len(processing_results) if processing_results else 0,
            'total_rows_before': total_rows_before,
            'total_rows_after': total_rows_after,
            'rows_removed': total_rows_before - total_rows_after,
            'processing_errors': [r['error'] for r in failed_files if r['error']]
        }

def main():
    """Main function to run the dataset processing workflow"""
    print("PriceOptima Dataset Processing Workflow")
    print("=" * 50)
    
    # Initialize configuration and processor
    config = DatasetConfig()
    processor = DatasetProcessor(config)
    
    # Check if raw data directory exists
    if not os.path.exists(config.RAW_DATA_DIR):
        print(f"Error: Raw data directory '{config.RAW_DATA_DIR}' not found!")
        print("Please ensure your datasets are in the DATASET/raw folder")
        return
    
    # Process all datasets
    try:
        summary = processor.process_all_datasets()
        
        print("\nProcessing Summary:")
        print(f"- Total files: {summary['total_files']}")
        print(f"- Successful: {summary['successful_files']}")
        print(f"- Failed: {summary['failed_files']}")
        print(f"- Success rate: {summary['success_rate']:.2%}")
        print(f"- Total rows processed: {summary['total_rows_after']:,}")
        
        if summary['failed_files'] > 0:
            print(f"\nFailed files:")
            for error in summary['processing_errors']:
                print(f"  - {error}")
        
        print(f"\nProcessed files saved to: {config.PROCESSED_DATA_DIR}")
        print(f"Processing report saved to: {config.REPORTS_DIR}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
