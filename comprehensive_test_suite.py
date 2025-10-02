#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for PriceOptima
Tests all algorithms, datasets, and report generation
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import time
import subprocess
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class PriceOptimaTestSuite:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = {}
        self.datasets = {}
        self.models = {}
        
    def log_test(self, test_name, status, details=""):
        """Log test results"""
        self.test_results[test_name] = {
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
    
    def test_backend_health(self):
        """Test if backend is running"""
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=5)
            if response.status_code == 200:
                self.log_test("Backend Health Check", "PASS", "Backend is running")
                return True
            else:
                self.log_test("Backend Health Check", "FAIL", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Backend Health Check", "FAIL", f"Connection error: {str(e)}")
            return False
    
    def load_test_datasets(self):
        """Load and validate test datasets"""
        datasets_to_test = [
            "data/processed/merged_input_dataset.csv",
            "data/processed/eda_descriptive.csv", 
            "data/processed/model_comparison.csv",
            "data/raw/wfp_food_prices_nga.csv"
        ]
        
        for dataset_path in datasets_to_test:
            try:
                if os.path.exists(dataset_path):
                    df = pd.read_csv(dataset_path)
                    self.datasets[dataset_path] = {
                        "data": df,
                        "shape": df.shape,
                        "columns": list(df.columns),
                        "dtypes": df.dtypes.to_dict(),
                        "null_counts": df.isnull().sum().to_dict()
                    }
                    self.log_test(f"Dataset Load: {os.path.basename(dataset_path)}", "PASS", 
                                f"Shape: {df.shape}, Columns: {len(df.columns)}")
                else:
                    self.log_test(f"Dataset Load: {os.path.basename(dataset_path)}", "FAIL", "File not found")
            except Exception as e:
                self.log_test(f"Dataset Load: {os.path.basename(dataset_path)}", "FAIL", str(e))
    
    def test_data_upload(self):
        """Test data upload functionality"""
        try:
            # Use the main dataset for testing
            main_dataset = "data/processed/merged_input_dataset.csv"
            if main_dataset not in self.datasets:
                self.log_test("Data Upload", "FAIL", "Main dataset not loaded")
                return False
            
            df = self.datasets[main_dataset]["data"]
            
            # Create a test CSV
            csv_buffer = df.head(100).to_csv(index=False)
            
            # Upload to backend
            files = {'file': ('test_data.csv', csv_buffer, 'text/csv')}
            response = requests.post(f"{self.base_url}/upload", files=files, timeout=30)
            
            if response.status_code == 200:
                upload_data = response.json()
                self.log_test("Data Upload", "PASS", 
                            f"Records: {upload_data['summary']['totalRecords']}")
                return True
            else:
                self.log_test("Data Upload", "FAIL", f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Data Upload", "FAIL", str(e))
            return False
    
    def test_eda_analysis(self):
        """Test EDA analysis algorithms"""
        try:
            response = requests.post(f"{self.base_url}/eda", json={}, timeout=30)
            
            if response.status_code == 200:
                eda_data = response.json()
                
                # Validate EDA results structure
                required_keys = ['overview', 'trends', 'correlations', 'insights', 'recommendations']
                missing_keys = [key for key in required_keys if key not in eda_data]
                
                if not missing_keys:
                    self.log_test("EDA Analysis", "PASS", 
                                f"Categories: {len(eda_data['overview']['category_distribution'])}")
                    return True
                else:
                    self.log_test("EDA Analysis", "FAIL", f"Missing keys: {missing_keys}")
                    return False
            else:
                self.log_test("EDA Analysis", "FAIL", f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("EDA Analysis", "FAIL", str(e))
            return False
    
    def test_ml_algorithms(self):
        """Test all ML algorithms"""
        algorithms = ["linear", "rf", "xgboost"]
        ml_results = {}
        
        for algo in algorithms:
            try:
                response = requests.post(f"{self.base_url}/ml", 
                                       json={"model": algo}, timeout=60)
                
                if response.status_code == 200:
                    ml_data = response.json()
                    
                    # Validate ML results
                    if 'metrics' in ml_data and 'r2' in ml_data['metrics']:
                        r2_score = ml_data['metrics']['r2']
                        rmse = ml_data['metrics']['rmse']
                        mae = ml_data['metrics']['mae']
                        
                        ml_results[algo] = {
                            'r2': r2_score,
                            'rmse': rmse,
                            'mae': mae,
                            'status': 'PASS'
                        }
                        
                        self.log_test(f"ML Algorithm: {algo.upper()}", "PASS", 
                                    f"R²: {r2_score:.3f}, RMSE: {rmse:.3f}")
                    else:
                        self.log_test(f"ML Algorithm: {algo.upper()}", "FAIL", "Invalid metrics")
                        ml_results[algo] = {'status': 'FAIL'}
                else:
                    self.log_test(f"ML Algorithm: {algo.upper()}", "FAIL", f"Status: {response.status_code}")
                    ml_results[algo] = {'status': 'FAIL'}
                    
            except Exception as e:
                self.log_test(f"ML Algorithm: {algo.upper()}", "FAIL", str(e))
                ml_results[algo] = {'status': 'FAIL'}
        
        self.models = ml_results
        return ml_results
    
    def test_rl_simulation(self):
        """Test RL simulation algorithms"""
        algorithms = ["qlearning", "dqn"]
        rl_results = {}
        
        for algo in algorithms:
            try:
                response = requests.post(f"{self.base_url}/rl", 
                                       json={"algorithm": algo}, timeout=30)
                
                if response.status_code == 200:
                    rl_data = response.json()
                    
                    # Validate RL results
                    if 'policy' in rl_data and 'trainingCurve' in rl_data:
                        policy = rl_data['policy']
                        training_curve = rl_data['trainingCurve']
                        
                        rl_results[algo] = {
                            'waste_reduction': policy.get('wasteReduction', 0),
                            'profit_increase': policy.get('profitIncrease', 0),
                            'customer_satisfaction': policy.get('customerSatisfaction', 0),
                            'training_episodes': len(training_curve),
                            'status': 'PASS'
                        }
                        
                        self.log_test(f"RL Algorithm: {algo.upper()}", "PASS", 
                                    f"Waste Reduction: {policy.get('wasteReduction', 0)}%")
                    else:
                        self.log_test(f"RL Algorithm: {algo.upper()}", "FAIL", "Invalid policy data")
                        rl_results[algo] = {'status': 'FAIL'}
                else:
                    self.log_test(f"RL Algorithm: {algo.upper()}", "FAIL", f"Status: {response.status_code}")
                    rl_results[algo] = {'status': 'FAIL'}
                    
            except Exception as e:
                self.log_test(f"RL Algorithm: {algo.upper()}", "FAIL", str(e))
                rl_results[algo] = {'status': 'FAIL'}
        
        return rl_results
    
    def test_report_generation(self):
        """Test all report generation formats"""
        report_types = [
            "summary_report",
            "technical_report", 
            "raw_data",
            "ml_results",
            "rl_policy",
            "visualizations"
        ]
        
        export_results = {}
        
        for report_type in report_types:
            try:
                response = requests.post(f"{self.base_url}/export", 
                                       json={"items": [report_type]}, timeout=30)
                
                if response.status_code == 200:
                    export_data = response.json()
                    
                    if 'files' in export_data and len(export_data['files']) > 0:
                        export_results[report_type] = {
                            'status': 'PASS',
                            'files': export_data['files'],
                            'message': export_data.get('message', '')
                        }
                        self.log_test(f"Report Generation: {report_type}", "PASS", 
                                    f"Files: {len(export_data['files'])}")
                    else:
                        self.log_test(f"Report Generation: {report_type}", "FAIL", "No files generated")
                        export_results[report_type] = {'status': 'FAIL'}
                else:
                    self.log_test(f"Report Generation: {report_type}", "FAIL", f"Status: {response.status_code}")
                    export_results[report_type] = {'status': 'FAIL'}
                    
            except Exception as e:
                self.log_test(f"Report Generation: {report_type}", "FAIL", str(e))
                export_results[report_type] = {'status': 'FAIL'}
        
        return export_results
    
    def test_file_downloads(self):
        """Test file download functionality"""
        try:
            # First generate some reports
            response = requests.post(f"{self.base_url}/export", 
                                   json={"items": ["summary_report", "raw_data"]}, timeout=30)
            
            if response.status_code == 200:
                export_data = response.json()
                files = export_data.get('files', [])
                
                download_results = {}
                
                for file_path in files:
                    filename = os.path.basename(file_path)
                    try:
                        download_response = requests.get(f"{self.base_url}/download/{filename}", timeout=10)
                        
                        if download_response.status_code == 200:
                            download_results[filename] = {
                                'status': 'PASS',
                                'size': len(download_response.content)
                            }
                            self.log_test(f"File Download: {filename}", "PASS", 
                                        f"Size: {len(download_response.content)} bytes")
                        else:
                            download_results[filename] = {'status': 'FAIL'}
                            self.log_test(f"File Download: {filename}", "FAIL", f"Status: {download_response.status_code}")
                    except Exception as e:
                        download_results[filename] = {'status': 'FAIL'}
                        self.log_test(f"File Download: {filename}", "FAIL", str(e))
                
                return download_results
            else:
                self.log_test("File Downloads", "FAIL", "Could not generate files for download")
                return {}
                
        except Exception as e:
            self.log_test("File Downloads", "FAIL", str(e))
            return {}
    
    def analyze_algorithm_performance(self):
        """Analyze and compare algorithm performance"""
        print("\n📊 Algorithm Performance Analysis:")
        print("=" * 50)
        
        # ML Performance Analysis
        if self.models:
            print("\n🤖 Machine Learning Models:")
            ml_df = pd.DataFrame([
                {
                    'Algorithm': algo.upper(),
                    'R² Score': results.get('r2', 0),
                    'RMSE': results.get('rmse', 0),
                    'MAE': results.get('mae', 0),
                    'Status': results.get('status', 'FAIL')
                }
                for algo, results in self.models.items()
            ])
            print(ml_df.to_string(index=False))
            
            # Find best performing model
            if any(results.get('status') == 'PASS' for results in self.models.values()):
                best_model = max(
                    [(algo, results) for algo, results in self.models.items() if results.get('status') == 'PASS'],
                    key=lambda x: x[1].get('r2', 0)
                )
                print(f"\n🏆 Best ML Model: {best_model[0].upper()} (R² = {best_model[1]['r2']:.3f})")
        
        # Dataset Quality Analysis
        if self.datasets:
            print("\n📈 Dataset Quality Analysis:")
            for dataset_name, dataset_info in self.datasets.items():
                print(f"\n{os.path.basename(dataset_name)}:")
                print(f"  Shape: {dataset_info['shape']}")
                print(f"  Columns: {len(dataset_info['columns'])}")
                null_count = sum(dataset_info['null_counts'].values())
                print(f"  Missing Values: {null_count}")
                print(f"  Data Quality: {'Good' if null_count < dataset_info['shape'][0] * 0.1 else 'Needs Attention'}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "passed": len([r for r in self.test_results.values() if r["status"] == "PASS"]),
                "failed": len([r for r in self.test_results.values() if r["status"] == "FAIL"]),
                "timestamp": datetime.now().isoformat()
            },
            "test_results": self.test_results,
            "algorithm_performance": self.models,
            "dataset_analysis": self.datasets
        }
        
        # Save report
        with open("test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Test Report saved to: test_report.json")
        return report
    
    def run_comprehensive_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting Comprehensive PriceOptima Test Suite")
        print("=" * 60)
        
        # Test 1: Backend Health
        if not self.test_backend_health():
            print("❌ Backend not running. Please start the backend first.")
            return False
        
        # Test 2: Load Datasets
        print("\n📊 Loading and validating datasets...")
        self.load_test_datasets()
        
        # Test 3: Data Upload
        print("\n📤 Testing data upload...")
        self.test_data_upload()
        
        # Test 4: EDA Analysis
        print("\n🔍 Testing EDA analysis...")
        self.test_eda_analysis()
        
        # Test 5: ML Algorithms
        print("\n🤖 Testing ML algorithms...")
        self.test_ml_algorithms()
        
        # Test 6: RL Simulation
        print("\n🧠 Testing RL simulation...")
        self.test_rl_simulation()
        
        # Test 7: Report Generation
        print("\n📄 Testing report generation...")
        self.test_report_generation()
        
        # Test 8: File Downloads
        print("\n💾 Testing file downloads...")
        self.test_file_downloads()
        
        # Analysis and Reporting
        self.analyze_algorithm_performance()
        report = self.generate_test_report()
        
        # Summary
        total_tests = report["test_summary"]["total_tests"]
        passed_tests = report["test_summary"]["passed"]
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 Test Suite Complete!")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80  # 80% success rate threshold

def main():
    """Main test execution"""
    test_suite = PriceOptimaTestSuite()
    success = test_suite.run_comprehensive_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the report for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())



