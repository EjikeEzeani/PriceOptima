#!/usr/bin/env python3
"""
Comprehensive Test Suite for Working Backend
Tests all endpoints and ensures seamless module transitions
"""

import requests
import json
import time
import os
import pandas as pd
import io

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
TEST_DATA_PATH = "data/processed/dummy_dashboard.csv"

def create_test_data():
    """Create test data if it doesn't exist"""
    if not os.path.exists(TEST_DATA_PATH):
        os.makedirs(os.path.dirname(TEST_DATA_PATH), exist_ok=True)
        
        # Generate sample data
        data = {
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'product': ['Tomatoes', 'Onions', 'Peppers', 'Carrots'] * 25,
            'category': ['Vegetables', 'Vegetables', 'Vegetables', 'Vegetables'] * 25,
            'price': [2.50, 1.80, 3.20, 2.10] * 25 + [0.1, -0.05, 0.15, -0.08] * 25,
            'quantity': [50, 75, 30, 60] * 25 + [5, -3, 8, -4] * 25,
            'revenue': [125, 135, 96, 126] * 25 + [12, -5, 25, -8] * 25,
            'waste': [5, 8, 3, 6] * 25 + [1, -1, 2, -1] * 25
        }
        
        df = pd.DataFrame(data)
        df.to_csv(TEST_DATA_PATH, index=False)
        print(f"Created test data: {TEST_DATA_PATH}")
    
    return TEST_DATA_PATH

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        print("✓ Health check passed")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_upload_data():
    """Test data upload endpoint"""
    print("Testing data upload...")
    try:
        # Create test data
        test_file = create_test_data()
        
        with open(test_file, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["summary"]["totalRecords"] > 0
        print("✓ Data upload passed")
        return True
    except Exception as e:
        print(f"✗ Data upload failed: {e}")
        return False

def test_eda_analysis():
    """Test EDA analysis endpoint"""
    print("Testing EDA analysis...")
    try:
        response = requests.post(f"{BASE_URL}/eda")
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "overview" in data
        assert "trends" in data
        assert "correlations" in data
        assert "insights" in data
        assert "recommendations" in data
        
        print("✓ EDA analysis passed")
        return True
    except Exception as e:
        print(f"✗ EDA analysis failed: {e}")
        return False

def test_ml_training():
    """Test ML training endpoint"""
    print("Testing ML training...")
    try:
        # Test Random Forest
        payload = {"model": "random_forest"}
        response = requests.post(f"{BASE_URL}/ml", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "modelId" in data
        assert "metrics" in data
        assert "predictions" in data
        assert "featureImportance" in data
        
        # Test Linear Regression
        payload = {"model": "linear_regression"}
        response = requests.post(f"{BASE_URL}/ml", json=payload)
        assert response.status_code == 200
        
        print("✓ ML training passed")
        return True
    except Exception as e:
        print(f"✗ ML training failed: {e}")
        return False

def test_rl_simulation():
    """Test RL simulation endpoint"""
    print("Testing RL simulation...")
    try:
        payload = {"algorithm": "dqn"}
        response = requests.post(f"{BASE_URL}/rl", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "algorithm" in data
        assert "finalReward" in data
        assert "policy" in data
        assert "trainingCurve" in data
        
        print("✓ RL simulation passed")
        return True
    except Exception as e:
        print(f"✗ RL simulation failed: {e}")
        return False

def test_export_reports():
    """Test export functionality"""
    print("Testing export functionality...")
    try:
        # Test export with multiple items
        payload = {
            "items": [
                "summary_report",
                "raw_data",
                "eda_analysis",
                "ml_models",
                "rl_simulation"
            ]
        }
        
        response = requests.post(f"{BASE_URL}/export", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert len(data["files"]) > 0
        
        # Test file download
        for filename in data["files"]:
            download_response = requests.get(f"{BASE_URL}/download/{filename}")
            assert download_response.status_code == 200
        
        print("✓ Export functionality passed")
        return True
    except Exception as e:
        print(f"✗ Export functionality failed: {e}")
        return False

def test_status_endpoint():
    """Test status endpoint"""
    print("Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        assert response.status_code == 200
        data = response.json()
        
        assert "data_uploaded" in data
        assert "eda_completed" in data
        assert "ml_models_trained" in data
        assert "rl_simulations_run" in data
        
        print("✓ Status endpoint passed")
        return True
    except Exception as e:
        print(f"✗ Status endpoint failed: {e}")
        return False

def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("\n" + "="*50)
    print("TESTING END-TO-END WORKFLOW")
    print("="*50)
    
    results = []
    
    # Step 1: Health check
    results.append(("Health Check", test_health_check()))
    
    # Step 2: Upload data
    results.append(("Data Upload", test_upload_data()))
    
    # Step 3: EDA Analysis
    results.append(("EDA Analysis", test_eda_analysis()))
    
    # Step 4: ML Training
    results.append(("ML Training", test_ml_training()))
    
    # Step 5: RL Simulation
    results.append(("RL Simulation", test_rl_simulation()))
    
    # Step 6: Export Reports
    results.append(("Export Reports", test_export_reports()))
    
    # Step 7: Status Check
    results.append(("Status Check", test_status_endpoint()))
    
    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Backend is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the logs above.")
        return False

def main():
    """Main test function"""
    print("Starting comprehensive backend test suite...")
    print(f"Testing backend at: {BASE_URL}")
    print("="*50)
    
    # Wait for backend to start
    print("Waiting for backend to start...")
    time.sleep(2)
    
    # Run tests
    success = test_end_to_end_workflow()
    
    if success:
        print("\n✅ Backend is ready for production use!")
        print("You can now start the frontend and use the application.")
    else:
        print("\n❌ Backend has issues that need to be resolved.")
    
    return success

if __name__ == "__main__":
    main()

