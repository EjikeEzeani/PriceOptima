#!/usr/bin/env python3
"""
Complete end-to-end test for the PriceOptima application
Tests the full data flow from upload to display
"""

import requests
import json
import time

def test_health_check():
    """Test if the backend is healthy"""
    print("[TEST] Testing backend health...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("[PASS] Backend is healthy")
            return True
        else:
            print(f"[FAIL] Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Backend health check error: {e}")
        return False

def test_upload():
    """Test file upload functionality"""
    print("\n[TEST] Testing file upload...")
    try:
        # Create test data
        test_data = """Date,Product,Category,Price,Quantity,Revenue
2024-01-01,Rice 5kg,Grains,2500,45,112500
2024-01-01,Tomatoes 1kg,Vegetables,800,120,96000
2024-01-02,Bread 1kg,Bakery,300,200,60000
2024-01-02,Milk 1L,Dairy,500,150,75000
2024-01-03,Chicken 1kg,Meat,1200,80,96000
2024-01-03,Apples 1kg,Fruits,600,100,60000
2024-01-04,Potatoes 5kg,Vegetables,1500,60,90000
2024-01-04,Yogurt 500ml,Dairy,400,80,32000"""
        
        # Save test data to file
        with open("test_flow_data.csv", "w") as f:
            f.write(test_data)
        
        # Upload file
        files = {'file': open('test_flow_data.csv', 'rb')}
        response = requests.post("http://localhost:8000/upload", files=files)
        files['file'].close()
        
        if response.status_code == 200:
            result = response.json()
            print("[PASS] Upload successful")
            print(f"   - Total records: {result['summary']['totalRecords']}")
            print(f"   - Products: {result['summary']['products']}")
            print(f"   - Categories: {result['summary']['categories']}")
            print(f"   - Total revenue: N{result['summary']['totalRevenue']:,.2f}")
            print(f"   - Preview rows: {len(result['preview'])}")
            print(f"   - Total rows field: {result.get('totalRows', 'MISSING')}")
            return result
        else:
            print(f"[FAIL] Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"[FAIL] Upload error: {e}")
        return None

def test_eda():
    """Test EDA analysis"""
    print("\n📊 Testing EDA analysis...")
    try:
        response = requests.post("http://localhost:8000/eda", json={})
        if response.status_code == 200:
            result = response.json()
            print("✅ EDA analysis successful")
            print(f"   - Categories: {len(result['overview']['category_distribution'])}")
            print(f"   - Insights: {len(result['insights'])}")
            print(f"   - Recommendations: {len(result['recommendations'])}")
            return result
        else:
            print(f"❌ EDA analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ EDA analysis error: {e}")
        return None

def test_ml():
    """Test ML training"""
    print("\n🤖 Testing ML training...")
    try:
        response = requests.post("http://localhost:8000/ml", json={"model": "random_forest"})
        if response.status_code == 200:
            result = response.json()
            print("✅ ML training successful")
            print(f"   - Model: {result['modelId']}")
            print(f"   - R² Score: {result['metrics']['r2']:.4f}")
            print(f"   - RMSE: {result['metrics']['rmse']:.4f}")
            print(f"   - MAE: {result['metrics']['mae']:.4f}")
            print(f"   - Predictions: {len(result['predictions'])}")
            return result
        else:
            print(f"❌ ML training failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ ML training error: {e}")
        return None

def test_rl():
    """Test RL simulation"""
    print("\n⚡ Testing RL simulation...")
    try:
        response = requests.post("http://localhost:8000/rl", json={"algorithm": "dqn"})
        if response.status_code == 200:
            result = response.json()
            print("✅ RL simulation successful")
            print(f"   - Algorithm: {result['algorithm']}")
            print(f"   - Final reward: {result['finalReward']}")
            print(f"   - Convergence episode: {result['convergenceEpisode']}")
            print(f"   - Waste reduction: {result['policy']['wasteReduction']}%")
            print(f"   - Profit increase: {result['policy']['profitIncrease']}%")
            return result
        else:
            print(f"❌ RL simulation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ RL simulation error: {e}")
        return None

def test_export():
    """Test export functionality"""
    print("\n📁 Testing export...")
    try:
        response = requests.post("http://localhost:8000/export", json={
            "items": ["summary_report", "raw_data", "ml_results"]
        })
        if response.status_code == 200:
            result = response.json()
            print("✅ Export successful")
            print(f"   - Status: {result['status']}")
            print(f"   - Exported items: {result['exported']}")
            print(f"   - Files created: {len(result['files'])}")
            for file in result['files']:
                print(f"     - {file}")
            return result
        else:
            print(f"❌ Export failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Export error: {e}")
        return None

def main():
    """Run complete end-to-end test"""
    print("🚀 Starting PriceOptima End-to-End Test")
    print("=" * 50)
    
    # Test sequence
    tests = [
        ("Health Check", test_health_check),
        ("Upload", test_upload),
        ("EDA Analysis", test_eda),
        ("ML Training", test_ml),
        ("RL Simulation", test_rl),
        ("Export", test_export),
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result is not None
            if result is None:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The application is working correctly.")
        print("💡 You can now use the frontend at http://localhost:3000")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()
