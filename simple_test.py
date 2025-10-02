#!/usr/bin/env python3
"""
Simple end-to-end test for the PriceOptima application
"""

import requests
import json

def test_upload():
    """Test file upload functionality"""
    print("Testing file upload...")
    try:
        # Create test data
        test_data = """Date,Product,Category,Price,Quantity,Revenue
2024-01-01,Rice 5kg,Grains,2500,45,112500
2024-01-01,Tomatoes 1kg,Vegetables,800,120,96000
2024-01-02,Bread 1kg,Bakery,300,200,60000
2024-01-02,Milk 1L,Dairy,500,150,75000
2024-01-03,Chicken 1kg,Meat,1200,80,96000"""
        
        # Save test data to file
        with open("test_data.csv", "w") as f:
            f.write(test_data)
        
        # Upload file
        files = {'file': open('test_data.csv', 'rb')}
        response = requests.post("http://localhost:8000/upload", files=files)
        files['file'].close()
        
        if response.status_code == 200:
            result = response.json()
            print("PASS: Upload successful")
            print(f"  Total records: {result['summary']['totalRecords']}")
            print(f"  Products: {result['summary']['products']}")
            print(f"  Categories: {result['summary']['categories']}")
            print(f"  Total revenue: N{result['summary']['totalRevenue']:,.2f}")
            print(f"  Preview rows: {len(result['preview'])}")
            print(f"  Total rows field: {result.get('totalRows', 'MISSING')}")
            return result
        else:
            print(f"FAIL: Upload failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return None
    except Exception as e:
        print(f"FAIL: Upload error: {e}")
        return None

def test_eda():
    """Test EDA analysis"""
    print("\nTesting EDA analysis...")
    try:
        response = requests.post("http://localhost:8000/eda", json={})
        if response.status_code == 200:
            result = response.json()
            print("PASS: EDA analysis successful")
            print(f"  Categories: {len(result['overview']['category_distribution'])}")
            print(f"  Insights: {len(result['insights'])}")
            return result
        else:
            print(f"FAIL: EDA analysis failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"FAIL: EDA analysis error: {e}")
        return None

def main():
    """Run tests"""
    print("Starting PriceOptima Test")
    print("=" * 40)
    
    # Test upload
    upload_result = test_upload()
    if upload_result is None:
        print("\nFAIL: Upload test failed, stopping")
        return False
    
    # Test EDA
    eda_result = test_eda()
    if eda_result is None:
        print("\nFAIL: EDA test failed")
        return False
    
    print("\n" + "=" * 40)
    print("SUCCESS: All tests passed!")
    print("The application is working correctly.")
    print("You can now use the frontend at http://localhost:3000")
    return True

if __name__ == "__main__":
    main()

