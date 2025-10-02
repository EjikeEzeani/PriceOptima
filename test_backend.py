#!/usr/bin/env python3
"""
Test script to verify the backend API is working correctly
"""

import requests
import pandas as pd
import io
import json

# Test data
test_data = {
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
    'product': ['Rice', 'Tomatoes', 'Milk', 'Bread', 'Apples'],
    'category': ['Grains', 'Vegetables', 'Dairy', 'Bakery', 'Fruits'],
    'price': [100, 50, 30, 25, 40],
    'quantity': [10, 20, 15, 30, 25],
    'revenue': [1000, 1000, 450, 750, 1000]
}

def test_backend():
    base_url = "http://localhost:8000"
    
    print("Testing Backend API...")
    
    # Create test CSV
    df = pd.DataFrame(test_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Test 1: Upload data
    print("\n1. Testing upload endpoint...")
    files = {'file': ('test_data.csv', csv_content, 'text/csv')}
    response = requests.post(f"{base_url}/upload", files=files)
    
    if response.status_code == 200:
        print("✅ Upload successful")
        upload_data = response.json()
        print(f"   Records: {upload_data['summary']['totalRecords']}")
    else:
        print(f"❌ Upload failed: {response.status_code}")
        return
    
    # Test 2: EDA analysis
    print("\n2. Testing EDA endpoint...")
    response = requests.post(f"{base_url}/eda", json={})
    
    if response.status_code == 200:
        print("✅ EDA analysis successful")
        eda_data = response.json()
        print(f"   Categories: {list(eda_data['overview']['category_distribution'].keys())}")
    else:
        print(f"❌ EDA failed: {response.status_code}")
    
    # Test 3: ML training
    print("\n3. Testing ML endpoint...")
    response = requests.post(f"{base_url}/ml", json={"model": "random_forest"})
    
    if response.status_code == 200:
        print("✅ ML training successful")
        ml_data = response.json()
        print(f"   R² Score: {ml_data['metrics']['r2']:.3f}")
    else:
        print(f"❌ ML training failed: {response.status_code}")
    
    # Test 4: Export
    print("\n4. Testing export endpoint...")
    response = requests.post(f"{base_url}/export", json={"items": ["summary_report", "raw_data"]})
    
    if response.status_code == 200:
        print("✅ Export successful")
        export_data = response.json()
        print(f"   Exported files: {len(export_data['files'])}")
    else:
        print(f"❌ Export failed: {response.status_code}")
    
    print("\n🎉 Backend testing completed!")

if __name__ == "__main__":
    test_backend()





