#!/usr/bin/env python3
"""
Test script to check the actual API response structure
"""

import requests
import json
import pandas as pd
import io

def test_upload_response():
    """Test the upload endpoint and see the actual response structure"""
    
    # Create a sample CSV data
    sample_data = """date,product,category,price,quantity,revenue
2024-01-01,Rice 5kg,Grains,2500,45,112500
2024-01-01,Tomatoes 1kg,Vegetables,800,120,96000
2024-01-02,Chicken 1kg,Meat,1500,30,45000
2024-01-02,Milk 1L,Dairy,600,80,48000
2024-01-03,Bread,Bakery,300,100,30000"""
    
    # Convert to CSV file-like object
    csv_file = io.StringIO(sample_data)
    files = {'file': ('test_data.csv', csv_file.getvalue(), 'text/csv')}
    
    try:
        # Make request to upload endpoint
        response = requests.post('http://127.0.0.1:8000/upload', files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful!")
            print("\n📊 Response Structure:")
            print(f"Keys: {list(data.keys())}")
            
            print(f"\n📋 Headers: {data.get('headers', [])}")
            print(f"📈 Summary: {data.get('summary', {})}")
            print(f"👀 Preview length: {len(data.get('preview', []))}")
            print(f"📄 Rows length: {len(data.get('rows', []))}")
            
            print(f"\n🔍 First preview item:")
            if data.get('preview'):
                print(json.dumps(data['preview'][0], indent=2))
            
            print(f"\n🔍 First row item:")
            if data.get('rows'):
                print(json.dumps(data['rows'][0], indent=2))
                
            # Check for object types in preview
            print(f"\n🔍 Preview data types:")
            if data.get('preview'):
                for i, item in enumerate(data['preview'][:2]):
                    print(f"Item {i}:")
                    for key, value in item.items():
                        print(f"  {key}: {type(value).__name__} = {value}")
            
            return data
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Testing API Response Structure")
    print("=" * 50)
    
    # Test upload response
    result = test_upload_response()
    
    if result:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
