#!/usr/bin/env python3
"""
Test lightweight backend
"""

import requests
import time

def test_lightweight_backend():
    """Test lightweight backend on port 8001"""
    print("🚀 Testing Lightweight Backend")
    print("=" * 40)
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get('http://127.0.0.1:8001/health', timeout=5)
        if response.status_code == 200:
            print("✅ Backend Health: OK")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Backend Health: HTTP {response.status_code}")
            return False
        
        # Test upload with small data
        print("\nTesting upload endpoint...")
        test_data = 'date,product,category,price,quantity,revenue\n2024-01-01,Test Product,Test Category,100,10,1000'
        files = {'file': ('test.csv', test_data, 'text/csv')}
        
        start_time = time.time()
        response = requests.post('http://127.0.0.1:8001/upload', files=files, timeout=10)
        upload_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Upload Test: OK ({upload_time:.2f}s)")
            data = response.json()
            print(f"   Records: {data.get('summary', {}).get('totalRecords', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
        else:
            print(f"❌ Upload Test: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            return False
        
        print(f"\n🎉 Lightweight backend is working perfectly!")
        print(f"   Upload time: {upload_time:.2f}s")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Backend not running on port 8001")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_lightweight_backend()
