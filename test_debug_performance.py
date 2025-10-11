#!/usr/bin/env python3
"""
Test debug panel performance improvements
"""

import time
import requests

def test_backend_health():
    """Test backend health endpoint speed"""
    start_time = time.time()
    try:
        response = requests.get('http://127.0.0.1:8000/health', timeout=5)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ Backend Health: {end_time - start_time:.3f}s")
            return True
        else:
            print(f"❌ Backend Health: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend Health: {str(e)}")
        return False

def test_cors_check():
    """Test CORS endpoint speed"""
    start_time = time.time()
    try:
        response = requests.options('http://127.0.0.1:8000/health', timeout=5)
        end_time = time.time()
        
        print(f"✅ CORS Check: {end_time - start_time:.3f}s")
        return True
    except Exception as e:
        print(f"❌ CORS Check: {str(e)}")
        return False

def test_quick_upload():
    """Test quick file upload speed"""
    start_time = time.time()
    try:
        test_data = 'date,product,category,price,quantity,revenue\n2024-01-01,Test Product,Test Category,100,10,1000'
        files = {'file': ('test.csv', test_data, 'text/csv')}
        
        response = requests.post('http://127.0.0.1:8000/upload', files=files, timeout=10)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ Quick Upload: {end_time - start_time:.3f}s")
            return True
        else:
            print(f"❌ Quick Upload: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Quick Upload: {str(e)}")
        return False

def main():
    print("🚀 Debug Panel Performance Test")
    print("=" * 40)
    
    # Test basic endpoints
    health_ok = test_backend_health()
    cors_ok = test_cors_check()
    upload_ok = test_quick_upload()
    
    print(f"\n📊 Results:")
    print(f"  Backend Health: {'✅' if health_ok else '❌'}")
    print(f"  CORS Check: {'✅' if cors_ok else '❌'}")
    print(f"  Quick Upload: {'✅' if upload_ok else '❌'}")
    
    if all([health_ok, cors_ok, upload_ok]):
        print(f"\n🎉 All basic tests passed! Debug panel should be fast.")
    else:
        print(f"\n⚠️  Some tests failed. Check backend status.")

if __name__ == "__main__":
    main()
