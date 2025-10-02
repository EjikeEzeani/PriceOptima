#!/usr/bin/env python3
"""
Performance test for upload optimization
"""

import time
import requests
import pandas as pd
import io

def create_large_test_data(rows=1000):
    """Create a larger test dataset"""
    data = []
    products = ['Rice 5kg', 'Tomatoes 1kg', 'Milk 1L', 'Bread', 'Chicken 1kg', 'Apples 1kg', 'Potatoes 2kg', 'Beef 1kg', 'Yogurt 500ml', 'Bananas 1kg']
    categories = ['Grains', 'Vegetables', 'Dairy', 'Bakery', 'Meat', 'Fruits']
    
    for i in range(rows):
        product = products[i % len(products)]
        category = categories[i % len(categories)]
        price = 500 + (i % 2000)
        quantity = 10 + (i % 100)
        revenue = price * quantity
        
        data.append({
            'date': f'2024-01-{(i % 30) + 1:02d}',
            'product': product,
            'category': category,
            'price': price,
            'quantity': quantity,
            'revenue': revenue
        })
    
    return pd.DataFrame(data)

def test_upload_performance():
    """Test upload performance with different dataset sizes"""
    base_url = "http://127.0.0.1:8000"
    
    # Test with different dataset sizes
    test_sizes = [100, 500, 1000, 2000]
    
    print("🚀 Performance Test - Upload Optimization")
    print("=" * 50)
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} rows...")
        
        # Create test data
        start_time = time.time()
        df = create_large_test_data(size)
        data_creation_time = time.time() - start_time
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Test upload
        files = {'file': ('test_data.csv', csv_content, 'text/csv')}
        
        upload_start = time.time()
        try:
            response = requests.post(f"{base_url}/upload", files=files)
            upload_time = time.time() - upload_start
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Upload successful in {upload_time:.2f}s")
                print(f"  📈 Data creation: {data_creation_time:.2f}s")
                print(f"  📤 Upload time: {upload_time:.2f}s")
                print(f"  📊 Total records: {data.get('summary', {}).get('totalRecords', 'N/A')}")
                print(f"  🎯 Preview rows: {len(data.get('preview', []))}")
                print(f"  📋 All rows sent: {len(data.get('rows', []))}")
            else:
                print(f"  ❌ Upload failed: {response.status_code}")
                print(f"  📝 Error: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print(f"\n🎉 Performance test completed!")

if __name__ == "__main__":
    test_upload_performance()
