#!/usr/bin/env python3
"""
Test Export Functionality
"""

import requests
import json

def test_export():
    base_url = "http://localhost:8000"
    
    print("Testing Export Functionality")
    print("=" * 40)
    
    # Test export with different report types
    export_items = ["summary_report", "raw_data", "ml_results"]
    
    try:
        response = requests.post(f"{base_url}/export", 
                               json={"items": export_items}, 
                               timeout=30)
        
        if response.status_code == 200:
            export_data = response.json()
            print("PASS: Export successful")
            print(f"  Status: {export_data.get('status')}")
            print(f"  Exported items: {export_data.get('exported')}")
            print(f"  Files generated: {len(export_data.get('files', []))}")
            print(f"  Message: {export_data.get('message')}")
            
            # List generated files
            for file in export_data.get('files', []):
                print(f"  - {file}")
                
        else:
            print(f"FAIL: Export failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            
    except Exception as e:
        print(f"FAIL: Export test failed: {str(e)}")

if __name__ == "__main__":
    test_export()
