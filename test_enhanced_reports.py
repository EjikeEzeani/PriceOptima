#!/usr/bin/env python3
"""
Test Enhanced Report Generation
"""

import requests
import json
import os

def test_enhanced_reports():
    base_url = "http://localhost:8000"
    
    print("Testing Enhanced Report Generation")
    print("=" * 50)
    
    # First upload some data
    print("1. Uploading test data...")
    test_data = """Date,Product,Category,Price,Quantity,Revenue
2024-01-01,Rice 5kg,Grains,2500,45,112500
2024-01-01,Tomatoes 1kg,Vegetables,800,120,96000
2024-01-02,Bread 1kg,Bakery,300,200,60000
2024-01-02,Milk 1L,Dairy,500,150,75000
2024-01-03,Chicken 1kg,Meat,1200,80,96000"""
    
    # Create a temporary CSV file
    with open("test_data.csv", "w") as f:
        f.write(test_data)
    
    try:
        # Upload data
        with open("test_data.csv", "rb") as f:
            files = {"file": ("test_data.csv", f, "text/csv")}
            response = requests.post(f"{base_url}/upload", files=files)
        
        if response.status_code == 200:
            print("[SUCCESS] Data uploaded successfully")
        else:
            print(f"[FAIL] Upload failed: {response.status_code}")
            return
        
        # Test all report types
        report_types = [
            "summary_report",
            "technical_report", 
            "raw_data",
            "ml_results",
            "presentation",
            "rl_policy",
            "visualizations"
        ]
        
        print("\n2. Testing report generation...")
        results = {}
        
        for report_type in report_types:
            print(f"\nTesting {report_type}...")
            try:
                response = requests.post(f"{base_url}/export", 
                                       json={"items": [report_type]}, 
                                       timeout=30)
                
                if response.status_code == 200:
                    export_data = response.json()
                    files_generated = len(export_data.get('files', []))
                    
                    if files_generated > 0:
                        print(f"[PASS] {report_type}: {files_generated} files generated")
                        results[report_type] = "PASS"
                        
                        # List the generated files and their types
                        for file in export_data.get('files', []):
                            file_ext = file.split('.')[-1].upper()
                            print(f"   - {file} ({file_ext})")
                    else:
                        print(f"[FAIL] {report_type}: No files generated")
                        results[report_type] = "FAIL"
                else:
                    print(f"[FAIL] {report_type}: Status {response.status_code}")
                    print(f"   Response: {response.text}")
                    results[report_type] = "FAIL"
                    
            except Exception as e:
                print(f"[ERROR] {report_type}: {str(e)}")
                results[report_type] = "ERROR"
        
        # Summary
        print("\n" + "=" * 50)
        print("ENHANCED REPORT GENERATION SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for status in results.values() if status == "PASS")
        total = len(results)
        
        for report_type, status in results.items():
            status_icon = "[PASS]" if status == "PASS" else "[FAIL]"
            print(f"{status_icon} {report_type}: {status}")
        
        print(f"\nOverall: {passed}/{total} report types working")
        
        if passed == total:
            print("SUCCESS: ALL ENHANCED REPORTS ARE WORKING!")
        else:
            print(f"WARNING: {total - passed} report types need attention")
        
        # Check file types in exports directory
        print(f"\nExports directory contains {len(os.listdir('exports'))} files")
        
        # Show file type distribution
        file_types = {}
        for file in os.listdir('exports'):
            ext = file.split('.')[-1].upper()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        print("\nFile type distribution:")
        for ext, count in file_types.items():
            print(f"  {ext}: {count} files")
        
        return results
        
    finally:
        # Clean up
        if os.path.exists("test_data.csv"):
            os.remove("test_data.csv")

if __name__ == "__main__":
    test_enhanced_reports()
