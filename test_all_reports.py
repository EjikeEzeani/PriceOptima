#!/usr/bin/env python3
"""
Test All Report Types
"""

import requests
import json
import os

def test_all_reports():
    base_url = "http://localhost:8000"
    
    print("Testing All Report Types")
    print("=" * 50)
    
    # Test all available report types
    all_report_types = [
        "summary_report",
        "raw_data", 
        "ml_results",
        "technical_report",
        "presentation",
        "rl_policy",
        "visualizations"
    ]
    
    results = {}
    
    for report_type in all_report_types:
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
                    
                    # List the generated files
                    for file in export_data.get('files', []):
                        print(f"   - {file}")
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
    print("REPORT GENERATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for status in results.values() if status == "PASS")
    total = len(results)
    
    for report_type, status in results.items():
        status_icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"{status_icon} {report_type}: {status}")
    
    print(f"\nOverall: {passed}/{total} report types working")
    
    if passed == total:
        print("SUCCESS: ALL REPORTS ARE WORKING PERFECTLY!")
    else:
        print(f"WARNING: {total - passed} report types need attention")
    
    # Check exports directory
    print(f"\nExports directory contains {len(os.listdir('exports'))} files")
    
    return results

if __name__ == "__main__":
    test_all_reports()
