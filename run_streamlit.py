#!/usr/bin/env python3
"""
Script to run the PriceOptima Streamlit dashboard
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app."""
    print("🚀 Starting PriceOptima Streamlit Dashboard...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
    
    # Check if plotly is installed
    try:
        import plotly
        print(f"✅ Plotly version: {plotly.__version__}")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    
    # Run the Streamlit app
    print("\n🌐 Starting Streamlit server...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    main()

