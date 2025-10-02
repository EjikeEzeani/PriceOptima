#!/usr/bin/env python3
"""
Startup script to run both backend and frontend
"""

import subprocess
import sys
import time
import os
import signal
import threading

def run_backend():
    """Run the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "api_backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    except KeyboardInterrupt:
        print("Backend stopped")

def run_frontend():
    """Run the Next.js frontend"""
    print("🚀 Starting Next.js frontend...")
    os.chdir("dynamic-pricing-dashboard")
    try:
        subprocess.run(["npm", "run", "dev"])
    except KeyboardInterrupt:
        print("Frontend stopped")

def main():
    print("🎯 Starting PriceOptima Application...")
    print("Backend will run on: http://localhost:8000")
    print("Frontend will run on: http://localhost:3000")
    print("Press Ctrl+C to stop both services")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Wait a bit for backend to start
    time.sleep(3)
    
    # Start frontend in main thread
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down application...")
        sys.exit(0)

if __name__ == "__main__":
    main()





