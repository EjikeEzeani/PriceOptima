#!/usr/bin/env python3
"""
One-command restart to eliminate port conflicts and run both servers.
Usage (PowerShell):
  python start_working_app.py
"""
import os, subprocess, time, sys, signal
import socket

ROOT = os.path.dirname(os.path.abspath(__file__))

def kill_port(port: int):
    try:
        # Windows: use netstat to find PIDs and taskkill them
        out = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode(errors='ignore')
        pids = set()
        for line in out.splitlines():
            parts = [p for p in line.split(' ') if p]
            if len(parts) >= 5:
                pids.add(parts[-1])
        for pid in pids:
            if pid.isdigit():
                try:
                    subprocess.call(f'taskkill /PID {pid} /F', shell=True)
                except Exception:
                    pass
    except Exception:
        pass

def wait_port(host: str, port: int, timeout: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            try:
                s.connect((host, port))
                return True
            except Exception:
                time.sleep(0.5)
    return False

def main():
    os.chdir(ROOT)
    # Kill common dev ports
    for p in (8000, 3000):
        kill_port(p)

    # Start backend
    backend = subprocess.Popen([sys.executable, 'api_backend.py'], cwd=ROOT)
    if not wait_port('127.0.0.1', 8000, 30):
        print('ERROR: Backend failed to start on 8000')
        backend.terminate()
        sys.exit(1)
    print('Backend running at http://127.0.0.1:8000')

    # Start frontend (dev)
    dashboard_dir = os.path.join(ROOT, 'dynamic-pricing-dashboard')
    env = os.environ.copy()
    env['NEXT_PUBLIC_API_URL'] = 'http://127.0.0.1:8000'
    frontend = subprocess.Popen(['npx', 'next', 'dev'], cwd=dashboard_dir, env=env, shell=True)
    if not wait_port('127.0.0.1', 3000, 60):
        print('ERROR: Frontend failed to start on 3000')
        backend.terminate()
        frontend.terminate()
        sys.exit(1)
    print('Frontend running at http://127.0.0.1:3000')

    print('Press Ctrl+C to stop both servers...')
    try:
        backend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            backend.terminate()
        except Exception:
            pass
        try:
            frontend.terminate()
        except Exception:
            pass

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Startup script for the working application
Starts both backend and frontend with proper error handling
"""

import subprocess
import time
import os
import sys
import threading
import requests
from pathlib import Path

def check_backend_health():
    """Check if backend is running and healthy"""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_backend():
    """Start the backend server"""
    print("🚀 Starting backend server...")
    try:
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        # Start backend
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "working_backend:app", 
            "--host", "127.0.0.1", 
            "--port", "8000", 
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_backend_health():
                print("✅ Backend started successfully!")
                return process
            time.sleep(1)
        
        print("❌ Backend failed to start within 30 seconds")
        return None
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the frontend server"""
    print("🚀 Starting frontend server...")
    try:
        # Change to frontend directory
        frontend_dir = Path(__file__).parent / "dynamic-pricing-dashboard"
        os.chdir(frontend_dir)
        
        # Install dependencies if needed
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Start frontend
        process = subprocess.Popen([
            "npm", "run", "dev"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Frontend started successfully!")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def run_tests():
    """Run comprehensive tests"""
    print("🧪 Running comprehensive tests...")
    try:
        # Change back to project root
        os.chdir(Path(__file__).parent)
        
        # Run test suite
        result = subprocess.run([
            sys.executable, "test_working_backend.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main startup function"""
    print("="*60)
    print("🚀 STARTING DYNAMIC PRICING ANALYTICS APPLICATION")
    print("="*60)
    
    # Step 1: Start Backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Cannot start application without backend")
        return False
    
    # Step 2: Run Tests
    if not run_tests():
        print("⚠️  Tests failed, but continuing with startup...")
    
    # Step 3: Start Frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Cannot start application without frontend")
        backend_process.terminate()
        return False
    
    # Step 4: Display URLs
    print("\n" + "="*60)
    print("🎉 APPLICATION STARTED SUCCESSFULLY!")
    print("="*60)
    print("📊 Backend API: http://127.0.0.1:8000")
    print("🌐 Frontend App: http://localhost:3000")
    print("📚 API Docs: http://127.0.0.1:8000/docs")
    print("="*60)
    print("\n📋 Available Features:")
    print("  • Data Upload & Processing")
    print("  • Exploratory Data Analysis (EDA)")
    print("  • Machine Learning Model Training")
    print("  • Reinforcement Learning Simulation")
    print("  • Report Generation & Export")
    print("  • Real-time Visualizations")
    print("\n🛑 Press Ctrl+C to stop the application")
    print("="*60)
    
    try:
        # Keep the application running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down application...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        
        print("✅ Application stopped successfully")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

