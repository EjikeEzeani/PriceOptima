@echo off
echo ========================================
echo Starting Dynamic Pricing Analytics App
echo ========================================
echo.

echo [1/4] Starting Backend Server...
start "Backend Server" cmd /k "cd /d C:\Users\USER\Downloads\Msc Project && python working_backend.py"

echo [2/4] Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo [3/4] Testing backend connection...
curl -s http://127.0.0.1:8000/health > nul
if %errorlevel% equ 0 (
    echo ✅ Backend is running successfully
) else (
    echo ❌ Backend failed to start
    pause
    exit /b 1
)

echo [4/4] Starting Frontend Server...
start "Frontend Server" cmd /k "cd /d C:\Users\USER\Downloads\Msc Project\dynamic-pricing-dashboard && npm run dev"

echo.
echo ========================================
echo Application Status
echo ========================================
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:3000
echo.
echo Test Data: test_data.csv (available in project root)
echo.
echo ========================================
echo Instructions
echo ========================================
echo 1. Wait for both servers to fully start
echo 2. Open http://localhost:3000 in your browser
echo 3. Upload the test_data.csv file
echo 4. Check browser console for any errors
echo.
echo Press any key to open the application in your browser...
pause > nul

start http://localhost:3000
