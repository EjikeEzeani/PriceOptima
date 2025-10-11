@echo off
echo Starting Dynamic Pricing Analytics Application...
echo.

echo Starting Backend Server...
start "Backend" cmd /k "cd /d C:\Users\USER\Downloads\Msc Project && python working_backend.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Server...
start "Frontend" cmd /k "cd /d C:\Users\USER\Downloads\Msc Project\dynamic-pricing-dashboard && npm run dev"

echo.
echo Application is starting...
echo Backend: http://127.0.0.1:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul
