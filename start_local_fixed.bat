@echo off
echo ========================================
echo    PriceOptima Local Development
echo ========================================
echo.

echo Starting Backend on port 8001...
start "Backend" cmd /k "cd /d C:\Users\USER\Downloads\Msc Project && python -m uvicorn render_super_minimal:app --host 127.0.0.1 --port 8001"

timeout /t 3 /nobreak >nul

echo Starting Frontend on port 3000...
start "Frontend" cmd /k "cd /d C:\Users\USER\Downloads\Msc Project\dynamic-pricing-dashboard && npm run dev"

echo.
echo ========================================
echo    Services Starting...
echo ========================================
echo Backend:  http://127.0.0.1:8001
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit this window...
pause >nul
