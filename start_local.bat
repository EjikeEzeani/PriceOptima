@echo off
echo Starting PriceOptima Application...

echo Starting Backend on port 8000...
start "Backend" cmd /k "cd /d C:\Users\USER\Downloads\Msc Project && python -c \"from working_backend import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)\""

timeout /t 3 /nobreak >nul

echo Starting Frontend on port 3000...
start "Frontend" cmd /k "cd /d C:\Users\USER\Downloads\Msc Project\dynamic-pricing-dashboard && npm run dev"

echo Both services are starting...
echo Backend: http://127.0.0.1:8000
echo Frontend: http://localhost:3000
pause
