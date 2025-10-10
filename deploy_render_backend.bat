@echo off
REM Quick Render Deployment Guide
REM This script provides step-by-step instructions for deploying to Render

echo ==========================================
echo   Render Backend Deployment Guide
echo ==========================================
echo.

echo Your Render backend is ready to deploy!
echo.

echo Available backend options:
echo.
echo 1. Super Minimal Backend (RECOMMENDED)
echo    - File: render_super_minimal.py
echo    - Requirements: backend_requirements_super_minimal.txt
echo    - Dependencies: Only 3 packages
echo    - Memory: Ultra-low
echo    - Build: Fastest
echo.
echo 2. Bare Minimum Backend
echo    - File: render_bare_minimum.py
echo    - Requirements: backend_requirements_bare_minimum.txt
echo    - Dependencies: 4 packages
echo.
echo 3. Ultra Light Backend
echo    - File: render_ultra_light.py
echo    - Requirements: backend_requirements_minimal.txt
echo    - Dependencies: 6 packages
echo.
echo 4. Render Optimized Backend
echo    - File: render_optimized_backend.py
echo    - Requirements: backend_requirements_render.txt
echo    - Dependencies: More packages
echo.

echo ==========================================
echo   Deployment Steps
echo ==========================================
echo.

echo Step 1: Go to Render Dashboard
echo https://dashboard.render.com
echo.

echo Step 2: Create New Web Service
echo - Click "New +" ^> "Web Service"
echo - Connect your GitHub repository
echo - Select: EjikeEzeani/PriceOptima
echo.

echo Step 3: Configure Service (Super Minimal)
echo - Name: priceoptima-backend
echo - Root Directory: (leave empty)
echo - Build Command: pip install -r backend_requirements_super_minimal.txt
echo - Start Command: python -m uvicorn render_super_minimal:app --host 0.0.0.0 --port $PORT
echo.

echo Step 4: Set Environment Variables
echo PYTHONUNBUFFERED=1
echo MALLOC_TRIM_THRESHOLD_=100000
echo MALLOC_MMAP_THRESHOLD_=131072
echo.

echo Step 5: Deploy
echo - Click "Create Web Service"
echo - Wait for deployment (2-5 minutes)
echo - Note your service URL
echo.

echo ==========================================
echo   Connect to Vercel
echo ==========================================
echo.

echo After Render deployment:
echo 1. Get your Render URL (e.g., https://priceoptima-backend.onrender.com)
echo 2. Go to Vercel project settings
echo 3. Add environment variable:
echo    - Name: NEXT_PUBLIC_API_URL
echo    - Value: https://your-render-url.onrender.com
echo 4. Redeploy Vercel
echo.

echo ==========================================
echo   Test Your Setup
echo ==========================================
echo.

echo Test Render Backend:
echo - Visit: https://your-render-url.onrender.com/health
echo - Should return: {"status": "healthy", "message": "Backend is running"}
echo.

echo Test Full Integration:
echo - Visit your Vercel frontend
echo - Upload a CSV file
echo - Click "Process Data"
echo - Should connect to Render and process the file
echo.

echo ==========================================
echo   Troubleshooting
echo ==========================================
echo.

echo If Render build fails:
echo - Check requirements file name is correct
echo - Ensure Root Directory is empty
echo - Verify Build Command matches requirements file
echo.

echo If Render won't start:
echo - Check Start Command is correct
echo - Verify environment variables are set
echo - Check Render logs for errors
echo.

echo If Vercel can't connect:
echo - Verify NEXT_PUBLIC_API_URL is set correctly
echo - Check Render service is running
echo - Test Render URL directly in browser
echo.

echo ==========================================
echo   Ready to Deploy!
echo ==========================================
echo.

echo Your backend files are ready:
echo ✅ render_super_minimal.py
echo ✅ backend_requirements_super_minimal.txt
echo ✅ All configuration files updated
echo ✅ Environment variables documented
echo.

echo Next: Follow the steps above to deploy on Render!
echo.

pause
