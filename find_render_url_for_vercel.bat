@echo off
REM Script to help find Render backend URL for Vercel environment variables

echo ==========================================
echo   Find Your Render Backend URL
echo ==========================================
echo.

echo To set up Vercel environment variables, you need your Render backend URL.
echo.

echo Your Render backend URL should look like:
echo https://your-app-name.onrender.com
echo.

echo To find your Render backend URL:
echo.
echo 1. Go to https://dashboard.render.com
echo 2. Sign in to your account
echo 3. Find your backend service (usually named "priceoptima" or similar)
echo 4. Click on the service name
echo 5. Copy the URL from the service details page
echo.

echo Common Render service names:
echo - priceoptima-1
echo - priceoptima-backend
echo - priceoptima-api
echo - your-username-priceoptima
echo.

echo Once you have your Render URL, set this environment variable in Vercel:
echo.
echo Variable Name: NEXT_PUBLIC_API_URL
echo Variable Value: https://your-actual-render-url.onrender.com
echo.

echo Example:
echo NEXT_PUBLIC_API_URL=https://priceoptima-1.onrender.com
echo.

echo Steps to set in Vercel:
echo 1. Go to your Vercel project dashboard
echo 2. Click Settings ^> Environment Variables
echo 3. Add new variable with name: NEXT_PUBLIC_API_URL
echo 4. Add value: https://your-render-url.onrender.com
echo 5. Select all environments (Production, Preview, Development)
echo 6. Save and redeploy
echo.

echo After setting the environment variable:
echo - Redeploy your Vercel project
echo - Test the connection between frontend and backend
echo - Verify file upload functionality works
echo.

pause
