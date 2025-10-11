@echo off
echo 🚀 Deploying PriceOptima Dashboard to Vercel
echo ================================================

echo.
echo 📋 Pre-deployment Checklist:
echo [ ] Backend deployed to Render/Railway
echo [ ] Backend URL noted
echo [ ] GitHub repository updated
echo [ ] Environment variables ready
echo.

echo 🔧 Step 1: Building the application...
cd dynamic-pricing-dashboard
call npm run build
if %errorlevel% neq 0 (
    echo ❌ Build failed! Please fix errors and try again.
    pause
    exit /b 1
)

echo.
echo ✅ Build successful!
echo.

echo 🌐 Step 2: Deploying to Vercel...
echo.
echo 📝 Instructions:
echo 1. Go to https://vercel.com
echo 2. Import your GitHub repository
echo 3. Set Root Directory to: dynamic-pricing-dashboard
echo 4. Add Environment Variable: NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com
echo 5. Click Deploy
echo.

echo 🎯 Vercel Configuration:
echo Framework: Next.js
echo Root Directory: dynamic-pricing-dashboard
echo Build Command: npm run build
echo Output Directory: out
echo.

echo 📊 Environment Variables Required:
echo NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com
echo.

echo ✅ Ready for deployment!
echo.
pause
