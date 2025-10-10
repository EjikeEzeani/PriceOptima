@echo off
REM Fix Vercel Root Directory Error
REM This script helps fix the "No Next.js version detected" error

echo ==========================================
echo   Fix Vercel Root Directory Error
echo ==========================================
echo.

echo Problem: Vercel can't find Next.js package.json
echo Solution: Set correct Root Directory in Vercel settings
echo.

echo Current project structure:
echo Msc Project/ ^(root repository^)
echo ├── dynamic-pricing-dashboard/ ^(Next.js app^)
echo │   ├── package.json ✅
echo │   ├── next.config.mjs ✅
echo │   └── app/ ✅
echo └── ... ^(other files^)
echo.

echo ==========================================
echo   Vercel Settings Fix
echo ==========================================
echo.

echo Step 1: Go to Vercel Project Settings
echo https://vercel.com/dashboard
echo.

echo Step 2: Click Settings ^> General
echo.

echo Step 3: Find "Root Directory" section
echo.

echo Step 4: Change Root Directory to:
echo dynamic-pricing-dashboard
echo.
echo NOT: empty, /, ./, or dynamic-pricing-dashboard/
echo.

echo Step 5: Save settings
echo.

echo Step 6: Redeploy your project
echo.

echo ==========================================
echo   Verification
echo ==========================================
echo.

echo After fixing Root Directory, Vercel should detect:
echo ✅ Framework: Next.js
echo ✅ Build Command: npm run build
echo ✅ Install Command: npm install
echo ✅ Output Directory: out
echo.

echo Build logs should show:
echo ✓ Detected Next.js
echo ✓ Installing dependencies
echo ✓ Building Next.js application
echo ✓ Static export completed
echo ✓ Deploying to CDN
echo.

echo ==========================================
echo   Alternative Solutions
echo ==========================================
echo.

echo If Root Directory fix doesn't work:
echo.
echo Option 1: Move files to repository root
echo - Move all files from dynamic-pricing-dashboard/ to root
echo - Set Root Directory to empty
echo.
echo Option 2: Create new Vercel project
echo - Delete current project
echo - Create new project
echo - Set Root Directory correctly from start
echo.

echo ==========================================
echo   Quick Check
echo ==========================================
echo.

echo Verify your package.json contains Next.js:
type dynamic-pricing-dashboard\package.json | findstr "next"
echo.

echo If you see "next": "14.2.16", then the file is correct.
echo The issue is just the Root Directory setting.
echo.

echo ==========================================
echo   Ready to Fix!
echo ==========================================
echo.

echo Next steps:
echo 1. Go to Vercel project settings
echo 2. Set Root Directory to: dynamic-pricing-dashboard
echo 3. Save and redeploy
echo 4. Build should succeed!
echo.

pause
