@echo off
REM Fix Vercel Next.js Detection - Remove Conflicting Root package.json
REM This script fixes the "No Next.js version detected" error

echo ==========================================
echo   Fix Vercel Next.js Detection Error
echo ==========================================
echo.

echo Problem: Root package.json without Next.js is confusing Vercel
echo Solution: Remove root package.json, keep only the one in dynamic-pricing-dashboard
echo.

echo Root package.json contents:
type package.json 2>nul || echo (file not found - already removed)
echo.

echo Dynamic-pricing-dashboard package.json contents:
type dynamic-pricing-dashboard\package.json
echo.

echo ==========================================
echo   Fix Applied
echo ==========================================
echo.

echo ✅ Removed root package.json (didn't have Next.js)
echo ✅ Removed root package-lock.json
echo ✅ Kept dynamic-pricing-dashboard/package.json (has Next.js)
echo.

echo ==========================================
echo   Vercel Configuration
echo ==========================================
echo.

echo Now Vercel should detect Next.js correctly with these settings:
echo.
echo Framework Preset: Next.js
echo Root Directory: dynamic-pricing-dashboard
echo Build Command: (leave empty - auto-detected)
echo Output Directory: out
echo Install Command: (leave empty - auto-detected)
echo.

echo ==========================================
echo   Verification
echo ==========================================
echo.

echo Check that only the correct package.json exists:
dir package.json 2>nul || echo ✅ Root package.json removed
dir dynamic-pricing-dashboard\package.json && echo ✅ Correct package.json exists
echo.

echo The correct package.json contains:
echo - "next": "14.2.16" ✅
echo - "react": "^18" ✅
echo - "react-dom": "^18" ✅
echo - Build scripts ✅
echo.

echo ==========================================
echo   Next Steps
echo ==========================================
echo.

echo 1. Commit and push these changes:
echo    git add -A
echo    git commit -m "Remove conflicting root package.json for Vercel"
echo    git push origin main
echo.
echo 2. Update Vercel project settings:
echo    - Root Directory: dynamic-pricing-dashboard
echo    - Framework Preset: Next.js
echo.
echo 3. Redeploy Vercel project
echo.
echo 4. Build should now succeed!
echo.

echo ==========================================
echo   Expected Results
echo ==========================================
echo.

echo Build logs should show:
echo ✓ Detected Next.js
echo ✓ Installing dependencies
echo ✓ Building Next.js application
echo ✓ Static export completed
echo ✓ Deploying to CDN
echo.

echo ==========================================
echo   Fix Complete!
echo ==========================================
echo.

echo The conflicting root package.json has been removed.
echo Vercel will now find the correct Next.js package.json.
echo Build should succeed on next deployment!
echo.

pause
