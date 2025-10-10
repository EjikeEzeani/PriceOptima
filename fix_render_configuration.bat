@echo off
REM Fix Render Configuration for Renamed Requirements Files
REM This script updates all Render configuration files to use the renamed requirements files

echo ==========================================
echo   Fix Render Configuration
echo ==========================================
echo.

echo Updating Render configuration files to use renamed requirements files...
echo.

REM Update RENDER_SUPER_MINIMAL.md
echo Updating RENDER_SUPER_MINIMAL.md...
powershell -Command "(Get-Content 'RENDER_SUPER_MINIMAL.md') -replace 'requirements_super_minimal.txt', 'backend_requirements_super_minimal.txt' | Set-Content 'RENDER_SUPER_MINIMAL.md'"

REM Update RENDER_BARE_MINIMUM.md
echo Updating RENDER_BARE_MINIMUM.md...
powershell -Command "(Get-Content 'RENDER_BARE_MINIMUM.md') -replace 'requirements_bare_minimum.txt', 'backend_requirements_bare_minimum.txt' | Set-Content 'RENDER_BARE_MINIMUM.md'"

REM Update RENDER_ULTRA_LIGHT.md
echo Updating RENDER_ULTRA_LIGHT.md...
powershell -Command "(Get-Content 'RENDER_ULTRA_LIGHT.md') -replace 'requirements_minimal.txt', 'backend_requirements_minimal.txt' | Set-Content 'RENDER_ULTRA_LIGHT.md'"

REM Update RENDER_DEPLOYMENT.md
echo Updating RENDER_DEPLOYMENT.md...
powershell -Command "(Get-Content 'RENDER_DEPLOYMENT.md') -replace 'render_requirements.txt', 'backend_requirements_render.txt' | Set-Content 'RENDER_DEPLOYMENT.md'"

REM Update RENDER_CONFIG.md
echo Updating RENDER_CONFIG.md...
powershell -Command "(Get-Content 'RENDER_CONFIG.md') -replace 'requirements_render.txt', 'backend_requirements_render.txt' | Set-Content 'RENDER_CONFIG.md'"

echo.
echo ✅ All Render configuration files updated!
echo.

echo Creating updated deployment instructions...
echo.

REM Create a comprehensive Render fix document
(
echo # Render Deployment Fix - Updated File Names
echo.
echo ## Problem: Render Cannot Find Requirements Files
echo After renaming requirements files to prevent Vercel Python detection, Render can no longer find them.
echo.
echo ## Solution: Updated Configuration
echo.
echo ### For Super Minimal Backend:
echo - **Requirements File**: `backend_requirements_super_minimal.txt`
echo - **Build Command**: `pip install -r backend_requirements_super_minimal.txt`
echo - **Start Command**: `python -m uvicorn render_super_minimal:app --host 0.0.0.0 --port $PORT`
echo.
echo ### For Bare Minimum Backend:
echo - **Requirements File**: `backend_requirements_bare_minimum.txt`
echo - **Build Command**: `pip install -r backend_requirements_bare_minimum.txt`
echo - **Start Command**: `python -m uvicorn render_bare_minimum:app --host 0.0.0.0 --port $PORT`
echo.
echo ### For Ultra Light Backend:
echo - **Requirements File**: `backend_requirements_minimal.txt`
echo - **Build Command**: `pip install -r backend_requirements_minimal.txt`
echo - **Start Command**: `python -m uvicorn render_ultra_light:app --host 0.0.0.0 --port $PORT`
echo.
echo ### For Render Optimized Backend:
echo - **Requirements File**: `backend_requirements_render.txt`
echo - **Build Command**: `pip install -r backend_requirements_render.txt`
echo - **Start Command**: `python -m uvicorn render_optimized_backend:app --host 0.0.0.0 --port $PORT`
echo.
echo ## Environment Variables:
echo ```
echo PYTHONUNBUFFERED=1
echo MALLOC_TRIM_THRESHOLD_=100000
echo MALLOC_MMAP_THRESHOLD_=131072
echo ```
echo.
echo ## Root Directory: ^(empty^)
echo.
echo ## This Fixes the Render Build Error!
) > RENDER_FIX_UPDATED.md

echo ✅ Updated deployment instructions created!
echo.

echo ==========================================
echo   Render Configuration Fixed!
echo ==========================================
echo.
echo ✅ All Render config files updated with correct file names
echo ✅ Build commands now reference renamed requirements files
echo ✅ Render should now find the requirements files
echo.
echo Next steps:
echo 1. Commit and push these changes
echo 2. Update your Render service configuration
echo 3. Redeploy on Render
echo.
echo Updated configurations:
echo - Super Minimal: backend_requirements_super_minimal.txt
echo - Bare Minimum: backend_requirements_bare_minimum.txt
echo - Ultra Light: backend_requirements_minimal.txt
echo - Render Optimized: backend_requirements_render.txt
echo.
pause
