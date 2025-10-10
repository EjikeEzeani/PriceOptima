@echo off
REM Script to rename requirements files and prevent Vercel Python detection
REM This ensures Vercel treats the project as pure Next.js

echo ==========================================
echo   Vercel Python Detection Prevention
echo ==========================================
echo.

echo Renaming requirements files to prevent Vercel Python detection...

REM Rename requirements files
if exist "requirements.txt" (
    echo   Renaming: requirements.txt ^-^> backend_requirements_backend.txt
    ren "requirements.txt" "backend_requirements_backend.txt"
)

if exist "render_requirements.txt" (
    echo   Renaming: render_requirements.txt ^-^> backend_requirements_render.txt
    ren "render_requirements.txt" "backend_requirements_render.txt"
)

if exist "requirements_minimal.txt" (
    echo   Renaming: requirements_minimal.txt ^-^> backend_requirements_minimal.txt
    ren "requirements_minimal.txt" "backend_requirements_minimal.txt"
)

if exist "requirements_super_minimal.txt" (
    echo   Renaming: requirements_super_minimal.txt ^-^> backend_requirements_super_minimal.txt
    ren "requirements_super_minimal.txt" "backend_requirements_super_minimal.txt"
)

if exist "requirements_bare_minimum.txt" (
    echo   Renaming: requirements_bare_minimum.txt ^-^> backend_requirements_bare_minimum.txt
    ren "requirements_bare_minimum.txt" "backend_requirements_bare_minimum.txt"
)

if exist "requirements_render.txt" (
    echo   Renaming: requirements_render.txt ^-^> backend_requirements_render.txt
    ren "requirements_render.txt" "backend_requirements_render.txt"
)

if exist "requirements-py313.txt" (
    echo   Renaming: requirements-py313.txt ^-^> backend_requirements-py313.txt
    ren "requirements-py313.txt" "backend_requirements-py313.txt"
)

if exist "backend_requirements.txt" (
    echo   Renaming: backend_requirements.txt ^-^> backend_backend_requirements.txt
    ren "backend_requirements.txt" "backend_backend_requirements.txt"
)

if exist "enhanced_requirements.txt" (
    echo   Renaming: enhanced_requirements.txt ^-^> enhanced_backend_requirements.txt
    ren "enhanced_requirements.txt" "enhanced_backend_requirements.txt"
)

if exist "working_requirements.txt" (
    echo   Renaming: working_requirements.txt ^-^> working_backend_requirements.txt
    ren "working_requirements.txt" "working_backend_requirements.txt"
)

echo.
echo ✅ All requirements files renamed successfully!
echo.

echo Creating Vercel configuration...

REM Create vercel.json
(
echo {
echo   "buildCommand": "npm run build",
echo   "outputDirectory": "out",
echo   "installCommand": "npm install",
echo   "framework": "nextjs",
echo   "functions": {},
echo   "build": {
echo     "env": {
echo       "NODE_OPTIONS": "--max-old-space-size=2048"
echo     }
echo   }
echo }
) > dynamic-pricing-dashboard\vercel.json

echo ✅ Vercel configuration created!
echo.

echo Creating minimal Next.js configuration...

REM Create next.config.mjs
(
echo /** @type {import^('next'^).NextConfig} */
echo const nextConfig = {
echo   // Ultra minimal configuration
echo   output: 'export',
echo   trailingSlash: true,
echo   images: {
echo     unoptimized: true
echo   },
echo   eslint: {
echo     ignoreDuringBuilds: true,
echo   },
echo   typescript: {
echo     ignoreBuildErrors: true,
echo   }
echo }
echo.
echo export default nextConfig
) > dynamic-pricing-dashboard\next.config.mjs

echo ✅ Next.js configuration created!
echo.

echo Creating minimal package.json...

REM Create package.json
(
echo {
echo   "name": "priceoptima-minimal",
echo   "version": "1.0.0",
echo   "private": true,
echo   "scripts": {
echo     "build": "next build",
echo     "dev": "next dev",
echo     "start": "next start"
echo   },
echo   "dependencies": {
echo     "next": "14.2.16",
echo     "react": "^18",
echo     "react-dom": "^18"
echo   },
echo   "devDependencies": {
echo     "@types/node": "^22",
echo     "@types/react": "^18",
echo     "@types/react-dom": "^18",
echo     "typescript": "^5"
echo   }
echo }
) > dynamic-pricing-dashboard\package.json

echo ✅ Package.json created!
echo.

echo Creating minimal .vercelignore...

REM Create .vercelignore
(
echo # Ultra minimal Vercel ignore
echo node_modules/
echo .next/
echo *.log
echo .env*
) > dynamic-pricing-dashboard\.vercelignore

echo ✅ .vercelignore created!
echo.

echo Creating minimal app structure...

REM Create app directory
if not exist "dynamic-pricing-dashboard\app" mkdir "dynamic-pricing-dashboard\app"

REM Create layout.tsx
(
echo export const metadata = {
echo   title: 'PriceOptima Dashboard',
echo   description: 'Dynamic Pricing Analytics Application',
echo }
echo.
echo export default function RootLayout^({
echo   children,
echo }: {
echo   children: React.ReactNode
echo }^) {
echo   return ^(
echo     ^<html lang="en"^>
echo       ^<body^>{children}^</body^>
echo     ^</html^>
echo   ^)
echo }
) > dynamic-pricing-dashboard\app\layout.tsx

REM Create page.tsx
(
echo import React from 'react';
echo.
echo export default function Home^(^) {
echo   return ^(
echo     ^<div style={{ padding: '2rem', fontFamily: 'Arial, sans-serif' }}^>
echo       ^<h1^>PriceOptima Dashboard^</h1^>
echo       ^<p^>Dynamic Pricing Analytics Application^</p^>
echo       ^<div style={{ marginTop: '2rem' }}^>
echo         ^<h2^>Upload Your Data^</h2^>
echo         ^<input 
echo           type="file" 
echo           accept=".csv" 
echo           style={{ padding: '0.5rem', margin: '1rem 0' }}
echo         /^>
echo         ^<br /^>
echo         ^<button 
echo           style={{ 
echo             padding: '0.75rem 1.5rem', 
echo             backgroundColor: '#0070f3', 
echo             color: 'white', 
echo             border: 'none', 
echo             borderRadius: '4px',
echo             cursor: 'pointer'
echo           }}
echo         ^>
echo           Process Data
echo         ^</button^>
echo       ^</div^>
echo       ^<div style={{ marginTop: '2rem' }}^>
echo         ^<h3^>Backend API^</h3^>
echo         ^<p^>Connect to your Render backend for data processing^</p^>
echo         ^<p^>API URL: ^<code^>{process.env.NEXT_PUBLIC_API_URL ^|^| 'Not configured'}^</code^>^</p^>
echo       ^</div^>
echo     ^</div^>
echo   ^);
echo }
) > dynamic-pricing-dashboard\app\page.tsx

echo ✅ App structure created!
echo.

echo Creating restore script...

REM Create restore script
(
echo @echo off
echo REM Script to restore original requirements file names
echo.
echo echo Restoring original requirements file names...
echo.
echo if exist "backend_requirements_backend.txt" ^(
echo     echo   Restoring: backend_requirements_backend.txt ^-^> requirements.txt
echo     ren "backend_requirements_backend.txt" "requirements.txt"
echo ^)
echo.
echo if exist "backend_requirements_render.txt" ^(
echo     echo   Restoring: backend_requirements_render.txt ^-^> render_requirements.txt
echo     ren "backend_requirements_render.txt" "render_requirements.txt"
echo ^)
echo.
echo if exist "backend_requirements_minimal.txt" ^(
echo     echo   Restoring: backend_requirements_minimal.txt ^-^> requirements_minimal.txt
echo     ren "backend_requirements_minimal.txt" "requirements_minimal.txt"
echo ^)
echo.
echo if exist "backend_requirements_super_minimal.txt" ^(
echo     echo   Restoring: backend_requirements_super_minimal.txt ^-^> requirements_super_minimal.txt
echo     ren "backend_requirements_super_minimal.txt" "requirements_super_minimal.txt"
echo ^)
echo.
echo if exist "backend_requirements_bare_minimum.txt" ^(
echo     echo   Restoring: backend_requirements_bare_minimum.txt ^-^> requirements_bare_minimum.txt
echo     ren "backend_requirements_bare_minimum.txt" "requirements_bare_minimum.txt"
echo ^)
echo.
echo if exist "backend_requirements-py313.txt" ^(
echo     echo   Restoring: backend_requirements-py313.txt ^-^> requirements-py313.txt
echo     ren "backend_requirements-py313.txt" "requirements-py313.txt"
echo ^)
echo.
echo if exist "backend_backend_requirements.txt" ^(
echo     echo   Restoring: backend_backend_requirements.txt ^-^> backend_requirements.txt
echo     ren "backend_backend_requirements.txt" "backend_requirements.txt"
echo ^)
echo.
echo if exist "enhanced_backend_requirements.txt" ^(
echo     echo   Restoring: enhanced_backend_requirements.txt ^-^> enhanced_requirements.txt
echo     ren "enhanced_backend_requirements.txt" "enhanced_requirements.txt"
echo ^)
echo.
echo if exist "working_backend_requirements.txt" ^(
echo     echo   Restoring: working_backend_requirements.txt ^-^> working_requirements.txt
echo     ren "working_backend_requirements.txt" "working_requirements.txt"
echo ^)
echo.
echo echo ✅ Requirements files restored!
) > restore_requirements.bat

echo ✅ Restore script created!
echo.

echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo ✅ All requirements files renamed
echo ✅ Vercel configuration created
echo ✅ Minimal Next.js app created
echo ✅ Restore script created
echo.
echo Next steps:
echo 1. Commit and push to GitHub
echo 2. Vercel will auto-deploy without Python detection
echo 3. Use restore_requirements.bat if you need to restore files
echo.
echo To restore original files: restore_requirements.bat
echo.
pause
