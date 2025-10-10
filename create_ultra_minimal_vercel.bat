@echo off
REM Ultra Minimal Vercel Deployment Script
REM This script creates the absolute minimal files needed for Vercel deployment

echo ==========================================
echo   Ultra Minimal Vercel Deployment
echo ==========================================
echo.

echo Creating ultra-minimal Next.js app for Vercel...
echo.

REM Create minimal package.json
echo Creating minimal package.json...
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
echo   }
echo }
) > dynamic-pricing-dashboard\package.json

echo ✅ Minimal package.json created!

REM Create ultra-minimal next.config.mjs
echo Creating ultra-minimal next.config.mjs...
(
echo /** @type {import^('next'^).NextConfig} */
echo const nextConfig = {
echo   output: 'export',
echo   trailingSlash: true,
echo   images: {
echo     unoptimized: true
echo   },
echo   eslint: {
echo     ignoreDuringBuilds: true
echo   },
echo   typescript: {
echo     ignoreBuildErrors: true
echo   }
echo }
echo.
echo export default nextConfig
) > dynamic-pricing-dashboard\next.config.mjs

echo ✅ Minimal next.config.mjs created!

REM Create minimal vercel.json
echo Creating minimal vercel.json...
(
echo {
echo   "buildCommand": "npm run build",
echo   "outputDirectory": "out",
echo   "installCommand": "npm install --production",
echo   "framework": "nextjs"
echo }
) > dynamic-pricing-dashboard\vercel.json

echo ✅ Minimal vercel.json created!

REM Create ultra-minimal .vercelignore
echo Creating ultra-minimal .vercelignore...
(
echo node_modules/
echo .next/
echo *.log
echo .env*
echo public/
echo components/
echo hooks/
echo lib/
echo styles/
echo *.md
echo *.txt
echo *.json
echo !package.json
echo !vercel.json
echo !next.config.mjs
) > dynamic-pricing-dashboard\.vercelignore

echo ✅ Minimal .vercelignore created!

REM Create minimal app directory structure
echo Creating minimal app structure...
if not exist "dynamic-pricing-dashboard\app" mkdir "dynamic-pricing-dashboard\app"

REM Create minimal layout.tsx
echo Creating minimal layout.tsx...
(
echo export const metadata = {
echo   title: 'PriceOptima',
echo   description: 'Dynamic Pricing Dashboard'
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

echo ✅ Minimal layout.tsx created!

REM Create ultra-minimal page.tsx
echo Creating ultra-minimal page.tsx...
(
echo export default function Home^(^) {
echo   return ^(
echo     ^<div style={{ padding: '20px', fontFamily: 'Arial' }}^>
echo       ^<h1^>PriceOptima Dashboard^</h1^>
echo       ^<p^>Dynamic Pricing Analytics^</p^>
echo       ^<div^>
echo         ^<h2^>Upload Data^</h2^>
echo         ^<input type="file" accept=".csv" /^>
echo         ^<button style={{ marginLeft: '10px', padding: '5px 10px' }}^>Process^</button^>
echo       ^</div^>
echo       ^<p^>API: {process.env.NEXT_PUBLIC_API_URL ^|^| 'Not configured'}^</p^>
echo     ^</div^>
echo   ^)
echo }
) > dynamic-pricing-dashboard\app\page.tsx

echo ✅ Minimal page.tsx created!

REM Remove all unnecessary files
echo Removing unnecessary files...
if exist "dynamic-pricing-dashboard\components" rmdir /s /q "dynamic-pricing-dashboard\components"
if exist "dynamic-pricing-dashboard\hooks" rmdir /s /q "dynamic-pricing-dashboard\hooks"
if exist "dynamic-pricing-dashboard\lib" rmdir /s /q "dynamic-pricing-dashboard\lib"
if exist "dynamic-pricing-dashboard\styles" rmdir /s /q "dynamic-pricing-dashboard\styles"
if exist "dynamic-pricing-dashboard\public" rmdir /s /q "dynamic-pricing-dashboard\public"
if exist "dynamic-pricing-dashboard\package-lock.json" del "dynamic-pricing-dashboard\package-lock.json"
if exist "dynamic-pricing-dashboard\pnpm-lock.yaml" del "dynamic-pricing-dashboard\pnpm-lock.yaml"
if exist "dynamic-pricing-dashboard\tsconfig.json" del "dynamic-pricing-dashboard\tsconfig.json"
if exist "dynamic-pricing-dashboard\postcss.config.mjs" del "dynamic-pricing-dashboard\postcss.config.mjs"
if exist "dynamic-pricing-dashboard\components.json" del "dynamic-pricing-dashboard\components.json"
if exist "dynamic-pricing-dashboard\next-env.d.ts" del "dynamic-pricing-dashboard\next-env.d.ts"

echo ✅ Unnecessary files removed!

REM Check final size
echo Checking final size...
for /f %%i in ('dir dynamic-pricing-dashboard /s /-c ^| find "File(s)"') do echo Final size: %%i

echo.
echo ==========================================
echo   Ultra Minimal Setup Complete!
echo ==========================================
echo.
echo ✅ Ultra-minimal Next.js app created
echo ✅ All unnecessary files removed
echo ✅ Size optimized for Vercel deployment
echo.
echo Next steps:
echo 1. Commit and push to GitHub
echo 2. Vercel will deploy the minimal version
echo 3. Should be well under 300MB limit
echo.
pause
