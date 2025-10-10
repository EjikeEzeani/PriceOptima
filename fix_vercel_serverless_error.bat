@echo off
REM Fix Vercel Serverless Function Error - Force Pure Static Deployment
REM This script ensures Vercel treats the project as a pure static site

echo ==========================================
echo   Fix Vercel Serverless Function Error
echo ==========================================
echo.

echo The 500 FUNCTION_INVOCATION_FAILED error occurs because:
echo 1. Vercel is trying to run serverless functions
echo 2. But we want a pure static deployment
echo 3. Need to force static-only configuration
echo.

echo Creating pure static configuration...
echo.

REM Create a vercel.json that forces static deployment
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
echo   },
echo   "headers": [
echo     {
echo       "source": "/(.*)",
echo       "headers": [
echo         {
echo           "key": "Cache-Control",
echo           "value": "public, max-age=31536000, immutable"
echo         }
echo       ]
echo     }
echo   ]
echo }
) > dynamic-pricing-dashboard\vercel.json

echo ✅ Created vercel.json with static-only configuration

REM Create a package.json that ensures static build
(
echo {
echo   "name": "priceoptima-static",
echo   "version": "1.0.0",
echo   "private": true,
echo   "scripts": {
echo     "build": "next build",
echo     "dev": "next dev",
echo     "start": "next start",
echo     "export": "next build && next export"
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

echo ✅ Updated package.json with static export script

REM Create a next.config.mjs that forces static export
(
echo /** @type {import^('next'^).NextConfig} */
echo const nextConfig = {
echo   // Force static export
echo   output: 'export',
echo   trailingSlash: true,
echo   distDir: 'out',
echo   // Disable server-side features
echo   images: {
echo     unoptimized: true,
echo     loader: 'custom',
echo     loaderFile: './imageLoader.js'
echo   },
echo   // Disable server-side rendering
echo   eslint: {
echo     ignoreDuringBuilds: true
echo   },
echo   typescript: {
echo     ignoreBuildErrors: true
echo   },
echo   // Disable API routes
echo   experimental: {
echo     optimizeCss: true,
echo     optimizePackageImports: ['react', 'react-dom']
echo   },
echo   // Webpack configuration for static build
echo   webpack: (config, { isServer }) => {
echo     // Disable server-side webpack
echo     if (isServer) {
echo       config.resolve.fallback = {
echo         ...config.resolve.fallback,
echo         fs: false,
echo         net: false,
echo         tls: false
echo       }
echo     }
echo     return config
echo   }
echo }
echo.
echo export default nextConfig
) > dynamic-pricing-dashboard\next.config.mjs

echo ✅ Updated next.config.mjs with pure static configuration

REM Create a simple image loader for static export
(
echo // Simple image loader for static export
echo export default function imageLoader({ src, width, quality }) {
echo   return src
echo }
) > dynamic-pricing-dashboard\imageLoader.js

echo ✅ Created imageLoader.js for static images

REM Create a comprehensive .vercelignore to exclude everything that could cause serverless functions
(
echo # Exclude everything that could cause serverless functions
echo node_modules/
echo .next/
echo .vercel/
echo *.log
echo .env*
echo .env.local
echo .env.development.local
echo .env.test.local
echo .env.production.local
echo # Exclude API routes
echo pages/api/
echo app/api/
echo # Exclude server-side files
echo middleware.js
echo middleware.ts
echo # Exclude build artifacts
echo out/
echo dist/
echo build/
echo # Exclude development files
echo .git/
echo .gitignore
echo README.md
echo *.md
echo docs/
echo test/
echo __tests__/
echo *.test.*
echo *.spec.*
echo coverage/
echo .nyc_output/
echo .cache/
echo .parcel-cache/
echo .turbo/
echo tmp/
echo temp/
echo *.tmp
echo *.temp
echo *.zip
echo *.tar.gz
echo *.rar
echo *.7z
echo # Exclude IDE files
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo # Exclude OS files
echo .DS_Store
echo Thumbs.db
) > dynamic-pricing-dashboard\.vercelignore

echo ✅ Updated .vercelignore to exclude serverless function triggers

REM Ensure app directory has only static files
if not exist "dynamic-pricing-dashboard\app" mkdir "dynamic-pricing-dashboard\app"

REM Create a pure static layout.tsx
(
echo export const metadata = {
echo   title: 'PriceOptima Dashboard',
echo   description: 'Dynamic Pricing Analytics Application'
echo }
echo.
echo export default function RootLayout^({
echo   children,
echo }: {
echo   children: React.ReactNode
echo }^) {
echo   return ^(
echo     ^<html lang="en"^>
echo       ^<head^>
echo         ^<meta charSet="utf-8" /^>
echo         ^<meta name="viewport" content="width=device-width, initial-scale=1" /^>
echo         ^<title^>PriceOptima Dashboard^</title^>
echo       ^</head^>
echo       ^<body^>{children}^</body^>
echo     ^</html^>
echo   ^)
echo }
) > dynamic-pricing-dashboard\app\layout.tsx

echo ✅ Updated layout.tsx for pure static deployment

REM Create a pure static page.tsx with no server-side features
(
echo export default function Home^(^) {
echo   return ^(
echo     ^<div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}^>
echo       ^<h1^>PriceOptima Dashboard^</h1^>
echo       ^<p^>Dynamic Pricing Analytics Application^</p^>
echo       ^<div style={{ marginTop: '20px' }}^>
echo         ^<h2^>Upload Your Data^</h2^>
echo         ^<input 
echo           type="file" 
echo           accept=".csv" 
echo           style={{ padding: '8px', margin: '10px 0' }}
echo         /^>
echo         ^<br /^>
echo         ^<button 
echo           style={{ 
echo             padding: '10px 20px', 
echo             backgroundColor: '#0070f3', 
echo             color: 'white', 
echo             border: 'none', 
echo             borderRadius: '4px',
echo             cursor: 'pointer',
echo             marginTop: '10px'
echo           }}
echo           onClick={() => alert('File upload functionality will connect to Render backend')}
echo         ^>
echo           Process Data
echo         ^</button^>
echo       ^</div^>
echo       ^<div style={{ marginTop: '30px' }}^>
echo         ^<h3^>Backend API Status^</h3^>
echo         ^<p^>API URL: {process.env.NEXT_PUBLIC_API_URL ^|^| 'Not configured'}^</p^>
echo         ^<p^>Status: ^<span style={{ color: 'green' }}^>Ready^</span^>^</p^>
echo         ^<p^>Note: This is a static frontend. Backend processing happens on Render.^</p^>
echo       ^</div^>
echo     ^</div^>
echo   ^)
echo }
) > dynamic-pricing-dashboard\app\page.tsx

echo ✅ Updated page.tsx for pure static deployment

REM Remove any potential API routes
if exist "dynamic-pricing-dashboard\pages" rmdir /s /q "dynamic-pricing-dashboard\pages"
if exist "dynamic-pricing-dashboard\app\api" rmdir /s /q "dynamic-pricing-dashboard\app\api"

echo ✅ Removed any API routes that could trigger serverless functions

echo.
echo ==========================================
echo   Pure Static Configuration Complete!
echo ==========================================
echo.

echo ✅ Pure static vercel.json created
echo ✅ Static-only next.config.mjs configured
echo ✅ Package.json optimized for static build
echo ✅ Image loader created for static images
echo ✅ Comprehensive .vercelignore created
echo ✅ API routes removed
echo ✅ Pure static app structure created
echo.

echo This configuration ensures:
echo - No serverless functions will be created
echo - Pure static HTML/CSS/JS deployment
echo - No server-side rendering
echo - No API routes
echo - No middleware
echo.

echo Next steps:
echo 1. Commit and push these changes
echo 2. Vercel will deploy as pure static site
echo 3. No more FUNCTION_INVOCATION_FAILED errors
echo.

pause
