# PowerShell script to start the Dynamic Pricing Analytics Application
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Dynamic Pricing Analytics App" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("127.0.0.1", $Port)
        $connection.Close()
        return $true
    }
    catch {
        return $false
    }
}

# Start Backend
Write-Host "[1/4] Starting Backend Server..." -ForegroundColor Yellow
if (Test-Port 8000) {
    Write-Host "Backend already running on port 8000" -ForegroundColor Green
} else {
    Start-Process -FilePath "python" -ArgumentList "working_backend.py" -WindowStyle Normal
    Write-Host "Backend server started" -ForegroundColor Green
}

# Wait for backend
Write-Host "[2/4] Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Test backend connection
Write-Host "[3/4] Testing backend connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend is running successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Backend health check failed" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Backend connection failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Start Frontend
Write-Host "[4/4] Starting Frontend Server..." -ForegroundColor Yellow
Set-Location "dynamic-pricing-dashboard"
if (Test-Port 3000) {
    Write-Host "Frontend already running on port 3000" -ForegroundColor Green
} else {
    Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WindowStyle Normal
    Write-Host "Frontend server started" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Application Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Backend:  http://127.0.0.1:8000" -ForegroundColor White
Write-Host "Frontend: http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "Test Data: test_data.csv (available in project root)" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Instructions" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Wait for both servers to fully start" -ForegroundColor White
Write-Host "2. Open http://localhost:3000 in your browser" -ForegroundColor White
Write-Host "3. Upload the test_data.csv file" -ForegroundColor White
Write-Host "4. Check browser console for any errors" -ForegroundColor White
Write-Host ""

# Open browser
Write-Host "Opening application in browser..." -ForegroundColor Yellow
Start-Process "http://localhost:3000"

Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
