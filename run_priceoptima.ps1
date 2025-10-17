# PriceOptima PowerShell Launcher
# Run with: .\run_priceoptima.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "🎯 PRICEOPTIMA - DYNAMIC PRICING OPTIMIZATION" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Starting PriceOptima Application..." -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "main.py")) {
    Write-Host "❌ main.py not found. Please run this from the PriceOptima directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

if (Test-Path "venv_new\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & "venv_new\Scripts\Activate.ps1"
}

# Install requirements if needed
if (Test-Path "requirements.txt") {
    Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow
    
    try {
        python -c "import streamlit" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "📦 Installing requirements..." -ForegroundColor Yellow
            pip install -r requirements.txt
        } else {
            Write-Host "✅ Dependencies already installed" -ForegroundColor Green
        }
    } catch {
        Write-Host "📦 Installing requirements..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
}

# Run the main application
Write-Host "🚀 Launching PriceOptima..." -ForegroundColor Green
Write-Host ""
python main.py

Write-Host ""
Write-Host "👋 PriceOptima session ended." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
