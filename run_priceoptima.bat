@echo off
echo ================================================
echo 🎯 PRICEOPTIMA - DYNAMIC PRICING OPTIMIZATION
echo ================================================
echo.
echo 🚀 Starting PriceOptima Application...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python first.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "main.py" (
    echo ❌ main.py not found. Please run this from the PriceOptima directory.
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
)

if exist "venv_new\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call venv_new\Scripts\activate.bat
)

REM Install requirements if needed
if exist "requirements.txt" (
    echo 📦 Checking dependencies...
    python -c "import streamlit" >nul 2>&1
    if errorlevel 1 (
        echo 📦 Installing requirements...
        pip install -r requirements.txt
    )
)

REM Run the main application
echo 🚀 Launching PriceOptima...
echo.
python main.py

echo.
echo 👋 PriceOptima session ended.
pause
