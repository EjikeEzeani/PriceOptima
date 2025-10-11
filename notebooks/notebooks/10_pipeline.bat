@echo off
:: ======================================
:: Interactive Pipeline Runner
:: ======================================

:: Activate virtual environment
set VENV_PATH=C:\Users\USER\Downloads\Msc Project\venv\Scripts\activate.bat
if exist "%VENV_PATH%" (
    call "%VENV_PATH%"
) else (
    echo Virtual environment not found. Please create it first.
    exit /b
)

:: Menu options
:menu
echo ======================================
echo M.Sc. Project Pipeline
echo 1. Merge Data
echo 2. Preprocess
echo 3. ML Forecasting
echo 4. RL Environment Setup
echo 5. RL Training
echo 6. Evaluation
echo 7. Dashboard
echo 8. Run All
echo 9. Exit
echo ======================================
set /p choice=Enter choice: 

if "%choice%"=="1" (
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/01_merge.py"
    goto menu
)
if "%choice%"=="2" (
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/02_preprocess.py"
    goto menu
)
if "%choice%"=="3" (
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/03_ml_forecast.py"
    goto menu
)
if "%choice%"=="4" (
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/06_rl_environment_safe.py"
    goto menu
)
if "%choice%"=="5" (
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/07_rl_training_safe.py"
    goto menu
)
if "%choice%"=="6" (
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/08_evaluation_safe.py"
    goto menu
)
if "%choice%"=="7" (
    streamlit run "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/09_dashboard_pipeline_safe.py"
    goto menu
)
if "%choice%"=="8" (
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/01_merge.py"
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/02_preprocess.py"
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/03_ml_forecast.py"
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/06_rl_environment_safe.py"
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/07_rl_training_safe.py"
    python "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/08_evaluation_safe.py"
    streamlit run "c:/Users/USER/Downloads/Msc Project/notebooks/notebooks/09_dashboard_pipeline_safe.py"
    goto menu
)
if "%choice%"=="9" exit /b

echo Invalid choice, try again
goto menu

