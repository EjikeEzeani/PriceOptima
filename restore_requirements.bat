@echo off
REM Script to restore original requirements file names

echo Restoring original requirements file names...

if exist "backend_requirements_backend.txt" (
    echo   Restoring: backend_requirements_backend.txt -> requirements.txt
    ren "backend_requirements_backend.txt" "requirements.txt"
)

if exist "backend_requirements_render.txt" (
    echo   Restoring: backend_requirements_render.txt -> render_requirements.txt
    ren "backend_requirements_render.txt" "render_requirements.txt"
)

if exist "backend_requirements_minimal.txt" (
    echo   Restoring: backend_requirements_minimal.txt -> requirements_minimal.txt
    ren "backend_requirements_minimal.txt" "requirements_minimal.txt"
)

if exist "backend_requirements_super_minimal.txt" (
    echo   Restoring: backend_requirements_super_minimal.txt -> requirements_super_minimal.txt
    ren "backend_requirements_super_minimal.txt" "requirements_super_minimal.txt"
)

if exist "backend_requirements_bare_minimum.txt" (
    echo   Restoring: backend_requirements_bare_minimum.txt -> requirements_bare_minimum.txt
    ren "backend_requirements_bare_minimum.txt" "requirements_bare_minimum.txt"
)

if exist "backend_requirements-py313.txt" (
    echo   Restoring: backend_requirements-py313.txt -> requirements-py313.txt
    ren "backend_requirements-py313.txt" "requirements-py313.txt"
)

if exist "backend_backend_requirements.txt" (
    echo   Restoring: backend_backend_requirements.txt -> backend_requirements.txt
    ren "backend_backend_requirements.txt" "backend_requirements.txt"
)

if exist "enhanced_backend_requirements.txt" (
    echo   Restoring: enhanced_backend_requirements.txt -> enhanced_requirements.txt
    ren "enhanced_backend_requirements.txt" "enhanced_requirements.txt"
)

if exist "working_backend_requirements.txt" (
    echo   Restoring: working_backend_requirements.txt -> working_requirements.txt
    ren "working_backend_requirements.txt" "working_requirements.txt"
)

echo ✅ Requirements files restored!
