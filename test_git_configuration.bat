@echo off
REM Test Git Configuration Script
REM This script tests if Git is properly configured for GitHub

echo ==========================================
echo   Testing Git Configuration
echo ==========================================
echo.

echo Current Git Configuration:
echo.
echo Name: 
git config --get user.name
echo.
echo Email: 
git config --get user.email
echo.

echo Testing Git Connection to GitHub...
echo.

REM Test if we can connect to GitHub
git ls-remote origin
if %errorlevel% equ 0 (
    echo ✅ Git connection to GitHub is working!
) else (
    echo ❌ Git connection to GitHub failed!
    echo.
    echo Possible issues:
    echo 1. Invalid email address
    echo 2. Authentication problems
    echo 3. Network issues
)

echo.
echo Testing commit capability...
echo.

REM Create a test file
echo Test file for Git configuration > test_git_config.txt

REM Add and commit the test file
git add test_git_config.txt
git commit -m "Test commit for Git configuration"

if %errorlevel% equ 0 (
    echo ✅ Git commit is working!
    
    REM Try to push
    git push origin main
    if %errorlevel% equ 0 (
        echo ✅ Git push to GitHub is working!
        echo.
        echo Your Git configuration is correct!
    ) else (
        echo ❌ Git push to GitHub failed!
        echo.
        echo This indicates an authentication or email issue.
        echo Please check your GitHub email configuration.
    )
    
    REM Clean up test file
    git reset --hard HEAD~1
    del test_git_config.txt
    
) else (
    echo ❌ Git commit failed!
    echo.
    echo This indicates a configuration issue.
    echo Please run configure_git_for_github.bat first.
)

echo.
pause
