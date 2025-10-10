@echo off
REM Git Configuration Script for GitHub
REM This script helps configure Git with proper email and name for GitHub

echo ==========================================
echo   Git Configuration for GitHub
echo ==========================================
echo.

echo Current Git Configuration:
echo.
git config --get user.name
git config --get user.email
echo.

echo To fix GitHub commit issues, you need to configure Git with:
echo 1. Your real name
echo 2. Your GitHub email address
echo.

echo Please run these commands with YOUR information:
echo.
echo git config --global user.name "Your Real Name"
echo git config --global user.email "your-email@example.com"
echo.

echo IMPORTANT: Use the SAME email address that you use for GitHub!
echo.

echo After configuring, you can:
echo 1. Make a small change to test
echo 2. Commit and push to verify it works
echo.

echo Example:
echo git config --global user.name "John Doe"
echo git config --global user.email "john.doe@gmail.com"
echo.

pause
