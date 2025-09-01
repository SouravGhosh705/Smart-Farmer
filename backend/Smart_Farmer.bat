@echo off
REM ================================================================================
REM Smart Farmer - Desktop Shortcut
REM ================================================================================
REM Quick access to Smart Farmer AI Agricultural Backend
REM ================================================================================

title Smart Farmer - AI Agricultural Backend

REM Change to the backend directory
cd /d "%~dp0"

REM Check if we're in the right directory
if not exist "fastapi_app.py" (
    echo ❌ Smart Farmer files not found in current directory
    echo Please ensure this shortcut is in the backend folder
    pause
    exit /b 1
)

REM Display welcome message
cls
echo.
echo ===============================================================================
echo                    🌾 SMART FARMER - AI AGRICULTURAL BACKEND 🌾
echo ===============================================================================
echo                              Welcome to Smart Farmer!
echo                     Your AI-Powered Agricultural Assistant
echo ===============================================================================
echo.

color 0A

echo 🚀 Quick Options:
echo.
echo 1. Start Smart Farmer Backend (Recommended)
echo 2. Open Main Menu (Full Options)
echo 3. Quick Test
echo 4. Open API Documentation
echo.
set /p quick_choice="Enter your choice (1-4): "

if "%quick_choice%"=="1" (
    echo.
    echo 🔥 Starting Smart Farmer Backend...
    call smartfarmer.bat
) else if "%quick_choice%"=="2" (
    call smartfarmer_menu.bat
) else if "%quick_choice%"=="3" (
    call test_smartfarmer.bat
) else if "%quick_choice%"=="4" (
    echo 🌐 Starting backend and opening documentation...
    start /B python fastapi_app.py
    timeout /t 5 /nobreak >nul
    start http://localhost:8000/docs
) else (
    echo 🚀 Starting Smart Farmer Backend (default)...
    call smartfarmer.bat
)

exit /b 0
