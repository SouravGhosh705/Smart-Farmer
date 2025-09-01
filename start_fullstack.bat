@echo off
REM ================================================================================
REM Smart Farmer - Full Stack Launcher (Backend + Frontend)
REM ================================================================================
REM Version: 3.0.0
REM Description: Starts both FastAPI backend and React frontend
REM ================================================================================

title Smart Farmer - Full Stack Application

echo.
echo ===============================================================================
echo                    🌾 SMART FARMER - FULL STACK LAUNCHER 🌾
echo ===============================================================================
echo                                Version 3.0.0
echo                      AI Agricultural Platform - Full Stack
echo ===============================================================================
echo.

REM Set color for better visibility
color 0B

REM Check if Node.js is installed
echo 🔍 Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    echo.
    echo 🔧 You can continue with backend only, or install Node.js for full stack
    set /p choice="Continue with backend only? (Y/N): "
    if /i "%choice%"=="N" (
        pause
        exit /b 1
    )
    set frontend_available=false
) else (
    echo ✅ Node.js is available
    node --version
    set frontend_available=true
)

REM Check if Python is installed
echo 🔍 Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is available
python --version
echo.

REM Check directories
if not exist "backend\fastapi_app.py" (
    echo ❌ Backend files not found in backend directory
    pause
    exit /b 1
)

if "%frontend_available%"=="true" (
    if not exist "frontend\package.json" (
        echo ❌ Frontend files not found in frontend directory
        set frontend_available=false
        echo ⚠️ Will start backend only
    )
)

echo ===============================================================================
echo 🚀 STARTING SMART FARMER FULL STACK APPLICATION
echo ===============================================================================
echo.

if "%frontend_available%"=="true" (
    echo 📱 Frontend: http://localhost:3000
)
echo 🖥️ Backend: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.

REM Start Backend in background
echo 🔧 Starting Backend Server...
start "Smart Farmer Backend" cmd /c "cd /d "%~dp0backend" && .\smartfarmer.bat"

REM Wait a moment for backend to initialize
timeout /t 3 /nobreak >nul

if "%frontend_available%"=="true" (
    echo 🎨 Starting Frontend Application...
    echo.
    echo ===============================================================================
    echo                           🌟 FRONTEND STARTING 🌟
    echo ===============================================================================
    echo.
    echo 📱 React development server will open at: http://localhost:3000
    echo 🔄 Hot reload enabled - changes will update automatically
    echo ⏹️ Press Ctrl+C in frontend window to stop frontend
    echo ⏹️ Close backend window to stop backend
    echo.
    
    REM Start Frontend
    cd /d "%~dp0frontend"
    
    REM Check if node_modules exists, if not install dependencies
    if not exist "node_modules" (
        echo 📦 Installing frontend dependencies...
        npm install
        if %errorlevel% neq 0 (
            echo ❌ Failed to install frontend dependencies
            echo 🔧 Trying with --legacy-peer-deps...
            npm install --legacy-peer-deps
        )
    )
    
    echo 🚀 Starting React development server...
    npm start
) else (
    echo.
    echo ===============================================================================
    echo                          ⚠️ BACKEND ONLY MODE ⚠️
    echo ===============================================================================
    echo.
    echo 🖥️ Backend is running at: http://localhost:8000
    echo 📚 API Documentation: http://localhost:8000/docs
    echo.
    echo To install Node.js for frontend:
    echo 1. Visit: https://nodejs.org
    echo 2. Download and install LTS version
    echo 3. Restart this script
    echo.
    echo Press any key to open API documentation...
    pause >nul
    start http://localhost:8000/docs
)

echo.
echo ===============================================================================
echo                          🌾 APPLICATION STOPPED 🌾
echo ===============================================================================
echo.
echo Thank you for using Smart Farmer!
echo.
pause
