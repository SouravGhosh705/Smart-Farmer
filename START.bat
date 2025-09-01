@echo off
title Smart Farmer - Universal Start
color 0A
cls

echo.
echo ========================================================
echo           🌾 SMART FARMER - UNIVERSAL START 🌾
echo ========================================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo 🚀 Starting Smart Farmer Application...
echo.

REM Kill existing processes
echo 🔄 Stopping any existing servers...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Smart Farmer Backend*" >nul 2>&1
taskkill /f /im node.exe /fi "WINDOWTITLE eq Smart Farmer Frontend*" >nul 2>&1
echo ✅ Cleanup completed
echo.

REM Check if Python exists
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Install Python from: https://python.org
    pause
    exit /b 1
)

REM Check if Node.js exists  
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found! Install Node.js from: https://nodejs.org
    pause
    exit /b 1
)

echo ✅ Python and Node.js are available
echo.

REM Start backend
echo 🔧 Starting Backend Server...
cd backend
if not exist app.py (
    echo ❌ app.py not found!
    pause
    exit /b 1
)

start "Smart Farmer Backend" /min cmd /k "title Smart Farmer Backend && python app.py"
echo ✅ Backend starting (may take 30 seconds to train models)
cd ..

REM Start frontend
echo 🌐 Starting Frontend Server...
cd frontend
if not exist package.json (
    echo ❌ package.json not found!
    pause
    exit /b 1
)

REM Install dependencies if node_modules doesn't exist
if not exist node_modules (
    echo 📦 Installing dependencies...
    call npm install --silent
)

set NODE_OPTIONS=--openssl-legacy-provider
start "Smart Farmer Frontend" /min cmd /k "title Smart Farmer Frontend && npm start"
echo ✅ Frontend starting
cd ..

echo.
echo ⏱️  Please wait 30-60 seconds for both servers to fully start...
echo.

REM Wait a bit then check
timeout /t 15 /nobreak >nul

echo 🎯 Checking server status...
netstat -an | find ":8000" | find "LISTENING" >nul
if errorlevel 1 (
    echo ⚠️  Backend may still be starting...
) else (
    echo ✅ Backend is running on port 8000
)

netstat -an | find ":3000" | find "LISTENING" >nul
if errorlevel 1 (
    echo ⚠️  Frontend may still be starting...
) else (
    echo ✅ Frontend is running on port 3000
)

echo.
echo ========================================================
echo 🎉 SMART FARMER IS STARTING!
echo ========================================================
echo.
echo 📱 Your application will be available at:
echo    🌐 http://localhost:3000
echo    🔗 http://localhost:8000
echo.
echo 💡 Features:
echo    ✅ Crop Recommendation (18 crops)
echo    ✅ Yield Prediction
echo    ✅ Weather Integration
echo    ✅ Soil Analysis
echo.
echo ⚠️  IMPORTANT:
echo    • Keep both server windows open
echo    • Wait for "Models trained successfully" message
echo    • Then open http://localhost:3000 in your browser
echo.

REM Open browser after a delay
echo 🌐 Opening browser in 5 seconds...
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo 🚀 Happy Farming! 🌱
echo.
echo Press any key to exit this window (servers will keep running)...
pause >nul
