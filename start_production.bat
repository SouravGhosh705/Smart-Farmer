@echo off
title Smart Farmer - Production Mode
color 0A
cls

echo.
echo ========================================================
echo        🌾 SMART FARMER - PRODUCTION MODE 🌾
echo ========================================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo 🚀 Starting Smart Farmer Application in Production Mode...
echo.

REM Kill existing processes
echo 🔄 Stopping any existing servers...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Smart Farmer Backend*" >nul 2>&1
taskkill /f /im node.exe /fi "WINDOWTITLE eq Smart Farmer Frontend*" >nul 2>&1
echo ✅ Cleanup completed
echo.

REM Set production environment variables
set FLASK_ENV=production
set FLASK_DEBUG=false
set NODE_ENV=production
set TRAIN_ON_STARTUP=false
set EAGER_LOAD_DATASETS=false
set EAGER_LOAD_MODELS=false

echo 🔧 Environment variables set for production mode
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

REM Build frontend for production if not exists
echo 🏗️ Checking production build...
cd frontend
if not exist "build" (
    echo 📦 Building frontend for production (this may take a few minutes)...
    call npm run build
    if errorlevel 1 (
        echo ❌ Frontend build failed!
        pause
        exit /b 1
    )
    echo ✅ Production build completed
) else (
    echo ✅ Production build already exists
)
cd ..

REM Start backend with production settings
echo 🔧 Starting Backend Server in Production Mode...
cd backend
if not exist app.py (
    echo ❌ app.py not found!
    pause
    exit /b 1
)

REM Install waitress for production server
pip install waitress --quiet >nul 2>&1

start "Smart Farmer Backend" /min cmd /k "title Smart Farmer Backend && python -c \"from waitress import serve; from app import app; print('Backend running on http://localhost:8000'); serve(app, host='0.0.0.0', port=8000, threads=4)\""
echo ✅ Backend starting with Waitress production server
cd ..

REM Start frontend with production build server
echo 🌐 Starting Frontend Production Server...
cd frontend

REM Install serve globally if not exists
call npm list -g serve >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing production server...
    call npm install -g serve --silent
)

start "Smart Farmer Frontend" /min cmd /k "title Smart Farmer Frontend && serve -s build -l 3000"
echo ✅ Frontend production server starting
cd ..

echo.
echo ⏱️  Please wait 15-30 seconds for both servers to fully start...
echo.

REM Wait for servers to start
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
echo 🎉 SMART FARMER IS RUNNING IN PRODUCTION MODE!
echo ========================================================
echo.
echo 📱 Your application is available at:
echo    🌐 Frontend: http://localhost:3000
echo    🔗 Backend API: http://localhost:8000
echo    📚 API Docs: http://localhost:8000/docs
echo.
echo 💡 Production Features:
echo    ✅ Optimized performance
echo    ✅ Models trained on-demand
echo    ✅ Production server (Waitress)
echo    ✅ Static build serving
echo.
echo ⚠️  IMPORTANT:
echo    • Both servers are running in production mode
echo    • ML models will be trained when first needed
echo    • Keep both server windows open
echo    • Much faster startup and better performance
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
