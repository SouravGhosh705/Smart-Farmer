@echo off
REM ================================================================================
REM Smart Farmer - Enhanced Agricultural Backend Startup Script
REM ================================================================================
REM Version: 3.0.0
REM Description: Complete startup script for Smart Farmer AI Backend
REM ================================================================================

title Smart Farmer - AI Agricultural Backend

echo.
echo ===============================================================================
echo                    🌾 SMART FARMER - AI AGRICULTURAL BACKEND 🌾
echo ===============================================================================
echo                                Version 3.0.0
echo                           Enhanced with AI and Analytics
echo ===============================================================================
echo.

REM Set color for better visibility
color 0A

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

REM Check if we're in the correct directory
if not exist "fastapi_app.py" (
    echo ❌ fastapi_app.py not found
    echo Please run this script from the backend directory
    pause
    exit /b 1
)

echo ✅ Backend files found

REM Create necessary directories
echo.
echo 📁 Creating necessary directories...
if not exist "static" mkdir static
if not exist "static\models" mkdir static\models
if not exist "static\labelencoder" mkdir static\labelencoder
if not exist "static\datasets" mkdir static\datasets
if not exist "static\uploads" mkdir static\uploads

echo ✅ Directories created

REM Check if dependencies are installed
echo.
echo 📦 Checking dependencies...
python -c "import fastapi, uvicorn, pandas, numpy, sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Some dependencies are missing
    echo 🔧 Installing required dependencies...
    
    pip install fastapi[all] uvicorn[standard] pandas numpy scikit-learn joblib requests aiohttp opencv-python==4.8.1.78 pillow python-multipart tensorflow-cpu torch torchvision --quiet
    
    if %errorlevel% neq 0 (
        echo ❌ Failed to install dependencies
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
    
    echo ✅ Dependencies installed successfully
) else (
    echo ✅ All dependencies are available
)

REM Display startup information
echo.
echo ===============================================================================
echo 🚀 STARTING SMART FARMER BACKEND
echo ===============================================================================
echo.
echo 📍 Backend URL: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo 📖 Alternative Docs: http://localhost:8000/redoc
echo.
echo 🌟 Available Services:
echo    ✅ Smart AI Chatbot (with fallback)
echo    ✅ AI Crop Disease Detection
echo    ✅ Real-time Weather Integration
echo    ✅ Enhanced Market Price Analysis
echo    ✅ Multi-language Translation
echo    ✅ Price Alerts and Forecasting
echo    ✅ Dataset Management
echo    ✅ Comprehensive Analytics
echo.
echo ⚠️ Note: For advanced chatbot features, install Ollama separately
echo    Visit: https://ollama.ai/download
echo.
echo ===============================================================================

REM Check if port 8000 is available
echo 🔍 Checking if port 8000 is available...
netstat -an | find ":8000" >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚠️ Port 8000 is already in use
    echo Would you like to:
    echo 1. Kill existing process and continue
    echo 2. Use a different port
    echo 3. Exit
    set /p choice="Enter your choice (1-3): "
    
    if "%choice%"=="1" (
        echo 🔧 Killing process on port 8000...
        for /f "tokens=5" %%a in ('netstat -ano ^| find ":8000"') do taskkill /PID %%a /F >nul 2>&1
        timeout /t 2 /nobreak >nul
    ) else if "%choice%"=="2" (
        set /p custom_port="Enter port number (default 8001): "
        if "%custom_port%"=="" set custom_port=8001
        set port=%custom_port%
        goto start_server
    ) else (
        echo 👋 Goodbye!
        pause
        exit /b 0
    )
)

:start_server
if not defined port set port=8000

echo.
echo 🚀 Starting Smart Farmer Backend on port %port%...
echo.
echo ===============================================================================
echo                           🌾 SERVER STARTING 🌾
echo ===============================================================================
echo.
echo ⏹️ Press Ctrl+C to stop the server
echo 🌐 Open http://localhost:%port%/docs for API documentation
echo.

REM Start the FastAPI server
python start_server.py --port %port%

echo.
echo ===============================================================================
echo                           🌾 SERVER STOPPED 🌾
echo ===============================================================================
echo.
echo Thank you for using Smart Farmer!
echo.
pause
