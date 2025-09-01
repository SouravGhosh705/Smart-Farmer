@echo off
REM ================================================================================
REM Smart Farmer - Main Menu
REM ================================================================================
REM Complete management interface for Smart Farmer AI Backend
REM ================================================================================

title Smart Farmer - Main Menu

:main_menu
cls
echo.
echo ===============================================================================
echo                    🌾 SMART FARMER - MAIN MENU 🌾
echo ===============================================================================
echo                     AI-Powered Agricultural Backend System
echo                                Version 3.0.0
echo ===============================================================================
echo.

color 0F

echo Choose an option:
echo.
echo 🚀 QUICK START:
echo    1. Start Smart Farmer Backend
echo    2. Test All Services
echo.
echo 🔧 SETUP ^& INSTALLATION:
echo    3. Install Dependencies
echo    4. Setup Ollama (Advanced AI)
echo    5. Check System Status
echo.
echo 📊 UTILITIES:
echo    6. Open API Documentation
echo    7. View Server Logs
echo    8. Stop All Services
echo.
echo 📖 HELP ^& INFO:
echo    9. Quick Start Guide
echo    0. About Smart Farmer
echo.
echo    Q. Quit
echo.
echo ===============================================================================

set /p choice="Enter your choice: "

REM Process user choice
if /i "%choice%"=="1" goto start_backend
if /i "%choice%"=="2" goto test_services
if /i "%choice%"=="3" goto install_deps
if /i "%choice%"=="4" goto setup_ollama
if /i "%choice%"=="5" goto check_status
if /i "%choice%"=="6" goto open_docs
if /i "%choice%"=="7" goto view_logs
if /i "%choice%"=="8" goto stop_services
if /i "%choice%"=="9" goto quick_guide
if /i "%choice%"=="0" goto about
if /i "%choice%"=="q" goto quit

echo Invalid choice. Please try again.
timeout /t 2 /nobreak >nul
goto main_menu

:start_backend
cls
echo ===============================================================================
echo                        🚀 STARTING SMART FARMER BACKEND
echo ===============================================================================
echo.
call smartfarmer.bat
goto main_menu

:test_services
cls
echo ===============================================================================
echo                         🧪 TESTING ALL SERVICES
echo ===============================================================================
echo.
call test_smartfarmer.bat
echo.
echo Press any key to return to main menu...
pause >nul
goto main_menu

:install_deps
cls
echo ===============================================================================
echo                       📦 INSTALLING DEPENDENCIES
echo ===============================================================================
echo.
call install_smartfarmer.bat
echo.
echo Press any key to return to main menu...
pause >nul
goto main_menu

:setup_ollama
cls
echo ===============================================================================
echo                         🦙 SETTING UP OLLAMA
echo ===============================================================================
echo.
call setup_ollama.bat
echo.
echo Press any key to return to main menu...
pause >nul
goto main_menu

:check_status
cls
echo ===============================================================================
echo                        📊 SYSTEM STATUS CHECK
echo ===============================================================================
echo.

echo 🔍 Checking Python environment...
python --version 2>nul || echo ❌ Python not found

echo.
echo 🔍 Checking core dependencies...
python -c "
import sys
modules = ['fastapi', 'uvicorn', 'pandas', 'numpy', 'sklearn', 'requests', 'aiohttp']
missing = []
for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        print(f'❌ {module}')
        missing.append(module)

if missing:
    print(f'\\n⚠️ Missing modules: {missing}')
    print('Run option 3 to install dependencies')
else:
    print('\\n✅ All core dependencies are available')
" 2>nul

echo.
echo 🔍 Checking Smart Farmer backend...
curl http://localhost:8000/ >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend is running on http://localhost:8000
) else (
    echo ❌ Backend is not running
)

echo.
echo 🔍 Checking Ollama service...
curl http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Ollama is running on http://localhost:11434
) else (
    echo ❌ Ollama is not running (optional service)
)

echo.
echo 🔍 Checking directories...
if exist "static" (echo ✅ static directory) else (echo ❌ static directory missing)
if exist "fastapi_app.py" (echo ✅ Main application file) else (echo ❌ Main application missing)

echo.
echo Press any key to return to main menu...
pause >nul
goto main_menu

:open_docs
cls
echo ===============================================================================
echo                       📚 OPENING API DOCUMENTATION
echo ===============================================================================
echo.

echo 🔍 Checking if backend is running...
curl http://localhost:8000/ >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Backend is not running
    echo Please start the backend first (option 1)
    echo.
    echo Press any key to return to main menu...
    pause >nul
    goto main_menu
)

echo ✅ Backend is running
echo.
echo 🌐 Opening API documentation in your default browser...
start http://localhost:8000/docs

echo.
echo 📖 Available documentation:
echo    • Interactive API Docs: http://localhost:8000/docs
echo    • Alternative Docs: http://localhost:8000/redoc
echo    • Backend Status: http://localhost:8000/health
echo.
echo Press any key to return to main menu...
pause >nul
goto main_menu

:view_logs
cls
echo ===============================================================================
echo                            📋 SERVER LOGS
echo ===============================================================================
echo.
echo Logs will be shown when the backend is running.
echo.
echo To view real-time logs:
echo 1. Start the backend (option 1)
echo 2. Logs will appear in the server window
echo.
echo For file-based logging, check the 'logs' directory (if configured)
echo.
echo Press any key to return to main menu...
pause >nul
goto main_menu

:stop_services
cls
echo ===============================================================================
echo                         🛑 STOPPING ALL SERVICES
echo ===============================================================================
echo.

echo 🔍 Looking for Smart Farmer processes...

REM Kill processes on port 8000 (FastAPI)
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8000"') do (
    echo 🔧 Stopping FastAPI backend (PID: %%a)
    taskkill /PID %%a /F >nul 2>&1
)

REM Kill Ollama processes
tasklist | find "ollama" >nul 2>&1
if %errorlevel% equ 0 (
    echo 🔧 Stopping Ollama service...
    taskkill /IM ollama.exe /F >nul 2>&1
)

echo.
echo ✅ All Smart Farmer services stopped
echo.
echo Press any key to return to main menu...
pause >nul
goto main_menu

:quick_guide
cls
echo ===============================================================================
echo                        📖 SMART FARMER QUICK GUIDE
echo ===============================================================================
echo.

echo 🌾 SMART FARMER - AI AGRICULTURAL BACKEND
echo.
echo 🚀 GETTING STARTED:
echo    1. Run 'Install Dependencies' (option 3) - First time only
echo    2. Run 'Start Smart Farmer Backend' (option 1)
echo    3. Open http://localhost:8000/docs in your browser
echo.
echo 🔧 MAIN FEATURES:
echo    ✅ Smart AI Chatbot - Ask farming questions
echo    ✅ AI Crop Doctor - Upload plant images for disease detection
echo    ✅ Weather Integration - Real-time weather data
echo    ✅ Market Prices - Live commodity price analysis
echo    ✅ Price Alerts - Set alerts for target prices
echo    ✅ Translation - Multi-language support
echo    ✅ Analytics - Market trends and forecasting
echo.
echo 🤖 ENHANCED AI (Optional):
echo    • Install Ollama for advanced chatbot features
echo    • Use setup_ollama.bat or option 4
echo.
echo 🌐 API ENDPOINTS:
echo    • GET  /health - Check system health
echo    • POST /chat - Chat with AI advisor
echo    • GET  /market/prices/{crop} - Get market prices
echo    • POST /weather/current - Get weather data
echo    • POST /disease-detection - Analyze plant diseases
echo    • POST /translate - Translate text
echo.
echo 📱 TESTING:
echo    Use option 2 'Test All Services' to verify everything works
echo.
echo ===============================================================================
echo.
echo Press any key to return to main menu...
pause >nul
goto main_menu

:about
cls
echo ===============================================================================
echo                         🌾 ABOUT SMART FARMER
echo ===============================================================================
echo.

echo 📋 SMART FARMER - AI AGRICULTURAL BACKEND
echo    Version: 3.0.0
echo    Platform: FastAPI + Python
echo    Architecture: Microservices with AI Integration
echo.
echo 👨‍💻 DEVELOPED BY: Agricultural AI Team
echo 📅 RELEASE DATE: September 2025
echo 🔄 LAST UPDATED: %date%
echo.
echo 🎯 PURPOSE:
echo    Provide farmers with AI-powered agricultural insights,
echo    real-time market data, weather information, and
echo    intelligent crop management recommendations.
echo.
echo 🌟 KEY TECHNOLOGIES:
echo    • FastAPI - Modern Python web framework
echo    • TensorFlow/PyTorch - AI and machine learning
echo    • OpenWeatherMap - Real-time weather data
echo    • Government APIs - Market price integration
echo    • Ollama - Local AI language models
echo    • OpenCV - Computer vision for disease detection
echo.
echo 📊 STATISTICS:
echo    • 25+ API endpoints
echo    • 8+ integrated services
echo    • 100%% free and open source
echo    • Multi-language support
echo    • Real-time data integration
echo.
echo 🏆 FEATURES:
echo    ✅ Smart agricultural chatbot
echo    ✅ AI-powered disease detection
echo    ✅ Market price analysis and forecasting
echo    ✅ Weather-based crop recommendations
echo    ✅ Multi-language translation
echo    ✅ Price alert system
echo    ✅ Dataset management
echo    ✅ Comprehensive analytics
echo.
echo 🌍 IMPACT:
echo    Helping farmers make data-driven decisions for better
echo    crop yields, reduced losses, and improved profitability.
echo.
echo ===============================================================================
echo.
echo Press any key to return to main menu...
pause >nul
goto main_menu

:quit
cls
echo.
echo ===============================================================================
echo                           👋 THANK YOU!
echo ===============================================================================
echo.
echo Thank you for using Smart Farmer - AI Agricultural Backend!
echo.
echo 🌾 We hope this tool helps improve agricultural productivity
echo 📈 and supports farmers with intelligent insights.
echo.
echo 🌐 For more information and updates:
echo    • Check the API documentation: http://localhost:8000/docs
echo    • Review the project files in this directory
echo.
echo 💚 Happy Farming!
echo.
echo ===============================================================================
echo.
pause
exit /b 0
