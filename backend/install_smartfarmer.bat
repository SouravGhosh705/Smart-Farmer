@echo off
REM ================================================================================
REM Smart Farmer - Installation Script
REM ================================================================================
REM This script installs all dependencies for Smart Farmer AI Backend
REM ================================================================================

title Smart Farmer - Installation

echo.
echo ===============================================================================
echo                    🌾 SMART FARMER - INSTALLATION SCRIPT 🌾
echo ===============================================================================
echo                         Installing AI Agricultural Backend
echo ===============================================================================
echo.

color 0B

REM Check Python installation
echo 🔍 Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo.
    echo 📥 Please download and install Python 3.8+ from:
    echo    https://www.python.org/downloads/
    echo.
    echo ⚙️ Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python is available
python --version

REM Check pip
echo.
echo 🔍 Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not available
    echo Installing pip...
    python -m ensurepip --upgrade
)

echo ✅ pip is available
pip --version

REM Upgrade pip
echo.
echo 🔧 Upgrading pip to latest version...
python -m pip install --upgrade pip

REM Create necessary directories
echo.
echo 📁 Creating project directories...
if not exist "static" mkdir static
if not exist "static\models" mkdir static\models
if not exist "static\labelencoder" mkdir static\labelencoder
if not exist "static\datasets" mkdir static\datasets
if not exist "static\uploads" mkdir static\uploads
if not exist "logs" mkdir logs

echo ✅ Directories created

REM Install core dependencies
echo.
echo ===============================================================================
echo 📦 INSTALLING CORE DEPENDENCIES
echo ===============================================================================
echo.

echo Installing FastAPI and web framework...
pip install fastapi[all] uvicorn[standard] python-multipart

echo.
echo Installing data science libraries...
pip install pandas numpy==1.26.4 scikit-learn==1.5.1 joblib

echo.
echo Installing HTTP and async libraries...
pip install requests aiohttp

echo.
echo Installing computer vision libraries...
pip install opencv-python==4.8.1.78 pillow

echo.
echo Installing AI/ML frameworks...
pip install tensorflow-cpu torch torchvision

echo.
echo Installing additional utilities...
pip install python-dateutil pytz typing-extensions

REM Check installation
echo.
echo ===============================================================================
echo 🧪 VERIFYING INSTALLATION
echo ===============================================================================
echo.

echo Testing core imports...
python -c "
try:
    import fastapi
    import uvicorn
    import pandas
    import numpy
    import sklearn
    import joblib
    import requests
    import aiohttp
    import cv2
    import PIL
    import tensorflow
    import torch
    import torchvision
    print('✅ All core dependencies successfully imported!')
    print('🎉 Installation completed successfully!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('❌ Installation failed - some dependencies are missing')
    exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    exit(1)
"

if %errorlevel% neq 0 (
    echo.
    echo ❌ Installation verification failed
    echo Please check the error messages above and try again
    pause
    exit /b 1
)

REM Create a simple requirements.txt for future reference
echo.
echo 📝 Creating requirements.txt...
echo # Smart Farmer - Enhanced Agricultural Backend Dependencies > requirements.txt
echo # Generated on %date% %time% >> requirements.txt
echo fastapi[all]>=0.104.0 >> requirements.txt
echo uvicorn[standard]>=0.24.0 >> requirements.txt
echo pandas>=2.0.0 >> requirements.txt
echo numpy==1.26.4 >> requirements.txt
echo scikit-learn==1.5.1 >> requirements.txt
echo joblib>=1.3.0 >> requirements.txt
echo requests>=2.32.0 >> requirements.txt
echo aiohttp>=3.9.0 >> requirements.txt
echo opencv-python==4.8.1.78 >> requirements.txt
echo pillow>=10.0.0 >> requirements.txt
echo python-multipart>=0.0.6 >> requirements.txt
echo tensorflow-cpu>=2.15.0 >> requirements.txt
echo torch>=2.0.0 >> requirements.txt
echo torchvision>=0.15.0 >> requirements.txt

echo ✅ requirements.txt created

REM Installation summary
echo.
echo ===============================================================================
echo                        🎉 INSTALLATION COMPLETE! 🎉
echo ===============================================================================
echo.
echo ✅ Smart Farmer Backend is ready to run!
echo.
echo 🚀 To start the application:
echo    Run: smartfarmer.bat
echo.
echo 🌐 Once started, access:
echo    • Backend API: http://localhost:8000
echo    • Documentation: http://localhost:8000/docs
echo    • Alternative Docs: http://localhost:8000/redoc
echo.
echo 📊 Available Features:
echo    ✅ Smart AI Chatbot
echo    ✅ AI Crop Disease Detection  
echo    ✅ Real-time Weather Integration
echo    ✅ Enhanced Market Price Analysis
echo    ✅ Multi-language Translation
echo    ✅ Price Alerts and Forecasting
echo    ✅ Dataset Management
echo    ✅ Comprehensive Analytics
echo.
echo 💡 Optional Enhancement:
echo    For advanced chatbot features, install Ollama:
echo    https://ollama.ai/download
echo.
echo ===============================================================================
echo.
pause
