@echo off
REM ================================================================================
REM Smart Farmer - Ollama Setup Script (Optional)
REM ================================================================================
REM This script helps setup Ollama for advanced chatbot features
REM ================================================================================

title Smart Farmer - Ollama Setup

echo.
echo ===============================================================================
echo                   🦙 SMART FARMER - OLLAMA SETUP 🦙
echo ===============================================================================
echo                      Advanced AI Chatbot Configuration
echo ===============================================================================
echo.

color 0E

echo 🤖 This script will help you set up Ollama for advanced AI chatbot features
echo.
echo ⚠️ Note: Ollama is optional. Smart Farmer works without it using fallback responses.
echo.

REM Check if Ollama is already installed
echo 🔍 Checking if Ollama is installed...
ollama --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Ollama is already installed
    ollama --version
    goto check_service
) else (
    echo ❌ Ollama is not installed
)

REM Guide user to install Ollama
echo.
echo ===============================================================================
echo 📥 OLLAMA INSTALLATION GUIDE
echo ===============================================================================
echo.
echo Ollama needs to be installed manually. Please follow these steps:
echo.
echo 1️⃣ Visit: https://ollama.ai/download
echo 2️⃣ Download Ollama for Windows
echo 3️⃣ Run the installer
echo 4️⃣ Restart your terminal/command prompt
echo 5️⃣ Run this script again
echo.
echo 🌐 Opening Ollama download page...
start https://ollama.ai/download
echo.
echo After installation, run this script again to continue setup.
pause
exit /b 0

:check_service
echo.
echo 🔍 Checking if Ollama service is running...

REM Try to connect to Ollama
curl http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Ollama service is not running
    echo 🚀 Starting Ollama service...
    
    REM Start Ollama in background
    start /B ollama serve
    
    echo ⏳ Waiting for service to start...
    timeout /t 5 /nobreak >nul
    
    REM Check again
    curl http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Failed to start Ollama service
        echo Please try starting it manually: ollama serve
        pause
        exit /b 1
    )
)

echo ✅ Ollama service is running

REM Check installed models
echo.
echo 🔍 Checking installed models...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ No models found
    goto install_models
)

echo ✅ Checking for agricultural models...
ollama list | findstr "llama3.2\|phi3\|llama3.1" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ No recommended agricultural models found
    goto install_models
) else (
    echo ✅ Agricultural models are available
    goto test_ollama
)

:install_models
echo.
echo ===============================================================================
echo 📥 INSTALLING AGRICULTURAL AI MODELS
echo ===============================================================================
echo.
echo Choose a model to install:
echo.
echo 1️⃣ llama3.2 (Recommended - Balanced performance)
echo 2️⃣ phi3 (Lightweight - Faster responses)
echo 3️⃣ llama3.1 (High quality - Best responses)
echo 4️⃣ Install all recommended models
echo 5️⃣ Skip model installation
echo.
set /p model_choice="Enter your choice (1-5): "

if "%model_choice%"=="1" (
    echo 📥 Installing llama3.2...
    ollama pull llama3.2
) else if "%model_choice%"=="2" (
    echo 📥 Installing phi3...
    ollama pull phi3
) else if "%model_choice%"=="3" (
    echo 📥 Installing llama3.1...
    ollama pull llama3.1
) else if "%model_choice%"=="4" (
    echo 📥 Installing all recommended models...
    echo This may take several minutes...
    ollama pull llama3.2
    ollama pull phi3
    ollama pull llama3.1
) else (
    echo ⏭️ Skipping model installation
    goto test_ollama
)

if %errorlevel% neq 0 (
    echo ❌ Model installation failed
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo ✅ Model installation completed

:test_ollama
echo.
echo ===============================================================================
echo 🧪 TESTING OLLAMA INTEGRATION
echo ===============================================================================
echo.

echo 🔍 Testing Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Cannot connect to Ollama service
    pause
    exit /b 1
)

echo ✅ Ollama connection successful

echo.
echo 🤖 Testing Smart Farmer Ollama integration...
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://localhost:8000/ollama/status' -Method GET; if ($response.ollama_available) { Write-Host '✅ Smart Farmer Ollama integration working' } else { Write-Host '❌ Integration failed' } } catch { Write-Host '❌ Backend not running - start smartfarmer.bat first' }"

REM Display setup summary
echo.
echo ===============================================================================
echo                        🎉 OLLAMA SETUP COMPLETE! 🎉
echo ===============================================================================
echo.
echo ✅ Ollama is now configured for Smart Farmer!
echo.
echo 🚀 Enhanced Features Now Available:
echo    • Intelligent agricultural advisory
echo    • Context-aware farming recommendations
echo    • Advanced crop management suggestions
echo    • Personalized farming guidance
echo    • Multi-turn conversations with memory
echo.
echo 🌐 To use advanced chatbot:
echo    1. Ensure Smart Farmer backend is running (smartfarmer.bat)
echo    2. Use the /chat endpoint
echo    3. Try complex agricultural questions
echo.
echo 📊 Available Models:
ollama list

echo.
echo 💡 Pro Tips:
echo    • Use llama3.2 for best overall performance
echo    • Use phi3 for faster responses on slower machines
echo    • Use llama3.1 for highest quality agricultural advice
echo.
echo ===============================================================================
echo.
pause
