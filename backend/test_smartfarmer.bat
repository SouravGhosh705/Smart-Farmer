@echo off
REM ================================================================================
REM Smart Farmer - Quick Test Script
REM ================================================================================
REM Tests all major endpoints of the Smart Farmer Backend
REM ================================================================================

title Smart Farmer - Quick Test

echo.
echo ===============================================================================
echo                    🌾 SMART FARMER - QUICK TEST SCRIPT 🌾
echo ===============================================================================
echo                        Testing AI Agricultural Backend
echo ===============================================================================
echo.

color 0C

set backend_url=http://localhost:8000

REM Check if backend is running
echo 🔍 Checking if Smart Farmer backend is running...
curl %backend_url%/ >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Smart Farmer backend is not running
    echo.
    echo 🚀 Please start the backend first:
    echo    Run: smartfarmer.bat
    echo.
    pause
    exit /b 1
)

echo ✅ Backend is running on %backend_url%

REM Test endpoints
echo.
echo ===============================================================================
echo 🧪 TESTING ENDPOINTS
echo ===============================================================================
echo.

echo 1️⃣ Testing Health Check...
curl -s %backend_url%/health | findstr "healthy" >nul
if %errorlevel% equ 0 (
    echo ✅ Health check passed
) else (
    echo ❌ Health check failed
)

echo.
echo 2️⃣ Testing System Status...
curl -s %backend_url%/system/status | findstr "operational" >nul
if %errorlevel% equ 0 (
    echo ✅ System status check passed
) else (
    echo ❌ System status check failed
)

echo.
echo 3️⃣ Testing Market Prices...
curl -s "%backend_url%/market/prices/rice" | findstr "commodity" >nul
if %errorlevel% equ 0 (
    echo ✅ Market prices working
) else (
    echo ❌ Market prices failed
)

echo.
echo 4️⃣ Testing Weather Service...
powershell -Command "try { $response = Invoke-RestMethod -Uri '%backend_url%/weather/current' -Method POST -Body '{\"city\": \"Delhi\"}' -ContentType 'application/json'; if ($response.location) { Write-Host '✅ Weather service working' } else { Write-Host '❌ Weather service failed' } } catch { Write-Host '❌ Weather service error' }"

echo.
echo 5️⃣ Testing Chatbot...
powershell -Command "try { $response = Invoke-RestMethod -Uri '%backend_url%/chat' -Method POST -Body '{\"message\": \"Hello\"}' -ContentType 'application/json'; if ($response.response) { Write-Host '✅ Chatbot working' } else { Write-Host '❌ Chatbot failed' } } catch { Write-Host '❌ Chatbot error' }"

echo.
echo 6️⃣ Testing Translation Service...
powershell -Command "try { $response = Invoke-RestMethod -Uri '%backend_url%/translate' -Method POST -Body '{\"text\": \"Hello\", \"target_language\": \"hi\"}' -ContentType 'application/json'; if ($response.translated_text) { Write-Host '✅ Translation service working' } else { Write-Host '❌ Translation service failed' } } catch { Write-Host '❌ Translation service error' }"

echo.
echo 7️⃣ Testing Market Analytics...
curl -s "%backend_url%/market/analytics/wheat" | findstr "analytics" >nul
if %errorlevel% equ 0 (
    echo ✅ Market analytics working
) else (
    echo ❌ Market analytics failed
)

echo.
echo 8️⃣ Testing Dataset Management...
curl -s "%backend_url%/datasets/available" | findstr "plantvillage" >nul
if %errorlevel% equ 0 (
    echo ✅ Dataset management working
) else (
    echo ❌ Dataset management failed
)

REM Test summary
echo.
echo ===============================================================================
echo                          📊 TEST SUMMARY
echo ===============================================================================
echo.
echo 🌐 Backend URL: %backend_url%
echo 📚 API Documentation: %backend_url%/docs
echo 📖 Alternative Docs: %backend_url%/redoc
echo.
echo 🔧 If any tests failed, check:
echo    1. Backend is running (smartfarmer.bat)
echo    2. All dependencies are installed (install_smartfarmer.bat)
echo    3. Internet connection for external APIs
echo.
echo 💡 For advanced features:
echo    • Install Ollama for enhanced chatbot
echo    • Get API keys for Data.gov.in for real market data
echo    • Set up proper SSL certificates for production
echo.
echo ===============================================================================
echo.

REM Open API documentation
echo 🌐 Would you like to open the API documentation? (y/n)
set /p open_docs="Enter choice: "
if /i "%open_docs%"=="y" (
    echo 🚀 Opening API documentation...
    start http://localhost:8000/docs
)

echo.
echo Testing completed!
pause
