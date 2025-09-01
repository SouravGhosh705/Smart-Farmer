@echo off
title AI-Assisted Farming - Status Check
color 0B

echo ========================================================
echo 📊 AI-ASSISTED FARMING - DEPLOYMENT STATUS CHECK
echo ========================================================
echo.

REM Check if backend is running
echo 🔧 Checking Backend Status...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo    ❌ Backend is NOT running
    set backend_status=DOWN
) else (
    echo    ✅ Backend is running on http://localhost:8000
    set backend_status=UP
)

REM Check if frontend is running
echo 🌐 Checking Frontend Status...
curl -s http://localhost:3000 >nul 2>&1
if errorlevel 1 (
    echo    ❌ Frontend is NOT running
    set frontend_status=DOWN
) else (
    echo    ✅ Frontend is running on http://localhost:3000
    set frontend_status=UP
)

echo.
echo ========================================================
echo 📋 SUMMARY
echo ========================================================
echo    Backend:  %backend_status%
echo    Frontend: %frontend_status%
echo.

if "%backend_status%"=="UP" if "%frontend_status%"=="UP" (
    echo 🎉 Application is FULLY OPERATIONAL!
    echo.
    echo 🌐 Access your application:
    echo    Frontend: http://localhost:3000
    echo    Backend:  http://localhost:8000
    echo.
    echo Would you like to open the application? (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        start http://localhost:3000
    )
) else (
    echo ⚠️  Application is NOT fully running
    echo.
    echo 🔧 To start the application, run: deploy_app.bat
)

echo.
pause
