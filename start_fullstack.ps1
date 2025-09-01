# Smart Farmer Full Stack Launcher
# ================================================================================
# Starts both FastAPI backend and React frontend in separate windows
# ================================================================================

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "                    🌾 SMART FARMER - FULL STACK LAUNCHER 🌾" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "                                Version 3.0.0" -ForegroundColor Yellow
Write-Host "                      AI Agricultural Platform - Full Stack" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check Node.js
Write-Host "🔍 Checking Node.js installation..." -ForegroundColor Blue
try {
    $nodeVersion = node --version 2>$null
    Write-Host "✅ Node.js is available: $nodeVersion" -ForegroundColor Green
    $nodeAvailable = $true
} catch {
    Write-Host "❌ Node.js is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org" -ForegroundColor Yellow
    $nodeAvailable = $false
}

# Check Python
Write-Host "🔍 Checking Python installation..." -ForegroundColor Blue
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ Python is available: $pythonVersion" -ForegroundColor Green
    $pythonAvailable = $true
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    $pythonAvailable = $false
}

if (-not $pythonAvailable) {
    Write-Host "Cannot start without Python. Exiting..." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING SMART FARMER FULL STACK APPLICATION" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# Start Backend
Write-Host "🔧 Starting Backend Server..." -ForegroundColor Blue
$backendPath = Join-Path $PSScriptRoot "backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; .\smartfarmer.bat"

# Wait for backend to initialize
Start-Sleep -Seconds 3

if ($nodeAvailable) {
    # Start Frontend
    Write-Host "🎨 Starting Frontend Application..." -ForegroundColor Blue
    $frontendPath = Join-Path $PSScriptRoot "frontend"
    
    # Check if node_modules exists
    $nodeModulesPath = Join-Path $frontendPath "node_modules"
    if (-not (Test-Path $nodeModulesPath)) {
        Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
        Push-Location $frontendPath
        try {
            npm install
            if ($LASTEXITCODE -ne 0) {
                Write-Host "🔧 Trying with --legacy-peer-deps..." -ForegroundColor Yellow
                npm install --legacy-peer-deps
            }
        } catch {
            Write-Host "Error installing dependencies: $_" -ForegroundColor Red
        } finally {
            Pop-Location
        }
    }
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; npm start"
    
    Write-Host ""
    Write-Host "===============================================================================" -ForegroundColor Cyan
    Write-Host "                           🌟 SERVICES STARTED 🌟" -ForegroundColor Green
    Write-Host "===============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📱 Frontend: http://localhost:3000" -ForegroundColor Magenta
    Write-Host "🖥️ Backend: http://localhost:8000" -ForegroundColor Magenta
    Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "🔄 Both services are running in separate windows" -ForegroundColor Green
    Write-Host "⏹️ Close the respective windows to stop each service" -ForegroundColor Yellow
    
} else {
    Write-Host ""
    Write-Host "===============================================================================" -ForegroundColor Cyan
    Write-Host "                          ⚠️ BACKEND ONLY MODE ⚠️" -ForegroundColor Yellow
    Write-Host "===============================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🖥️ Backend is running at: http://localhost:8000" -ForegroundColor Magenta
    Write-Host "📚 API Documentation: http://localhost:8000/docs" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "To install Node.js for frontend:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://nodejs.org" -ForegroundColor White
    Write-Host "2. Download and install LTS version" -ForegroundColor White
    Write-Host "3. Restart this script" -ForegroundColor White
}

Write-Host ""
Write-Host "🌟 Smart Farmer Full Stack Application launcher completed!" -ForegroundColor Green
Write-Host "💡 Tip: Keep this window open to monitor the launch status" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to close this launcher window"
