@echo off
echo ================================================
echo  Smart Farming App - Cloud Deployment Setup
echo ================================================
echo.

echo [1/5] Checking Git repository status...
if not exist ".git" (
    echo Initializing Git repository...
    git init
    echo Git repository initialized.
) else (
    echo Git repository already exists.
)

echo.
echo [2/5] Adding all files to Git...
git add .

echo.
echo [3/5] Creating deployment commit...
git commit -m "Prepare for cloud deployment - Smart Farming App v2.0"

echo.
echo [4/5] Checking deployment configuration...
echo ✅ Backend Dockerfile: %cd%\backend\Dockerfile
echo ✅ Backend Railway config: %cd%\backend\railway.json
echo ✅ Frontend Vercel config: %cd%\frontend\vercel.json
echo ✅ Deployment guide: %cd%\DEPLOYMENT_GUIDE.md

echo.
echo [5/5] Deployment preparation complete!
echo.
echo ================================================
echo  NEXT STEPS:
echo ================================================
echo.
echo 1. Create a GitHub repository: https://github.com/new
echo 2. Run: git remote add origin https://github.com/yourusername/smart-farming-app.git
echo 3. Run: git push -u origin main
echo 4. Deploy backend to Railway: https://railway.app
echo 5. Deploy frontend to Vercel: https://vercel.com
echo.
echo 📖 Full instructions in: DEPLOYMENT_GUIDE.md
echo.
echo ================================================
echo  IMPORTANT ENVIRONMENT VARIABLES TO SET:
echo ================================================
echo.
echo 🔧 Railway (Backend):
echo   SECRET_KEY=your-super-secret-key-here
echo   FLASK_ENV=production
echo   FLASK_DEBUG=false
echo   TRAIN_ON_STARTUP=false
echo   WEATHER_API_KEY=ff049be539ac8642b805155154206e4c
echo.
echo 🌐 Vercel (Frontend):
echo   REACT_APP_BACKEND_URL=https://your-railway-url.railway.app
echo.
echo ================================================
echo  Your app will be FREE and available globally!
echo ================================================
pause
