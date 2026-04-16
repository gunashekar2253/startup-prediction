@echo off
title Startup Success Predictor - Full Stack Launcher
color 0A

echo ==========================================
echo  STARTUP SUCCESS PREDICTOR - LAUNCHER
echo ==========================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+.
    pause
    exit /b 1
)

:: Check if models are trained
if not exist "model\startup_success_rf_model.pkl" (
    echo [WARNING] Random Forest model not found. Training now...
    echo.
    python train_advanced_model.py
    if errorlevel 1 (
        echo [ERROR] Model training failed. Please run setup_db.py and etl_pipeline.py first.
        pause
        exit /b 1
    )
)

if not exist "model\startup_success_nn_model.pkl" (
    echo [WARNING] Neural Network model not found. Training now...
    echo.
    python train_nn_model.py
    if errorlevel 1 (
        echo [ERROR] Neural Network training failed.
        pause
        exit /b 1
    )
)

echo [✓] Models found and ready.
echo.

:: Start Flask Backend in a new window
echo [1/2] Starting Flask Backend (http://127.0.0.1:5000)...
start "Flask Backend" cmd /k "python Main.py"
timeout /t 3 /nobreak >nul

:: Start React Frontend
echo [2/2] Starting React Frontend (http://localhost:5173)...
cd frontend
start "React Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ==========================================
echo  Both services are starting...
echo  Backend : http://127.0.0.1:5000
echo  Frontend: http://localhost:5173
echo  Press any key to exit this launcher.
echo ==========================================
pause
