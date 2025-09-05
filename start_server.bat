@echo off
REM AI Recruitment Assistant - Windows Startup Script
REM ==================================================

echo.
echo ========================================
echo  AI Recruitment Assistant Server
echo ========================================
echo.

REM Change to the project directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

REM Check if Python is available
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python first.
    echo See SETUP.md for installation instructions.
    pause
    exit /b 1
)

echo Python found: 
py --version

REM Check for trained model
if not exist "models\demo-fine-tuned" (
    if not exist "models\fine-tuned" (
        echo.
        echo WARNING: No trained model found!
        echo You can either:
        echo 1. Train a demo model quickly: py train_demo.py
        echo 2. Train the full model: py src\training\train_recruitment_model.py
        echo 3. Continue anyway to use the base model
        echo.
        set /p choice="Continue anyway? (y/N): "
        if /i not "%choice%"=="y" (
            echo Exiting...
            pause
            exit /b 1
        )
    )
)

REM Get local IP address for remote access
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do set LOCAL_IP=%%a
set LOCAL_IP=%LOCAL_IP: =%

echo.
echo Starting AI Recruitment Assistant API Server...
echo.
echo Server will be available at:
echo  - Local:  http://localhost:8000
echo  - Remote: http://%LOCAL_IP%:8000
echo  - Docs:   http://localhost:8000/docs
echo.
echo For MacBook access, use: http://%LOCAL_IP%:8000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the server with remote access enabled
py src\deployment\api_server.py --host 0.0.0.0 --port 8000

pause
