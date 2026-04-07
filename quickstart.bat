@echo off
REM 🌙 Emotion-Aware Digital Comfort Pet - Quick Start Script
REM For Windows systems
REM This script automates the entire setup process

echo.
echo 🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙
echo EMOTION-AWARE DIGITAL COMFORT PET
echo Quick Start Setup (Windows)
echo 🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙
echo.

REM Check if Python is installed
echo 📋 Checking prerequisites...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH.
    echo    Please install Python 3.8 or higher from https://www.python.org/downloads/
    echo    Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION% found
echo.

REM Create virtual environment
echo 🔧 Setting up virtual environment...
if exist "venv" (
    echo ⚠️  Virtual environment already exists. Using existing one.
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
)
echo.

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo ✓ pip upgraded
echo.

REM Install requirements
echo 📥 Installing dependencies (this may take a minute)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

REM Check if model exists
echo 🤖 Checking for trained model...
if exist "models\emotion_model.pkl" (
    echo ✓ Model found!
    set /p RETRAIN="Do you want to retrain the model? (y/n): "
    if /i "%RETRAIN%"=="y" (
        echo 🔄 Training new model...
        python train_model.py
    )
) else (
    echo 📊 Model not found. Training now...
    python train_model.py
    if %errorlevel% neq 0 (
        echo ❌ Model training failed
        pause
        exit /b 1
    )
)
echo.

REM Launch app
echo 🚀 Launching Comfort Pet app...
echo.
echo ════════════════════════════════════════════════════════
echo ✨ Your Comfort Pet is starting!
echo 📱 Open your browser: http://localhost:8501
echo ⌨️  Press Ctrl+C to stop the server
echo ════════════════════════════════════════════════════════
echo.

streamlit run app.py

pause