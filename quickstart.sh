#!/bin/bash

# 🌙 Emotion-Aware Digital Comfort Pet - Quick Start Script
# For macOS and Linux systems
# This script automates the entire setup process

echo "🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙"
echo "EMOTION-AWARE DIGITAL COMFORT PET"
echo "Quick Start Setup (macOS/Linux)"
echo "🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙🌙"
echo ""

# Check if Python is installed
echo "📋 Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    echo "   Visit: https://www.python.org/downloads/"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version found"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "⚠️  pip3 not found. Trying pip..."
    if ! command -v pip &> /dev/null; then
        echo "❌ pip is not installed. Please install pip."
        exit 1
    fi
    PIP_CMD="pip"
else
    PIP_CMD="pip3"
fi

echo "✓ pip found"
echo ""

# Create virtual environment
echo "🔧 Setting up virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Using existing one."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
$PIP_CMD install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "📥 Installing dependencies (this may take a minute)..."
$PIP_CMD install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo "✓ Dependencies installed"
echo ""

# Check if model exists
echo "🤖 Checking for trained model..."
if [ -f "models/emotion_model.pkl" ]; then
    echo "✓ Model found!"
    read -p "Do you want to retrain the model? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Training new model..."
        python3 train_model.py
    fi
else
    echo "📊 Model not found. Training now..."
    python3 train_model.py
    if [ $? -ne 0 ]; then
        echo "❌ Model training failed"
        exit 1
    fi
fi
echo ""

# Launch app
echo "🚀 Launching Comfort Pet app..."
echo ""
echo "════════════════════════════════════════════════════════"
echo "✨ Your Comfort Pet is starting!"
echo "📱 Open your browser: http://localhost:8501"
echo "⌨️  Press Ctrl+C to stop the server"
echo "════════════════════════════════════════════════════════"
echo ""

streamlit run app.py