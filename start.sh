#!/bin/bash

# Quick start script for Visual Pollution Detection API

echo "🚀 Starting Visual Pollution Detection API..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment and run server
source .venv/bin/activate && python run_server.py