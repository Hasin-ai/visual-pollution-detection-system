#!/bin/bash

# Visual Pollution Detection API Setup Script

echo "🔧 Setting up Visual Pollution Detection API..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if model file exists
if [ ! -f "best.pt" ]; then
    echo "⚠️  Warning: Model file 'best.pt' not found!"
    echo "Please place your trained YOLOv11 model file in the current directory."
else
    echo "✅ Model file found: best.pt"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the server:"
echo "  source .venv/bin/activate"
echo "  python run_server.py"
echo ""
echo "Or simply:"
echo "  ./start.sh"