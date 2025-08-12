#!/usr/bin/env python3
"""
Startup script for Visual Pollution Detection API
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    """Run the FastAPI server"""
    
    # Check if model file exists
    model_path = Path("best.pt")
    if not model_path.exists():
        print("❌ Error: Model file 'best.pt' not found!")
        print("Please ensure the trained model file is in the current directory.")
        sys.exit(1)
    
    print("🚀 Starting Visual Pollution Detection API...")
    print(f"📁 Model file: {model_path.absolute()}")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()