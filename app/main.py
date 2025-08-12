"""
Visual Pollution Detection API
FastAPI application for detecting visual pollution in images using YOLOv11
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from app.routers import prediction
from app.services.model_service import ModelService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model service instance
model_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events"""
    global model_service
    
    # Startup: Load the model
    logger.info("Loading visual pollution detection model...")
    try:
        model_path = Path("best.pt")
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model_service = ModelService(str(model_path))
        await model_service.load_model()
        logger.info("Model loaded successfully!")
        
        # Store model service in app state
        app.state.model_service = model_service
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down application...")
    if model_service:
        await model_service.cleanup()

# Create FastAPI application
app = FastAPI(
    title="Visual Pollution Detection API",
    description="API for detecting visual pollution in images using YOLOv11",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Include routers
app.include_router(prediction.router, prefix="/api", tags=["prediction"])

# Mount static files
static_path = Path("app/static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse, tags=["root"])
async def root():
    """Serve the main page"""
    try:
        static_file = Path("app/static/index.html")
        if static_file.exists():
            with open(static_file, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(content="""
            <html>
                <head><title>Visual Pollution Detection API</title></head>
                <body>
                    <h1>Visual Pollution Detection API</h1>
                    <p>Welcome to the Visual Pollution Detection API!</p>
                    <ul>
                        <li><a href="/docs">API Documentation (Swagger)</a></li>
                        <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                        <li><a href="/api/health">Health Check</a></li>
                    </ul>
                    <p>To test the API, upload an image to <code>/api/predict</code></p>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving root page: {e}")
        return HTMLResponse(content="<h1>Service Unavailable</h1>", status_code=503)

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    try:
        if hasattr(app.state, 'model_service') and app.state.model_service.is_loaded:
            return {
                "status": "healthy",
                "message": "Visual Pollution Detection API is running",
                "model_loaded": True,
                "version": "1.0.0"
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Model not loaded",
                "model_loaded": False,
                "version": "1.0.0"
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )