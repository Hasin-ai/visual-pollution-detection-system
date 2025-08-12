"""
API routes for visual pollution prediction
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from PIL import Image
import logging
from typing import Optional
import io

from app.models.prediction_models import PredictionResponse, PredictionError, ModelInfoResponse
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()

def get_model_service(request: Request) -> ModelService:
    """Dependency to get the model service from app state"""
    if not hasattr(request.app.state, 'model_service'):
        raise HTTPException(
            status_code=503, 
            detail="Model service not available"
        )
    
    model_service = request.app.state.model_service
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return model_service

@router.post("/predict", response_model=PredictionResponse, responses={
    400: {"model": PredictionError, "description": "Bad request"},
    413: {"model": PredictionError, "description": "File too large"},
    500: {"model": PredictionError, "description": "Internal server error"},
    503: {"model": PredictionError, "description": "Service unavailable"}
})
async def predict_visual_pollution(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Query(0.25, ge=0.1, le=0.9, description="Confidence threshold for detections"),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Predict visual pollution in an uploaded image
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **confidence**: Confidence threshold for detections (0.1-0.9)
    
    Returns detailed information about detected visual pollution elements including:
    - Number and types of detections
    - Confidence scores
    - Bounding box coordinates
    - Overall pollution level assessment
    """
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Expected image, got: {file.content_type}"
            )
        
        # Check file size (limit to 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
            )
        
        # Validate that we have file content
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        logger.info(f"Processing image: {file.filename}, size: {file_size} bytes, type: {file.content_type}")
        
        # Load and validate image
        try:
            image = Image.open(io.BytesIO(contents))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check image dimensions
            min_size = 32
            max_size_pixels = 4096
            if image.width < min_size or image.height < min_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too small. Minimum size: {min_size}x{min_size}"
                )
            
            if image.width > max_size_pixels or image.height > max_size_pixels:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too large. Maximum size: {max_size_pixels}x{max_size_pixels}"
                )
                
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Run prediction
        try:
            result = await model_service.predict(image, confidence_threshold=confidence)
            
            logger.info(f"Prediction completed: {result['num_detections']} detections, "
                       f"pollution level: {result['pollution_level']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(
    model_service: ModelService = Depends(get_model_service)
):
    """
    Get information about the loaded model
    
    Returns details about the model including:
    - Loading status
    - Device being used
    - Available classes
    - Class descriptions
    """
    try:
        info = model_service.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )
    



@router.get("/classes")
async def get_classes(model_service: ModelService = Depends(get_model_service)):
    """
    Get list of detection classes and their descriptions
    """
    try:
        return {
            "classes": model_service.CLASS_NAMES,
            "descriptions": model_service.CLASS_DESCRIPTIONS,
            "total_classes": len(model_service.CLASS_NAMES)
        }
    except Exception as e:
        logger.error(f"Error getting classes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get classes: {str(e)}"
        )

@router.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint for the prediction service
    """
    try:
        if hasattr(request.app.state, 'model_service'):
            model_service = request.app.state.model_service
            if model_service.is_loaded:
                return {
                    "status": "healthy",
                    "service": "prediction",
                    "model_loaded": True,
                    "message": "Prediction service is ready"
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "service": "prediction", 
                        "model_loaded": False,
                        "message": "Model not loaded"
                    }
                )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "service": "prediction",
                    "model_loaded": False,
                    "message": "Model service not available"
                }
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "service": "prediction",
                "message": f"Health check failed: {str(e)}"
            }
        )

@router.get("/classes")
async def get_classes(model_service: ModelService = Depends(get_model_service)):
    """
    Get list of detection classes and their descriptions
    """
    try:
        return {
            "classes": model_service.CLASS_NAMES,
            "descriptions": model_service.CLASS_DESCRIPTIONS,
            "total_classes": len(model_service.CLASS_NAMES)
        }
    except Exception as e:
        logger.error(f"Error getting classes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get classes: {str(e)}"
        )