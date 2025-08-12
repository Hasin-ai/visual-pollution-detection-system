"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class PollutionLevel(str, Enum):
    """Enumeration of pollution levels"""
    CLEAN = "Clean"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    SEVERE = "Severe"

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate") 
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")

class Detection(BaseModel):
    """Individual detection result"""
    class_name: str = Field(..., alias="class", description="Detected class name")
    class_id: int = Field(..., description="Class ID number")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    description: str = Field(..., description="Human-readable description of the class")

class ImageInfo(BaseModel):
    """Information about the processed image"""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    format: str = Field(..., description="Image format")

class ModelInfo(BaseModel):
    """Information about the model used for inference"""
    confidence_threshold: float = Field(..., description="Confidence threshold used")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    classes: List[str] = Field(..., description="List of all possible classes")

class PredictionResponse(BaseModel):
    """Complete prediction response"""
    success: bool = Field(..., description="Whether the prediction was successful")
    num_detections: int = Field(..., description="Total number of detections")
    avg_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    pollution_level: PollutionLevel = Field(..., description="Overall pollution level assessment")
    class_counts: Dict[str, int] = Field(..., description="Count of each detected class")
    detections: List[Detection] = Field(..., description="List of individual detections")
    image_info: ImageInfo = Field(..., description="Information about the processed image")
    model_info: ModelInfo = Field(..., description="Information about the model")

class PredictionError(BaseModel):
    """Error response for failed predictions"""
    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[str] = Field(None, description="Additional error details")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")

class ModelInfoResponse(BaseModel):
    """Model information response"""
    is_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Device being used")
    model_path: str = Field(..., description="Path to the model file")
    classes: List[str] = Field(..., description="List of detection classes")
    class_descriptions: Dict[str, str] = Field(..., description="Descriptions of each class")
    cuda_available: bool = Field(..., description="Whether CUDA is available")