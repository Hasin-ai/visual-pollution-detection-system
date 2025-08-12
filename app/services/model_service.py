"""
Model service for loading and running inference with the YOLOv11 visual pollution detection model
"""

import torch
import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelService:
    """Service for managing the visual pollution detection model"""
    
    # Visual pollution class names based on your training
    CLASS_NAMES = [
        "Billboard", 
        "Brick", 
        "Construction_Material", 
        "Street_Litter", 
        "Tower",
        "Wire"
    ]
    
    CLASS_DESCRIPTIONS = {
        "Billboard": "Large advertising signs and banners",
        "Street_Litter": "Garbage and waste on streets",
        "Construction_Material": "Building materials on roads",
        "Brick": "Brick piles and construction debris",
        "Wire": "Overhead electrical and communication wires",
        "Tower": "Communication towers and poles"
    }
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"Initializing ModelService with device: {self.device}")
    
    async def load_model(self):
        """Load the YOLOv11 model asynchronously"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model_sync
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        try:
            # Load YOLOv11 model
            model = YOLO(self.model_path)
            
            # Move to appropriate device
            if self.device == "cuda" and torch.cuda.is_available():
                model.to(self.device)
                logger.info("Model moved to GPU")
            
            # Warm up the model with a dummy inference
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = model(dummy_image, verbose=False)
            
            return model
            
        except Exception as e:
            logger.error(f"Error in synchronous model loading: {e}")
            raise e
    
    async def predict(self, image: Image.Image, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """Run inference on an image"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            logger.info(f"Running inference with confidence threshold: {confidence_threshold}")
            
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                image,
                confidence_threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e
    
    def _predict_sync(self, image: Image.Image, confidence_threshold: float) -> Dict[str, Any]:
        """Synchronous prediction"""
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Ensure RGB format
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # Convert RGB to BGR for OpenCV (YOLO expects BGR)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run inference
            results = self.model(img_array, conf=confidence_threshold, verbose=False)
            
            # Parse results
            detections = []
            total_confidence = 0.0
            class_counts = {cls: 0 for cls in self.CLASS_NAMES}
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if cls_id < len(self.CLASS_NAMES):
                            class_name = self.CLASS_NAMES[cls_id]
                            class_counts[class_name] += 1
                            total_confidence += conf
                            
                            x1, y1, x2, y2 = box
                            detections.append({
                                "class": class_name,
                                "class_id": int(cls_id),
                                "confidence": float(conf),
                                "bbox": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2)
                                },
                                "description": self.CLASS_DESCRIPTIONS.get(class_name, "Unknown")
                            })
            
            # Calculate overall metrics
            num_detections = len(detections)
            avg_confidence = total_confidence / num_detections if num_detections > 0 else 0.0
            
            # Determine overall pollution level
            pollution_level = self._calculate_pollution_level(detections, class_counts)
            
            return {
                "success": True,
                "num_detections": num_detections,
                "avg_confidence": float(avg_confidence),
                "pollution_level": pollution_level,
                "class_counts": class_counts,
                "detections": detections,
                "image_info": {
                    "width": image.width,
                    "height": image.height,
                    "format": image.format or "Unknown"
                },
                "model_info": {
                    "confidence_threshold": confidence_threshold,
                    "device": self.device,
                    "classes": self.CLASS_NAMES
                }
            }
            
        except Exception as e:
            logger.error(f"Error in synchronous prediction: {e}")
            raise e
    
    def _calculate_pollution_level(self, detections: List[Dict], class_counts: Dict[str, int]) -> str:
        """Calculate overall pollution level based on detections"""
        if not detections:
            return "Clean"
        
        # Weight different types of pollution
        pollution_weights = {
            "Street_Litter": 3,
            "Construction_Material": 2,
            "Brick": 2,
            "Billboard": 1,
            "Wire": 1,
            "Tower": 1
        }
        
        weighted_score = 0
        for class_name, count in class_counts.items():
            weight = pollution_weights.get(class_name, 1)
            weighted_score += count * weight
        
        # Determine level based on weighted score
        if weighted_score == 0:
            return "Clean"
        elif weighted_score <= 2:
            return "Low"
        elif weighted_score <= 5:
            return "Moderate"
        elif weighted_score <= 10:
            return "High"
        else:
            return "Severe"
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
                logger.info("Thread pool executor shut down")
            
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "is_loaded": self.is_loaded,
            "device": self.device,
            "model_path": self.model_path,
            "classes": self.CLASS_NAMES,
            "class_descriptions": self.CLASS_DESCRIPTIONS,
            "cuda_available": torch.cuda.is_available()
        }