# Visual Pollution Detection API

A FastAPI-based web service for detecting visual pollution in images using YOLOv11.

## Features

- 🔍 **AI-Powered Detection**: Uses YOLOv11 to detect 6 types of visual pollution
- 🌐 **REST API**: Easy-to-use HTTP endpoints for image analysis
- 🎨 **Web Interface**: Built-in web UI for testing and visualization
- 📊 **Detailed Results**: Comprehensive analysis with confidence scores and bounding boxes
- ⚡ **Fast Processing**: Optimized for quick inference
- 📚 **Auto Documentation**: Interactive API docs with Swagger UI

## Detected Pollution Types

1. **Billboard** - Large advertising signs and banners
2. **Street Litter** - Garbage and waste on streets
3. **Construction Material** - Building materials on roads
4. **Brick** - Brick piles and construction debris
5. **Wire** - Overhead electrical and communication wires
6. **Tower** - Communication towers and poles

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
./setup.sh
```

### 2. Start the Server

```bash
# Quick start
./start.sh

# Or manually
source .venv/bin/activate
python run_server.py
```

### 3. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### POST `/api/predict`
Upload an image for visual pollution detection.

**Parameters:**
- `file`: Image file (JPEG, PNG, etc.)
- `confidence`: Confidence threshold (0.1-0.9, default: 0.25)

**Response:**
```json
{
  "success": true,
  "num_detections": 3,
  "avg_confidence": 0.85,
  "pollution_level": "Moderate",
  "class_counts": {
    "Billboard": 1,
    "Street_Litter": 2,
    "Wire": 0
  },
  "detections": [
    {
      "class": "Billboard",
      "confidence": 0.92,
      "bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 200},
      "description": "Large advertising signs and banners"
    }
  ]
}
```

### GET `/api/model/info`
Get information about the loaded model.

### GET `/api/health`
Health check endpoint.

### GET `/api/classes`
Get list of detection classes and descriptions.

## Project Structure

```
visual_pollution_api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   └── prediction_models.py  # Pydantic models
│   ├── services/
│   │   ├── model_service.py      # ML model service
│   │   └── gemini_service.py     # Gemini API service
│   ├── routers/
│   │   └── prediction.py         # API routes
│   └── static/
│       ├── index.html            # Web interface
│       ├── style.css             # Styling
│       └── script.js             # Frontend logic
├── best.pt                  # YOLOv11 model file
├── requirements.txt         # Python dependencies
├── run_server.py           # Server startup script
├── setup.sh                # Environment setup
├── start.sh                # Quick start script
└── README.md               # This file
```

## Requirements

- Python 3.8+
- PyTorch
- FastAPI
- Ultralytics YOLOv11
- OpenCV
- PIL/Pillow

## Development

### Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python -m app.main
```

### Testing the API

```bash
# Test with curl
curl -X POST "http://localhost:8000/api/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg" \
     -F "confidence=0.25"
```

## Model Information

The API uses a custom-trained YOLOv11 model specifically designed for visual pollution detection. The model file (`best.pt`) should be placed in the root directory.

## Environment Variables

Optional environment variables:

- `GEMINI_API_KEY`: Google Gemini API key for enhanced descriptions
- `LOG_LEVEL`: Logging level (default: INFO)

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please check the API documentation at `/docs` when the server is running.