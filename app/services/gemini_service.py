"""
Service for interacting with Google Gemini API
"""

import httpx
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for Google Gemini API interactions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            logger.warning("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
    
    async def generate_content(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate content using Gemini API"""
        if not self.api_key:
            return self._get_fallback_content()
        
        try:
            url = f"{self.base_url}/models/gemini-pro:generateContent"
            
            headers = {
                "Content-Type": "application/json",
            }
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{url}?key={self.api_key}",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "candidates" in data and len(data["candidates"]) > 0:
                        candidate = data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            return candidate["content"]["parts"][0]["text"]
                    
                    logger.warning("Unexpected Gemini API response format")
                    return self._get_fallback_content()
                    
                else:
                    logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                    return self._get_fallback_content()
                    
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return self._get_fallback_content()
    
    def _get_fallback_content(self) -> str:
        """Fallback content when Gemini API is not available"""
        return """
# Visual Pollution: Understanding the Impact

Visual pollution refers to the presence of elements in the environment that are aesthetically displeasing and negatively impact the visual quality of an area. This form of pollution significantly affects urban landscapes and the quality of life for residents.

## What is Visual Pollution?

Visual pollution encompasses various unsightly elements that clutter and degrade the visual environment, including:

- **Billboards and excessive advertising**: Large advertising signs that dominate skylines and obstruct natural views
- **Street litter and waste**: Garbage, debris, and improperly disposed materials on streets and public spaces
- **Construction materials**: Building supplies, equipment, and debris left in public areas
- **Brick piles and construction waste**: Accumulated construction materials blocking walkways and views
- **Overhead wires and cables**: Tangled electrical and communication lines that create visual chaos
- **Communication towers**: Poorly placed or excessive telecommunication infrastructure

## Impact on Communities

Visual pollution has far-reaching effects on urban communities:

### Environmental Impact
- Degrades the aesthetic quality of neighborhoods
- Reduces property values in affected areas
- Contributes to overall environmental degradation
- Disrupts natural landscapes and urban planning

### Social and Psychological Effects
- Increases stress and reduces quality of life for residents
- Creates a sense of neglect and abandonment in communities
- Impacts mental health and overall well-being
- Reduces civic pride and community engagement

### Economic Consequences
- Decreases tourism and business investment
- Lowers property values and rental rates
- Increases cleanup and maintenance costs for municipalities
- Reduces the attractiveness of areas for development

## The Role of AI in Detection

Modern AI systems, like this visual pollution detection model, play a crucial role in:

- **Automated monitoring**: Systematic identification of pollution sources across large areas
- **Data collection**: Gathering comprehensive data for urban planning decisions
- **Prioritization**: Helping authorities focus cleanup efforts where they're needed most
- **Tracking progress**: Monitoring improvements and changes over time
- **Resource allocation**: Optimizing the deployment of cleanup and maintenance resources

## Our Detection Model

This YOLOv11-based model is specifically trained to identify six major types of visual pollution commonly found in urban environments, particularly focusing on street-level pollution in cities like Dhaka. The model helps automate the process of identifying and cataloging visual pollution, supporting efforts to create cleaner, more aesthetically pleasing urban environments.

By leveraging computer vision and machine learning, we can work towards building more sustainable and visually appealing cities for everyone.
        """.strip()
    
    async def get_visual_pollution_overview(self) -> str:
        """Get a comprehensive overview of visual pollution"""
        prompt = """
        Please provide a comprehensive overview of visual pollution that covers:
        
        1. Definition and explanation of what visual pollution is
        2. Common types and examples of visual pollution in urban environments
        3. The impact of visual pollution on communities, environment, and quality of life
        4. How visual pollution affects urban planning and development
        5. The role of technology and AI in detecting and managing visual pollution
        6. Why automated detection systems are important for urban management
        
        Focus particularly on street-level visual pollution including billboards, litter, construction materials, overhead wires, and other urban clutter. Write this in a clear, informative style suitable for a technical documentation or README file.
        """
        
        return await self.generate_content(prompt, max_tokens=1500)