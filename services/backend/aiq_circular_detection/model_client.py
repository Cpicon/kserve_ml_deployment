"""Model client for circular object detection inference."""
import base64
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Protocol

import httpx

from config import Settings

logger = logging.getLogger(__name__)


class CircleDetection(Protocol):
    """Protocol for circle detection results."""
    bbox: List[float]
    centroid: Dict[str, float]
    radius: float


class ModelClient(ABC):
    """Abstract base class for model inference clients."""
    
    @abstractmethod
    async def detect_circles(self, image_path: str) -> List[Dict[str, any]]:
        """Detect circular objects in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected circles, each containing:
                - bbox: [x_min, y_min, x_max, y_max]
                - centroid: {"x": float, "y": float}
                - radius: float value in pixels
        """
        pass


class RealModelClient(ModelClient):
    """Model client that calls the actual inference server."""
    
    def __init__(self, settings: Settings):
        """Initialize the real model client.
        
        Args:
            settings: Application settings containing model server configuration
        """
        self.settings = settings
        if not settings.model_server_url:
            raise ValueError("MODEL_SERVER_URL must be set for real mode")
        
        self.base_url = settings.model_server_url.rstrip("/")
        self.model_name = settings.model_name
        self.timeout = settings.model_service_timeout
        
    async def detect_circles(self, image_path: str) -> List[Dict[str, any]]:
        """Detect circles by calling the model inference server.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected circles from the model
            
        Raises:
            httpx.HTTPError: If the model server request fails
        """
        # Log start of inference
        logger.debug(f"Starting model inference for image: {image_path}")
        start_time = time.time()
        
        try:
            # Read image file
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image_bytes = image_file.read_bytes()
            
            # Prepare the request payload
            # The payload should contain base64 encoded image and a label
            payload = {
                "image": base64.b64encode(image_bytes).decode("utf-8"),
                "label": "circular_objects"
            }
            
            # Construct the inference URL
            url = f"{self.base_url}/v1/models/{self.model_name}:predict"
            
            # Make the request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                
                # Check for HTTP errors
                if response.status_code != 200:
                    logger.error(
                        f"Model server returned error: {response.status_code} - {response.text}"
                    )
                    response.raise_for_status()
                
                # Parse the response
                result = response.json()
                
                # Extract circles from the response
                # Assuming the response format is a list of circle objects
                circles = result.get("predictions", [])
                
                # Log success
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Model inference completed in {elapsed_time:.2f}s, "
                    f"detected {len(circles)} circles"
                )
                
                return circles
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during model inference: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during model inference: {e}")
            raise


class DummyModelClient(ModelClient):
    """Mock model client that returns fixed fake data."""
    
    def __init__(self, settings: Settings):
        """Initialize the fake model client.
        
        Args:
            settings: Application settings (not used but kept for consistency)
        """
        self.settings = settings
        
    async def detect_circles(self, image_path: str) -> List[Dict[str, any]]:
        """Return fake circle detection results.
        
        Args:
            image_path: Path to the image file (used for logging only)
            
        Returns:
            Fixed list with one fake circle
        """
        # Log start of fake inference
        logger.debug(f"Starting dummy model inference for image: {image_path}")
        start_time = time.time()
        
        # Return fixed dummy data
        dummy_circles = [
            {
                "bbox": [10, 10, 50, 50],
                "centroid": {"x": 30, "y": 30},
                "radius": 20
            }
        ]
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(
            f"Dummy model inference completed in {elapsed_time:.2f}s, "
            f"returned {len(dummy_circles)} circles"
        )
        
        return dummy_circles


def create_model_client(settings: Settings) -> ModelClient:
    """Factory function to create the appropriate model client based on settings.
    
    Args:
        settings: Application settings
        
    Returns:
        ModelClient instance (either RealModelClient or DummyModelClient)
    """
    if settings.mode == "dummy":
        logger.info("Creating DummyModelClient")
        return DummyModelClient(settings)
    else:
        logger.info("Creating RealModelClient")
        return RealModelClient(settings) 