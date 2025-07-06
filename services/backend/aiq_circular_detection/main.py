import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Initialize database
from aiq_circular_detection.db import init_db
from aiq_circular_detection.dependencies import (
    get_image_repository,
    get_model_client,
    get_object_db_repository,
    get_storage_client,
)
from aiq_circular_detection.model_client import ModelClient
from aiq_circular_detection.repositories import ImageRepository
from aiq_circular_detection.repositories.object_db import CircularObjectDBRepository
from aiq_circular_detection.schemas import (
    CircularObjectResponse,
    DetectionResponse,
    ImageUploadResponse,
    ObjectDetailResponse,
    ObjectListSummaryResponse,
    ObjectSummary,
)
from aiq_circular_detection.storage.base import StorageClient
from config import get_settings

# Get settings and configure logging before anything else
settings = get_settings()
settings.configure_logging()

# Create logger for this module
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Storage type: {settings.storage_type}")
    logger.info(f"Storage root: {settings.storage_root}")
    
    # Ensure storage directory exists
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    init_db()
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app with settings
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health-check endpoint."""
    return {
        "status": "ok",
        "environment": settings.environment,
        "version": settings.app_version,
    }


@app.get("/config")
def get_config() -> dict[str, Any]:
    """Get current configuration (non-sensitive values only)."""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "api_prefix": settings.api_prefix,
        "storage_type": settings.storage_type,
        "mode": settings.mode,
        "model_name": settings.model_name,
        "log_level": settings.log_level,
        "log_json": settings.log_json,
        "max_upload_size": settings.max_upload_size,
    }


def _generate_image_id(content: bytes) -> str:
    """Generate SHA-256 hash ID from image content.
    
    Args:
        content: Raw image bytes
        
    Returns:
        str: Hexadecimal SHA-256 hash of the content
    """
    return hashlib.sha256(content).hexdigest()


@app.post("/images/", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile,
    image_repository: ImageRepository = Depends(get_image_repository),
    storage_client: StorageClient = Depends(get_storage_client),
    model_client: ModelClient = Depends(get_model_client),
    object_repo: CircularObjectDBRepository = Depends(get_object_db_repository)
) -> ImageUploadResponse:
    """Upload an image file.
    
    The image ID is generated based on the file content using SHA-256 hashing.
    This ensures that duplicate images are detected and rejected.
    
    Args:
        file: The uploaded image file.
        image_repository: Repository for storing image metadata.
        storage_client: Client for storing the actual image files.
        
    Returns:
        ImageUploadResponse: Response containing the unique image ID.
        
    Raises:
        HTTPException: If the file is empty, it already exists or cannot be saved.
    """
    # Read the file content
    content = await file.read()
    
    # Check if a file is empty
    if not content:
        raise HTTPException(status_code=400, detail="File cannot be empty")
    
    # Generate ID from content
    image_id = _generate_image_id(content)
    logger.info(f"Processing image: {file.filename}, content length: {len(content)}, ID: {image_id}")
    
    # Check if the image already exists in the repository
    if image_repository.exists(image_id):
        # Image already exists, get its path and return
        try:
            existing_path = image_repository.get_path(image_id)
            logger.info(f"Image already exists: {image_id} -> {existing_path}")
            
            # Return the existing image ID
            detail = f"Image already exists with ID: {image_id}"
            raise HTTPException(status_code=409, detail=detail)
        except KeyError:
            # This shouldn't happen, but handle gracefully
            logger.error(f"Image {image_id} exists but path not found")
            raise HTTPException(status_code=500, detail="Internal repository inconsistency")
    
    try:
        # Image doesn't exist, save it
        logger.info(f"Saving new image: {image_id}")
        
        # Save the image using a storage client
        file_path = storage_client.save_image(content)
        
        # Add to repository
        stored_id = image_repository.add_image(content, file_path)
        
        # Verify the ID matches (it should)
        if stored_id != image_id:
            logger.error(f"ID mismatch: expected {image_id}, got {stored_id}")
            raise HTTPException(status_code=500, detail="Internal ID generation inconsistency")
        
        logger.info(f"Successfully uploaded image: {image_id} -> {file_path}")
        
        # Perform model inference
        detection_response = None
        try:
            # Call model client to detect circles
            logger.debug(f"Starting circle detection for image {image_id}")
            circles = await model_client.detect_circles(str(file_path))
            
            # Save detected circles to database
            saved_circles = []
            for circle in circles:
                try:
                    # Create circular object in database
                    db_object = object_repo.create_object(
                        image_id=image_id,
                        bbox=circle["bbox"],
                        centroid=circle["centroid"],
                        radius=circle["radius"]
                    )
                    
                    # Convert to response schema
                    circle_response = CircularObjectResponse(
                        object_id=db_object.id,
                        bbox=db_object.bbox,
                        centroid=db_object.centroid,
                        radius=db_object.radius
                    )
                    saved_circles.append(circle_response)
                    
                except Exception as e:
                    logger.error(f"Failed to save detected circle: {e}")
                    # Continue with other circles even if one fails
            
            # Create detection response
            detection_response = DetectionResponse(
                detections=saved_circles,
                count=len(saved_circles)
            )
            
            logger.info(f"Successfully detected and saved {len(saved_circles)} circles for image {image_id}")
            
        except Exception as e:
            # Log the error but don't fail the upload
            logger.error(f"Model inference failed for image {image_id}: {e}")
            # Check if it's an HTTP error that should be propagated as 502
            if isinstance(e, httpx.HTTPStatusError):
                if e.response.status_code >= 500:
                    raise HTTPException(status_code=502, detail="Model inference failed")
        
        # Return response with image ID and detection results
        return ImageUploadResponse(
            image_id=image_id,
            detection=detection_response
        )
        
    except ValueError as e:
        # This shouldn't happen since we already checked existence
        logger.error(f"Unexpected duplicate during save: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error during save")
    except Exception as e:
        logger.error(f"Failed to upload image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")


@app.get("/images/{image_id}/objects", response_model=ObjectListSummaryResponse, tags=["objects"])
async def list_image_objects(
    image_id: str,
    image_repo: ImageRepository = Depends(get_image_repository),
    object_repo: CircularObjectDBRepository = Depends(get_object_db_repository)
) -> ObjectListSummaryResponse:
    """List all circular objects detected in a specific image.
    
    Args:
        image_id: SHA-256 hash ID of the image
        image_repo: Image database repository
        object_repo: Circular object database repository
        
    Returns:
        ObjectListSummaryResponse: Response containing image metadata and list of object summaries
        
    Raises:
        HTTPException: If the image is not found
    """
    # Check if the image exists
    if not image_repo.exists(image_id):
        logger.warning(f"Image not found: {image_id}")
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get all objects for this image
    objects = object_repo.list_objects(image_id=image_id)
    
    # Convert to ObjectSummary instances
    summaries = []
    for obj in objects:
        # Convert float bbox to int bbox
        bbox_int = [int(coord) for coord in obj.bbox]
        
        summary = ObjectSummary(
            object_id=obj.id,
            bbox=bbox_int
        )
        summaries.append(summary)
    
    logger.info(f"Listing {len(summaries)} objects for image {image_id}.")
    
    # Return the comprehensive response
    return ObjectListSummaryResponse(
        image_id=image_id,
        count=len(summaries),
        objects=summaries
    )


@app.get("/images/{image_id}/objects/{object_id}", response_model=ObjectDetailResponse, tags=["objects"])
async def get_object_detail(
    image_id: str,
    object_id: str,
    image_repo: ImageRepository = Depends(get_image_repository),
    object_repo: CircularObjectDBRepository = Depends(get_object_db_repository)
) -> ObjectDetailResponse:
    """Get detailed information about a specific circular object.
    
    Args:
        image_id: SHA-256 hash ID of the image
        object_id: UUID of the circular object
        image_repo: Image database repository
        object_repo: Circular object database repository
        
    Returns:
        ObjectDetailResponse: Detailed information about the circular object
        
    Raises:
        HTTPException: If the image or object is not found
    """
    # Check if the image exists
    if not image_repo.exists(image_id):
        logger.warning(f"Image not found: {image_id}")
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Parse object_id as UUID
    try:
        from uuid import UUID
        object_uuid = UUID(object_id)
    except ValueError:
        logger.warning(f"Invalid object ID format: {object_id}")
        raise HTTPException(status_code=404, detail="Object not found")
    
    # Get the object
    obj = object_repo.get_object(object_uuid)
    
    if not obj:
        logger.warning(f"Object not found: {object_id}")
        raise HTTPException(status_code=404, detail="Object not found")
    
    # Verify the object belongs to the specified image
    if obj.image_id != image_id:
        logger.warning(f"Object {object_id} does not belong to image {image_id}")
        raise HTTPException(status_code=404, detail="Object not found")
    
    # Convert float bbox to int bbox
    bbox_int = [int(coord) for coord in obj.bbox]
    
    # Extract centroid as tuple
    centroid_tuple = (obj.centroid["x"], obj.centroid["y"])
    
    logger.info(f"Returning object {object_id} for image {image_id}.")
    
    # Return the detailed response
    return ObjectDetailResponse(
        object_id=obj.id,
        bbox=bbox_int,
        centroid=centroid_tuple,
        radius=obj.radius
    ) 