import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from aiq_circular_detection.schemas import ImageUploadResponse
from aiq_circular_detection.storage.local import LocalStorageClient

# Get settings and configure logging before anything else
settings = get_settings()
settings.configure_logging()

# Create logger for this module
logger = logging.getLogger(__name__)

# Initialize storage client
storage_client = LocalStorageClient()

# In-memory mapping from image_id to file path (placeholder for now)
image_id_to_path: dict[str, str] = {}


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
        "log_level": settings.log_level,
        "log_json": settings.log_json,
        "max_upload_size": settings.max_upload_size,
    }


@app.post("/images/", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile) -> ImageUploadResponse:
    """Upload an image file.
    
    Args:
        file: The uploaded image file.
        
    Returns:
        ImageUploadResponse: Response containing the unique image ID.
        
    Raises:
        HTTPException: If the file is empty or cannot be saved.
    """
    # Read the file content
    content = await file.read()
    
    # Check if file is empty
    if not content:
        raise HTTPException(status_code=400, detail="File cannot be empty")
    
    try:
        # Save the image using storage client
        file_path = storage_client.save_image(content)
        
        # Generate unique image ID
        image_id = uuid.uuid4()
        
        # Store mapping in memory (placeholder for now)
        image_id_to_path[str(image_id)] = file_path
        
        logger.info(f"Uploaded image: {image_id} -> {file_path}")
        
        # Return response with image ID
        return ImageUploadResponse(image_id=image_id)
    except Exception as e:
        logger.error(f"Failed to upload image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}") 