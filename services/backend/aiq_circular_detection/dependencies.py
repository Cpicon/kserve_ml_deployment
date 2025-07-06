"""FastAPI dependency injection configuration."""

import logging

from fastapi import Depends
from sqlalchemy.orm import Session

from aiq_circular_detection.db import get_db
from aiq_circular_detection.repositories import (
    ImageDBRepository,
    ImageRepository,
    InMemoryImageRepository,
)
from aiq_circular_detection.repositories.object_db import CircularObjectDBRepository
from aiq_circular_detection.storage.base import StorageClient
from aiq_circular_detection.storage.local import LocalStorageClient
from config import get_settings

logger = logging.getLogger(__name__)


# Global instance for in-memory repository
_in_memory_repository: InMemoryImageRepository | None = None

# Global instance for storage client
_storage_client: StorageClient | None = None


def get_image_repository(db: Session = Depends(get_db)) -> ImageRepository:
    """Get the appropriate image repository instance based on configuration.
    
    This function returns the correct repository implementation based on
    the METADATA_STORAGE environment variable:
    - "memory": Uses InMemoryImageRepository (data lost on restart)
    - "database": Uses ImageDBRepository (data persisted in database)
    
    Note: This is separate from STORAGE_TYPE which controls where actual
    image files are stored (local, S3, Azure).
    
    Args:
        db: Database session (only used for database metadata storage)
        
    Returns:
        ImageRepository: The configured image repository instance
    """
    settings = get_settings()
    
    if settings.metadata_storage == "database":
        # Create a new database repository instance with the session
        logger.debug("Using database repository for image metadata")
        return ImageDBRepository(db)
    else:
        # Use singleton in-memory repository
        global _in_memory_repository
        if _in_memory_repository is None:
            _in_memory_repository = InMemoryImageRepository()
            logger.info(f"Created in-memory repository for image metadata (metadata_storage={settings.metadata_storage})")
        return _in_memory_repository


def get_object_db_repository(db: Session = Depends(get_db)) -> CircularObjectDBRepository:
    """Get a CircularObjectDBRepository instance.
    
    Args:
        db: Database session from get_db dependency
        
    Returns:
        CircularObjectDBRepository: Repository for circular object operations
    """
    return CircularObjectDBRepository(db)


def get_storage_client() -> StorageClient:
    """Get the appropriate storage client based on configuration.
    
    This function returns the correct storage implementation based on
    the STORAGE_TYPE environment variable:
    - "local": Uses LocalStorageClient (stores files on local filesystem)
    - Future: Could support "s3", "azure", etc.
    
    Returns:
        StorageClient: The configured storage client instance
    """
    global _storage_client
    
    if _storage_client is None:
        settings = get_settings()
        
        # For now, we only support local storage
        # In the future, this could be extended to support other storage types
        if settings.storage_type == "local":
            _storage_client = LocalStorageClient()
            logger.info(f"Created local storage client with root: {settings.storage_root}")
        else:
            # Default to local storage for unknown types
            logger.warning(f"Unknown storage type: {settings.storage_type}, defaulting to local")
            _storage_client = LocalStorageClient()
    
    return _storage_client 