"""FastAPI dependency injection configuration."""

import logging
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from aiq_circular_detection.db import get_db
from aiq_circular_detection.repositories import ImageRepository, InMemoryImageRepository, ImageDBRepository
from config import get_settings

logger = logging.getLogger(__name__)


# Global instance for in-memory repository
_in_memory_repository: InMemoryImageRepository | None = None


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