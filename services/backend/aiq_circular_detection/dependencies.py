"""FastAPI dependency injection configuration."""

import logging
from functools import lru_cache

from aiq_circular_detection.repositories import ImageRepository, InMemoryImageRepository

logger = logging.getLogger(__name__)


# Global instance of the image repository
_image_repository: ImageRepository | None = None


@lru_cache
def get_image_repository() -> ImageRepository:
    """Get the image repository instance.
    
    This function provides a singleton instance of the image repository
    that can be injected into FastAPI endpoints.
    
    Returns:
        ImageRepository: The configured image repository instance.
    """
    global _image_repository
    
    if _image_repository is None:
        # For now, we use the in-memory implementation
        # In the future, this could be configured based on settings
        _image_repository = InMemoryImageRepository()
        logger.info("Created image repository instance")
    
    return _image_repository 