"""Image repository for managing image metadata."""

import hashlib
import logging
from typing import Optional, Protocol

from sqlalchemy.orm import Session

from aiq_circular_detection.models.db import Image

logger = logging.getLogger(__name__)


class ImageRepository(Protocol):
    """Interface for image metadata storage.
    
    This repository manages the mapping between image IDs and their storage paths.
    Implementations can use various backends such as in-memory, database, or cache.
    """
    
    def add_image(self, content: bytes, path: str) -> str:
        """Add a new image entry based on its content.
        
        The ID is generated from the image content to ensure uniqueness.
        
        Args:
            content: Raw image bytes for ID generation.
            path: Storage path where the image is saved.
            
        Returns:
            str: Generated unique ID for the image.
            
        Raises:
            ValueError: If an image with the same content already exists.
        """
        ...
    
    def get_path(self, image_id: str) -> str:
        """Get the storage path for an image.
        
        Args:
            image_id: Unique identifier for the image.
            
        Returns:
            str: Storage path of the image.
            
        Raises:
            KeyError: If image_id does not exist.
        """
        ...
    
    def exists(self, image_id: str) -> bool:
        """Check if an image exists in the repository.
        
        Args:
            image_id: Unique identifier for the image.
            
        Returns:
            bool: True if the image exists, False otherwise.
        """
        ...
    
    def delete(self, image_id: str) -> None:
        """Delete an image entry.
        
        Args:
            image_id: Unique identifier for the image.
            
        Raises:
            KeyError: If image_id does not exist.
        """
        ...
    
    def count(self) -> int:
        """Get the total number of images in the repository.
        
        Returns:
            int: Number of images stored.
        """
        ...


class InMemoryImageRepository(ImageRepository):
    """In-memory implementation of ImageRepository.
    
    This implementation stores image metadata in a simple dictionary.
    Data is not persisted and will be lost when the application restarts.
    Suitable for development, testing, and small-scale deployments.
    
    Uses SHA-256 hashing for content-based ID generation to ensure
    uniqueness and prevent duplicate images.
    """
    
    def __init__(self):
        """Initialize the in-memory repository."""
        self._storage: dict[str, str] = {}
        logger.info("Initialized InMemoryImageRepository")
    
    def _generate_id(self, content: bytes) -> str:
        """Generate a unique ID based on image content.
        
        Uses SHA-256 hashing to create a deterministic ID from the image bytes.
        This ensures that identical images will always have the same ID.
        
        Args:
            content: Raw image bytes.
            
        Returns:
            str: Hexadecimal SHA-256 hash of the content.
        """
        return hashlib.sha256(content).hexdigest()
    
    def add_image(self, content: bytes, path: str) -> str:
        """Add a new image entry based on its content.
        
        Args:
            content: Raw image bytes for ID generation.
            path: Storage path where the image is saved.
            
        Returns:
            str: Generated unique ID for the image.
            
        Raises:
            ValueError: If an image with the same content already exists.
        """
        if not content:
            raise ValueError("Image content cannot be empty")
        
        # Generate ID from content
        image_id = self._generate_id(content)
        
        # Check if image already exists
        if image_id in self._storage:
            existing_path = self._storage[image_id]
            raise ValueError(
                f"Image with identical content already exists. "
                f"ID: {image_id}, existing path: {existing_path}"
            )
        
        # Store the mapping
        self._storage[image_id] = path
        logger.debug(f"Added image: {image_id} -> {path}")
        
        return image_id
    
    def get_path(self, image_id: str) -> str:
        """Get the storage path for an image.
        
        Args:
            image_id: Unique identifier for the image.
            
        Returns:
            str: Storage path of the image.
            
        Raises:
            KeyError: If image_id does not exist.
        """
        if image_id not in self._storage:
            raise KeyError(f"Image with ID {image_id} not found")
        
        path = self._storage[image_id]
        logger.debug(f"Retrieved path for image {image_id}: {path}")
        return path
    
    def exists(self, image_id: str) -> bool:
        """Check if an image exists in the repository.
        
        Args:
            image_id: Unique identifier for the image.
            
        Returns:
            bool: True if the image exists, False otherwise.
        """
        return image_id in self._storage
    
    def delete(self, image_id: str) -> None:
        """Delete an image entry.
        
        Args:
            image_id: Unique identifier for the image.
            
        Raises:
            KeyError: If image_id does not exist.
        """
        if image_id not in self._storage:
            raise KeyError(f"Image with ID {image_id} not found")
        
        del self._storage[image_id]
        logger.debug(f"Deleted image: {image_id}")
    
    def count(self) -> int:
        """Get the total number of images in the repository.
        
        Returns:
            int: Number of images stored.
        """
        return len(self._storage)
    
    def clear(self) -> None:
        """Clear all entries from the repository.
        
        This is mainly useful for testing purposes.
        """
        self._storage.clear()
        logger.debug("Cleared all images from repository")


class ImageDBRepository(ImageRepository):
    """SQLAlchemy-based implementation of ImageRepository.
    
    This implementation stores image metadata in a database using SQLAlchemy.
    Data is persisted and will survive application restarts.
    Suitable for production deployments.
    
    Uses SHA-256 hashing for content-based ID generation to ensure
    uniqueness and prevent duplicate images.
    """
    
    def __init__(self, db: Session):
        """Initialize the repository with a database session.
        
        Args:
            db: SQLAlchemy session for database operations
        """
        self.db = db
        logger.info("Initialized ImageDBRepository")
    
    def _generate_id(self, content: bytes) -> str:
        """Generate a SHA-256 hash ID from image content.
        
        Args:
            content: Raw image bytes
            
        Returns:
            str: Hexadecimal SHA-256 hash of the content
        """
        return hashlib.sha256(content).hexdigest()
    
    def add_image(self, content: bytes, path: str) -> str:
        """Add a new image entry based on its content.
        
        Args:
            content: Raw image bytes for ID generation.
            path: Storage path where the image is saved.
            
        Returns:
            str: Generated unique ID for the image.
            
        Raises:
            ValueError: If an image with the same content already exists.
        """
        if not content:
            raise ValueError("Image content cannot be empty")
        
        # Generate content-based ID
        image_id = self._generate_id(content)
        logger.debug(f"Generated image ID: {image_id}")
        
        # Check if image already exists
        existing = self._get_image(image_id)
        if existing:
            raise ValueError(
                f"Image with identical content already exists. "
                f"ID: {image_id}, existing path: {existing.path}"
            )
        
        try:
            # Create database record
            image = Image(id=image_id, path=path)
            self.db.add(image)
            self.db.commit()
            
            logger.info(f"Created image record: {image_id}")
            return image_id
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create image: {e}")
            raise
    
    def get_path(self, image_id: str) -> str:
        """Get the storage path for an image.
        
        Args:
            image_id: Unique identifier for the image.
            
        Returns:
            str: Storage path of the image.
            
        Raises:
            KeyError: If image_id does not exist.
        """
        image = self._get_image(image_id)
        if not image:
            raise KeyError(f"Image with ID {image_id} not found")
        
        logger.debug(f"Retrieved path for image {image_id}: {image.path}")
        return image.path
    
    def exists(self, image_id: str) -> bool:
        """Check if an image exists in the repository.
        
        Args:
            image_id: Unique identifier for the image.
            
        Returns:
            bool: True if the image exists, False otherwise.
        """
        return self.db.query(Image).filter(Image.id == image_id).count() > 0
    
    def delete(self, image_id: str) -> None:
        """Delete an image entry.
        
        This will cascade delete all associated CircularObject records.
        
        Args:
            image_id: Unique identifier for the image.
            
        Raises:
            KeyError: If image_id does not exist.
        """
        image = self._get_image(image_id)
        if not image:
            raise KeyError(f"Image with ID {image_id} not found")
        
        try:
            self.db.delete(image)
            self.db.commit()
            logger.info(f"Deleted image: {image_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete image {image_id}: {e}")
            raise
    
    def count(self) -> int:
        """Get the total number of images in the repository.
        
        Returns:
            int: Number of images stored.
        """
        return self.db.query(Image).count()
    
    def _get_image(self, image_id: str) -> Optional[Image]:
        """Internal method to retrieve an image by its ID.
        
        Args:
            image_id: The SHA-256 hash ID of the image
            
        Returns:
            Optional[Image]: The Image instance if found, None otherwise
        """
        return self.db.query(Image).filter(Image.id == image_id).first() 