"""Storage client interface for image persistence."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageClient(Protocol):
    """Abstract interface for image storage operations.
    
    This interface defines the minimal contract for storing and retrieving
    images. Implementations can use various backends such as local filesystem,
    S3, Azure Blob Storage, etc.
    
    Following the Interface Segregation Principle (ISP), this interface
    only includes the essential methods needed for image storage operations.
    """
    
    def save_image(self, content: bytes) -> str:
        """Save image content to storage.
        
        Args:
            content: Raw image bytes to store.
            
        Returns:
            str: Unique storage path/identifier for the saved image.
            
        Raises:
            StorageError: If the image cannot be saved.
        """
        ...
    
    def read_image(self, path: str) -> bytes:
        """Read image content from storage.
        
        Args:
            path: Storage path/identifier of the image to retrieve.
            
        Returns:
            bytes: Raw image content.
            
        Raises:
            StorageError: If the image cannot be read.
            FileNotFoundError: If the image does not exist.
        """
        ...


class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass 