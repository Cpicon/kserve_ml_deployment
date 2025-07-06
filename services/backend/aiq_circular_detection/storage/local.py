"""Local filesystem implementation of StorageClient."""
import logging
import uuid
from pathlib import Path
from typing import Optional

from config import get_settings

from .base import StorageClient, StorageError

logger = logging.getLogger(__name__)


class LocalStorageClient(StorageClient):
    """Local filesystem storage implementation.
    
    Stores images as files in a configurable directory on the local filesystem.
    Each image is saved with a unique filename to prevent collisions.
    """
    
    def __init__(self, storage_root: Optional[str | Path] = None):
        """Initialize local storage client.
        
        Args:
            storage_root: Root directory for storing images.
                         If not provided, uses the configured storage root from settings.
        """
        settings = get_settings()
        
        # Use provided storage root or fall back to settings
        if storage_root is not None:
            self.storage_root = Path(storage_root)
        else:
            self.storage_root = settings.storage_root
        
        self._ensure_storage_dir()
        logger.info(f"Initialized LocalStorageClient with root: {self.storage_root}")
    
    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        try:
            self.storage_root.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured storage directory exists: {self.storage_root}")
        except Exception as e:
            logger.error(f"Failed to create storage directory: {e}")
            raise StorageError(f"Failed to create storage directory: {e}")
    
    def save_image(self, content: bytes) -> str:
        """Save image content to local filesystem.
        
        Args:
            content: Raw image bytes to store.
            
        Returns:
            str: Relative path to the saved image file.
            
        Raises:
            StorageError: If the image cannot be saved.
        """
        if not content:
            raise ValueError("Image content cannot be empty")
        
        # Generate unique filename
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.jpg"
        file_path = self.storage_root / filename
        
        try:
            file_path.write_bytes(content)
            logger.debug(f"Saved image to: {file_path}")
            
            # Return the path as string
            # Try to make it relative to CWD, otherwise return absolute path
            try:
                relative_path = str(file_path.relative_to(Path.cwd()))
                logger.debug(f"Returning relative path: {relative_path}")
                return relative_path
            except ValueError:
                # Path is outside CWD, return absolute path
                logger.debug(f"Returning absolute path: {file_path}")
                return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise StorageError(f"Failed to save image: {e}")
    
    def read_image(self, path: str) -> bytes:
        """Read image content from local filesystem.
        
        Args:
            path: Path to the image file (can be relative or absolute).
            
        Returns:
            bytes: Raw image content.
            
        Raises:
            StorageError: If the image cannot be read.
            FileNotFoundError: If the image does not exist.
        """
        if not path:
            raise ValueError("Image path cannot be empty")
        
        file_path = Path(path)
        
        # If path is not absolute, treat it as relative to storage root
        if not file_path.is_absolute():
            # Check if it's already a full relative path or just filename
            if file_path.parent == Path(".") or file_path.parent == Path():
                file_path = self.storage_root / file_path.name
        
        if not file_path.exists():
            logger.warning(f"Image not found: {path}")
            raise FileNotFoundError(f"Image not found: {path}")
        
        try:
            content = file_path.read_bytes()
            logger.debug(f"Successfully read image from: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read image from {file_path}: {e}")
            raise StorageError(f"Failed to read image: {e}") 