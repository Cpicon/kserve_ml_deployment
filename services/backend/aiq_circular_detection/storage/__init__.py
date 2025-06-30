"""Storage module for image persistence."""

from .base import StorageClient
from .local import LocalStorageClient

__all__ = ["StorageClient", "LocalStorageClient"] 