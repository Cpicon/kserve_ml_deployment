"""Unit tests for storage module."""

from pathlib import Path
from typing import Protocol
from unittest.mock import Mock

import pytest

from aiq_circular_detection.storage import LocalStorageClient, StorageClient
from aiq_circular_detection.storage.base import StorageError


class TestStorageClient:
    """Test StorageClient interface."""
    
    def test_interface_protocol(self):
        """Test that StorageClient is a proper Protocol."""
        assert issubclass(StorageClient, Protocol)
    
    def test_interface_methods_defined(self):
        """Test that interface has required methods."""
        assert hasattr(StorageClient, 'save_image')
        assert hasattr(StorageClient, 'read_image')
    
    def test_runtime_checkable(self):
        """Test that StorageClient can be used with isinstance at runtime."""
        # Create a mock that implements the interface
        mock_client = Mock(spec=StorageClient)
        mock_client.save_image = Mock(return_value="path/to/image")
        mock_client.read_image = Mock(return_value=b"image data")
        
        # Should be able to check if it implements the protocol
        assert isinstance(mock_client, StorageClient)
    
    def test_non_compliant_class(self):
        """Test that non-compliant classes don't match the protocol."""
        class BadClient:
            def save_image(self, content: bytes) -> str:
                return "path"
            # Missing read_image method
        
        client = BadClient()
        assert not isinstance(client, StorageClient)


class TestLocalStorageClient:
    """Test LocalStorageClient implementation."""
    
    @pytest.fixture
    def storage_client(self, tmp_path):
        """Create a LocalStorageClient with temporary directory."""
        return LocalStorageClient(storage_root=str(tmp_path / "images"))
    
    def test_init_creates_directory(self, tmp_path):
        """Test that initialization creates storage directory."""
        storage_root = tmp_path / "test_images"
        assert not storage_root.exists()
        
        _ = LocalStorageClient(storage_root=str(storage_root))
        assert storage_root.exists()
        assert storage_root.is_dir()
    
    def test_init_with_existing_directory(self, tmp_path):
        """Test initialization with existing directory."""
        storage_root = tmp_path / "existing_images"
        storage_root.mkdir()
        
        # Should not raise error
        _ = LocalStorageClient(storage_root=str(storage_root))
        assert storage_root.exists()
    
    def test_init_default_directory(self, monkeypatch, tmp_path):
        """Test initialization with default directory from settings."""
        # Mock the settings to return our temp path
        from config import get_settings
        mock_settings = get_settings()
        mock_settings.storage_root = tmp_path / "data" / "images"
        
        def mock_get_settings():
            return mock_settings
        
        monkeypatch.setattr("aiq_circular_detection.storage.local.get_settings", mock_get_settings)
        
        client = LocalStorageClient()
        assert client.storage_root == tmp_path / "data" / "images"
        # Check the directory was created
        assert client.storage_root.exists()
    
    def test_save_image_success(self, storage_client):
        """Test successful image save."""
        content = b"test image content"
        
        path = storage_client.save_image(content)
        
        # Path should be returned
        assert isinstance(path, str)
        assert path.endswith(".jpg")
        
        # File should exist with correct content
        saved_file = Path(path)
        assert saved_file.exists()
        assert saved_file.read_bytes() == content
    
    def test_save_image_unique_paths(self, storage_client):
        """Test that multiple saves generate unique paths."""
        content1 = b"image 1"
        content2 = b"image 2"
        
        path1 = storage_client.save_image(content1)
        path2 = storage_client.save_image(content2)
        
        assert path1 != path2
        assert Path(path1).exists()
        assert Path(path2).exists()
    
    def test_save_image_empty_content(self, storage_client):
        """Test saving empty content raises ValueError."""
        with pytest.raises(ValueError, match="Image content cannot be empty"):
            storage_client.save_image(b"")
    
    def test_save_image_storage_error(self, storage_client, monkeypatch):
        """Test StorageError when save fails."""
        # Mock write_bytes to raise an exception
        def mock_write_bytes(content):
            raise IOError("Simulated write failure")
        
        # We need to patch Path.write_bytes
        from pathlib import Path
        monkeypatch.setattr(Path, 'write_bytes', mock_write_bytes)
        
        with pytest.raises(StorageError, match="Failed to save image"):
            storage_client.save_image(b"test content")
    
    def test_read_image_success(self, storage_client):
        """Test successful image read."""
        content = b"test image content"
        path = storage_client.save_image(content)
        
        read_content = storage_client.read_image(path)
        assert read_content == content
    
    def test_read_image_with_filename_only(self, storage_client):
        """Test reading image with just filename."""
        content = b"test content"
        full_path = storage_client.save_image(content)
        filename = Path(full_path).name
        
        read_content = storage_client.read_image(filename)
        assert read_content == content
    
    def test_read_image_absolute_path(self, storage_client, tmp_path):
        """Test reading image with absolute path."""
        # Create a file outside storage root
        external_file = tmp_path / "external.img"
        content = b"external content"
        external_file.write_bytes(content)
        
        read_content = storage_client.read_image(str(external_file))
        assert read_content == content
    
    def test_read_image_not_found(self, storage_client):
        """Test FileNotFoundError for non-existent image."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            storage_client.read_image("non_existent.img")
    
    def test_read_image_empty_path(self, storage_client):
        """Test reading with empty path raises ValueError."""
        with pytest.raises(ValueError, match="Image path cannot be empty"):
            storage_client.read_image("")
    
    def test_read_image_storage_error(self, storage_client, monkeypatch):
        """Test StorageError when read fails."""
        # First, save a file successfully
        path = storage_client.save_image(b"content")
        
        # Mock read_bytes to raise an exception
        def mock_read_bytes(self):
            raise IOError("Simulated read failure")
        
        # Patch Path.read_bytes
        from pathlib import Path
        monkeypatch.setattr(Path, 'read_bytes', mock_read_bytes)
        
        with pytest.raises(StorageError, match="Failed to read image"):
            storage_client.read_image(path)
    
    def test_implements_storage_client_protocol(self, storage_client):
        """Test that LocalStorageClient implements StorageClient protocol."""
        assert isinstance(storage_client, StorageClient)
    
    def test_roundtrip_storage(self, storage_client):
        """Test complete save and read cycle."""
        original_content = b"test image data \x00\x01\x02"
        
        # Save image
        path = storage_client.save_image(original_content)
        
        # Read it back
        retrieved_content = storage_client.read_image(path)
        
        assert retrieved_content == original_content
    
    @pytest.mark.parametrize("content", [
        b"small image",
        b"x" * 1024 * 1024,  # 1MB
        b"\x00\x01\x02\x03" * 1000,  # Binary data
    ])
    def test_various_content_sizes(self, storage_client, content):
        """Test storage with various content sizes."""
        path = storage_client.save_image(content)
        retrieved = storage_client.read_image(path)
        assert retrieved == content
    
    def test_uses_settings_storage_root(self, monkeypatch, tmp_path):
        """Test that LocalStorageClient uses storage root from settings."""
        # Set environment variable for storage root
        custom_root = tmp_path / "custom_storage"
        monkeypatch.setenv("STORAGE_ROOT", str(custom_root))
        
        # Clear settings cache to pick up new env var
        from config import get_settings
        get_settings.cache_clear()
        
        # Create client without specifying storage root
        client = LocalStorageClient()
        
        # Should use the settings storage root
        assert client.storage_root == custom_root
        assert custom_root.exists()
    
    def test_override_settings_storage_root(self, tmp_path):
        """Test that explicit storage root overrides settings."""
        override_root = tmp_path / "override"
        
        # Create client with explicit storage root
        client = LocalStorageClient(storage_root=override_root)
        
        # Should use the provided root, not settings
        assert client.storage_root == override_root
        assert override_root.exists() 