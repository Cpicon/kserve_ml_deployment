"""Unit tests for ImageRepository implementations."""

import hashlib

import pytest

from aiq_circular_detection.repositories import InMemoryImageRepository


class TestInMemoryImageRepository:
    """Test cases for InMemoryImageRepository."""
    
    @pytest.fixture
    def repository(self):
        """Create a fresh repository instance for each test."""
        return InMemoryImageRepository()
    
    @pytest.fixture
    def sample_images(self):
        """Generate sample image data for testing."""
        return [
            (f"image content {i}".encode(), f"path/to/image_{i}.jpg")
            for i in range(3)
        ]
    
    def test_add_image_and_get_path(self, repository):
        """Test adding an image and retrieving its path."""
        content = b"test image content"
        path = "path/to/test/image.jpg"
        
        # Add the image
        image_id = repository.add_image(content, path)
        
        # Verify ID is a valid SHA-256 hash
        assert len(image_id) == 64
        assert all(c in "0123456789abcdef" for c in image_id)
        
        # Verify ID is deterministic (same content = same ID)
        expected_id = hashlib.sha256(content).hexdigest()
        assert image_id == expected_id
        
        # Retrieve the path
        retrieved_path = repository.get_path(image_id)
        assert retrieved_path == path
    
    def test_add_duplicate_content_raises_error(self, repository):
        """Test that adding duplicate content raises ValueError."""
        content = b"duplicate image content"
        path1 = "path/to/image1.jpg"
        path2 = "path/to/image2.jpg"
        
        # Add the first image
        image_id1 = repository.add_image(content, path1)
        
        # Try to add the same content again
        with pytest.raises(ValueError) as exc_info:
            repository.add_image(content, path2)
        
        assert image_id1 in str(exc_info.value)
        assert "already exists" in str(exc_info.value)
        assert path1 in str(exc_info.value)
    
    def test_add_empty_content_raises_error(self, repository):
        """Test that adding empty content raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            repository.add_image(b"", "path/to/empty.jpg")
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_get_path_nonexistent_raises_error(self, repository):
        """Test that getting path for non-existent ID raises KeyError."""
        # Generate a valid but non-existent SHA-256 hash
        fake_id = "a" * 64
        
        with pytest.raises(KeyError) as exc_info:
            repository.get_path(fake_id)
        
        assert fake_id in str(exc_info.value)
        assert "not found" in str(exc_info.value)
    
    def test_exists(self, repository):
        """Test checking if an image exists."""
        content = b"test image"
        path = "path/to/image.jpg"
        
        # Add the image
        image_id = repository.add_image(content, path)
        
        # Check exists
        assert repository.exists(image_id)
        
        # Check non-existent
        assert not repository.exists("b" * 64)
    
    def test_delete(self, repository):
        """Test deleting an image."""
        content = b"image to delete"
        path = "path/to/image.jpg"
        
        # Add and verify
        image_id = repository.add_image(content, path)
        assert repository.exists(image_id)
        
        # Delete
        repository.delete(image_id)
        
        # Verify it's gone
        assert not repository.exists(image_id)
        
        # Try to get path after deletion
        with pytest.raises(KeyError):
            repository.get_path(image_id)
    
    def test_delete_nonexistent_raises_error(self, repository):
        """Test that deleting non-existent ID raises KeyError."""
        fake_id = "c" * 64
        
        with pytest.raises(KeyError) as exc_info:
            repository.delete(fake_id)
        
        assert fake_id in str(exc_info.value)
        assert "not found" in str(exc_info.value)
    
    def test_count(self, repository, sample_images):
        """Test counting images in the repository."""
        # Initially empty
        assert repository.count() == 0
        
        # Add images
        image_ids = []
        for content, path in sample_images:
            image_id = repository.add_image(content, path)
            image_ids.append(image_id)
        
        # Check count
        assert repository.count() == len(sample_images)
        
        # Delete one
        repository.delete(image_ids[0])
        assert repository.count() == len(sample_images) - 1
    
    def test_clear(self, repository, sample_images):
        """Test clearing all images from the repository."""
        # Add images
        image_ids = []
        for content, path in sample_images:
            image_id = repository.add_image(content, path)
            image_ids.append(image_id)
        
        assert repository.count() == len(sample_images)
        
        # Clear
        repository.clear()
        
        # Verify empty
        assert repository.count() == 0
        
        # Verify all images are gone
        for image_id in image_ids:
            assert not repository.exists(image_id)
    
    def test_deterministic_id_generation(self, repository):
        """Test that the same content always generates the same ID."""
        content = b"deterministic content"
        path1 = "path1.jpg"
        path2 = "path2.jpg"
        
        # Add image
        id1 = repository.add_image(content, path1)
        
        # Clear and add same content again
        repository.clear()
        id2 = repository.add_image(content, path2)
        
        # IDs should be the same
        assert id1 == id2
    
    def test_different_content_different_ids(self, repository):
        """Test that different content generates different IDs."""
        contents = [
            b"content 1",
            b"content 2",
            b"content 3",
            b"content 1 ",  # Note the extra space
            b"Content 1",   # Different case
        ]
        
        ids = set()
        for i, content in enumerate(contents):
            image_id = repository.add_image(content, f"path_{i}.jpg")
            ids.add(image_id)
        
        # All IDs should be unique
        assert len(ids) == len(contents)
    
    def test_multiple_operations(self, repository):
        """Test a sequence of mixed operations."""
        images = [(f"content_{i}".encode(), f"path_{i}.jpg") for i in range(5)]
        image_ids = []
        
        # Add all images
        for content, path in images:
            image_id = repository.add_image(content, path)
            image_ids.append(image_id)
        
        assert repository.count() == 5
        
        # Delete some
        repository.delete(image_ids[1])
        repository.delete(image_ids[3])
        
        assert repository.count() == 3
        
        # Verify remaining
        assert repository.exists(image_ids[0])
        assert not repository.exists(image_ids[1])
        assert repository.exists(image_ids[2])
        assert not repository.exists(image_ids[3])
        assert repository.exists(image_ids[4])
        
        # Get paths for remaining
        assert repository.get_path(image_ids[0]) == images[0][1]
        assert repository.get_path(image_ids[2]) == images[2][1]
        assert repository.get_path(image_ids[4]) == images[4][1] 