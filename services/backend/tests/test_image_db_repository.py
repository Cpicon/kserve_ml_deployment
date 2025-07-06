"""Unit tests for ImageDBRepository."""

import hashlib
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aiq_circular_detection.models.db import Base
from aiq_circular_detection.repositories.image import ImageDBRepository

from aiq_circular_detection.models import CircularObject


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()
    engine.dispose()


@pytest.fixture
def repository(db_session):
    """Create an ImageDBRepository instance for testing."""
    return ImageDBRepository(db_session)


class TestImageDBRepository:
    """Test cases for ImageDBRepository."""
    
    def test_generate_id(self, repository):
        """Test SHA-256 ID generation."""
        content = b"test image content"
        expected_id = hashlib.sha256(content).hexdigest()
        
        generated_id = repository._generate_id(content)
        
        assert generated_id == expected_id
        assert len(generated_id) == 64  # SHA-256 produces 64 hex chars
    
    def test_add_image_success(self, repository):
        """Test successful image addition."""
        content = b"test image content"
        path = "data/test-image.jpg"
        
        # Add image
        image_id = repository.add_image(content, path)
        
        # Verify image properties
        assert image_id == hashlib.sha256(content).hexdigest()
        assert repository.exists(image_id)
        assert repository.get_path(image_id) == path
    
    def test_add_image_empty_content(self, repository):
        """Test adding image with empty content."""
        with pytest.raises(ValueError, match="Image content cannot be empty"):
            repository.add_image(b"", "data/test.jpg")
    
    def test_add_image_duplicate(self, repository):
        """Test adding duplicate image."""
        content = b"test image content"
        path1 = "data/test1.jpg"
        path2 = "data/test2.jpg"
        
        # Add first image
        _ = repository.add_image(content, path1)
        
        # Try to add duplicate
        with pytest.raises(ValueError, match="Image with identical content already exists"):
            repository.add_image(content, path2)
    
    def test_get_path_exists(self, repository):
        """Test retrieving path for existing image."""
        content = b"test image content"
        path = "data/test-image.jpg"
        image_id = repository.add_image(content, path)
        
        # Retrieve path
        retrieved_path = repository.get_path(image_id)
        
        assert retrieved_path == path
    
    def test_get_path_not_exists(self, repository):
        """Test retrieving path for non-existent image."""
        fake_id = "a" * 64  # Valid SHA-256 format
        
        with pytest.raises(KeyError, match="Image with ID .* not found"):
            repository.get_path(fake_id)
    
    def test_delete_success(self, repository):
        """Test successful image deletion."""
        content = b"test image content"
        path = "data/test-image.jpg"
        image_id = repository.add_image(content, path)
        
        # Delete image
        repository.delete(image_id)
        
        assert not repository.exists(image_id)
        assert repository.count() == 0
    
    def test_delete_not_exists(self, repository):
        """Test deleting non-existent image."""
        fake_id = "a" * 64
        
        with pytest.raises(KeyError, match="Image with ID .* not found"):
            repository.delete(fake_id)
    
    def test_count(self, repository):
        """Test counting images."""
        assert repository.count() == 0
        # Add images with unique content
        repository.add_image(b"image1", "data/image1.jpg")
        assert repository.count() == 1
        
        repository.add_image(b"image2", "data/image2.jpg")
        assert repository.count() == 2

    
    def test_exists(self, repository):
        """Test checking image existence."""
        content = b"test image content"
        path = "data/test-image.jpg"
        image_id = repository.add_image(content, path)
        
        assert repository.exists(image_id) is True
        assert repository.exists("nonexistent" * 8) is False  # 64 chars
    
    def test_cascade_delete(self, repository, db_session):
        """Test that deleting an image cascades to circular objects."""
        
        # Add image
        content = b"test image content"
        path = "data/test-image.jpg"
        image_id = repository.add_image(content, path)
        
        # Add a circular object to the image
        circular_object = CircularObject(
            image_id=image_id,
            bbox=[10, 20, 30, 40],
            centroid={"x": 20.0, "y": 30.0},
            radius=10.0
        )
        db_session.add(circular_object)
        db_session.commit()
        
        # Verify circular object exists
        assert db_session.query(CircularObject).count() == 1
        
        # Delete image
        repository.delete(image_id)
        
        # Verify circular object was cascade deleted
        assert db_session.query(CircularObject).count() == 0
    
    def test_transaction_rollback(self, repository, db_session):
        """Test that transactions are properly rolled back on error."""
        content = b"test image content"
        path = "data/test-image.jpg"
        
        # Mock a database error during commit
        with patch.object(db_session, 'commit', side_effect=Exception("DB error")):
            with pytest.raises(Exception, match="DB error"):
                repository.add_image(content, path)
        
        # Verify no image was saved
        assert repository.count() == 0
    
    def test_unique_paths(self, repository):
        """Test that different images can have unique paths."""
        # Add multiple images with different content
        image_id1 = repository.add_image(b"content1", "data/image1.jpg")
        image_id2 = repository.add_image(b"content2", "data/image2.jpg")
        image_id3 = repository.add_image(b"content3", "data/image3.jpg")
        
        # Verify all images exist with correct paths
        assert repository.get_path(image_id1) == "data/image1.jpg"
        assert repository.get_path(image_id2) == "data/image2.jpg"
        assert repository.get_path(image_id3) == "data/image3.jpg"
        assert repository.count() == 3 