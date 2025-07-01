"""End-to-end integration tests for image upload with model inference."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from aiq_circular_detection.db import init_db
from aiq_circular_detection.main import app
from aiq_circular_detection.models.db import CircularObject
from aiq_circular_detection.repositories.object_db import CircularObjectDBRepository
from config import Settings, get_settings


@pytest.fixture
def test_settings():
    """Override settings for testing."""
    test_settings = Settings(
        mode="dummy",  # Use dummy model client
        metadata_storage="memory",  # Use memory for testing to avoid database issues
        database_url="sqlite:///test_db.sqlite3",
        storage_root=Path("test_data/images"),
        log_level="DEBUG",
    )
    
    # No need to call init_db when using memory metadata storage
    
    return test_settings


class MockCircularObjectDBRepository:
    """Mock repository that doesn't check for image existence in database."""
    
    def __init__(self, db):
        self.db = db
        self.objects = []
    
    def create_object(self, image_id, bbox, centroid, radius):
        """Create a mock circular object without checking image existence."""
        obj = CircularObject(
            id=uuid4(),
            image_id=image_id,
            bbox=bbox,
            centroid=centroid,
            radius=radius
        )
        self.objects.append(obj)
        return obj
    
    def list_objects(self, image_id=None, limit=None, offset=0):
        """List mock objects."""
        if image_id:
            return [obj for obj in self.objects if obj.image_id == image_id]
        return self.objects
    
    def get_object(self, object_id):
        """Get a mock object by ID."""
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None


@pytest.fixture
def client(test_settings):
    """Create test client with overridden settings."""
    # Clear the settings cache first
    from config import get_settings
    get_settings.cache_clear()
    
    # Override the settings dependency
    app.dependency_overrides[get_settings] = lambda: test_settings
    
    # Clear any cached dependencies
    from aiq_circular_detection import dependencies
    dependencies._model_client = None
    dependencies._storage_client = None
    dependencies._in_memory_repository = None
    
    # Override the object repository to use our mock
    from aiq_circular_detection.dependencies import get_object_db_repository
    mock_repo = MockCircularObjectDBRepository(None)
    app.dependency_overrides[get_object_db_repository] = lambda: mock_repo
    
    # Mock init_db to avoid database initialization issues
    with patch('aiq_circular_detection.db.init_db'):
        with TestClient(app) as client:
            yield client
    
    # Clean up
    app.dependency_overrides.clear()
    
    # Clean up test database
    test_db_path = Path("test_db.sqlite3")
    if test_db_path.exists():
        test_db_path.unlink()
    
    # Clean up test storage
    if test_settings.storage_root.exists():
        import shutil
        shutil.rmtree(test_settings.storage_root)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a simple RGB image
    img = PILImage.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


class TestUploadEndToEnd:
    """End-to-end tests for image upload with model inference."""
    
    def test_upload_image_with_dummy_inference(self, client, sample_image):
        """Test that uploading an image returns dummy circle detections."""
        # Upload the image
        response = client.post(
            "/images/",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        
        # Check response status
        assert response.status_code == 200
        
        # Parse response
        data = response.json()
        
        # Verify image_id is present
        assert "image_id" in data
        assert len(data["image_id"]) == 64  # SHA-256 hash
        
        # Verify detection results
        assert "detection" in data
        detection = data["detection"]
        assert detection is not None
        assert "count" in detection
        assert "detections" in detection
        
        # Check dummy circles were returned
        assert detection["count"] == 1
        assert len(detection["detections"]) == 1
        
        # Verify the dummy circle data
        circle = detection["detections"][0]
        assert "object_id" in circle  # UUID of saved object
        assert circle["bbox"] == [10.0, 10.0, 50.0, 50.0]
        assert circle["centroid"] == {"x": 30.0, "y": 30.0}
        assert circle["radius"] == 20.0
    
    def test_upload_duplicate_image_returns_409(self, client, sample_image):
        """Test that uploading the same image twice returns 409 Conflict."""
        # First upload
        response1 = client.post(
            "/images/",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        assert response1.status_code == 200
        
        # Reset the BytesIO position
        sample_image.seek(0)
        
        # Second upload of same image
        response2 = client.post(
            "/images/",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        
        # Should return 409 Conflict
        assert response2.status_code == 409
        assert "already exists" in response2.json()["detail"]
    
    def test_list_objects_after_upload(self, client, sample_image):
        """Test that objects can be listed after upload."""
        # Upload image
        upload_response = client.post(
            "/images/",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        assert upload_response.status_code == 200
        
        image_id = upload_response.json()["image_id"]
        
        # List objects for the image
        list_response = client.get(f"/images/{image_id}/objects")
        assert list_response.status_code == 200
        
        data = list_response.json()
        assert data["image_id"] == image_id
        assert data["count"] == 1
        assert len(data["objects"]) == 1
        
        # Check the object summary
        obj = data["objects"][0]
        assert "object_id" in obj
        assert obj["bbox"] == [10, 10, 50, 50]  # Integer bbox in summary
    
    def test_get_object_detail_after_upload(self, client, sample_image):
        """Test that object details can be retrieved after upload."""
        # Upload image
        upload_response = client.post(
            "/images/",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        assert upload_response.status_code == 200
        
        data = upload_response.json()
        image_id = data["image_id"]
        object_id = data["detection"]["detections"][0]["object_id"]
        
        # Get object detail
        detail_response = client.get(f"/images/{image_id}/objects/{object_id}")
        assert detail_response.status_code == 200
        
        detail = detail_response.json()
        assert detail["object_id"] == object_id
        assert detail["bbox"] == [10, 10, 50, 50]
        assert detail["centroid"] == [30.0, 30.0]  # Tuple format in detail
        assert detail["radius"] == 20.0
    
    def test_empty_file_returns_400(self, client):
        """Test that uploading an empty file returns 400 Bad Request."""
        empty_file = io.BytesIO(b"")
        
        response = client.post(
            "/images/",
            files={"file": ("empty.jpg", empty_file, "image/jpeg")}
        )
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower() 