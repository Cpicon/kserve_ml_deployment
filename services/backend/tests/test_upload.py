"""Tests for image upload endpoint."""
import hashlib
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aiq_circular_detection.dependencies import get_image_repository
from aiq_circular_detection.main import app


@pytest.fixture(autouse=True)
def clear_repository():
    """Clear the image repository before each test."""
    repository = get_image_repository()
    if hasattr(repository, 'clear'):
        repository.clear()
    yield
    if hasattr(repository, 'clear'):
        repository.clear()


def create_test_jpeg(width: int = 10, height: int = 10, seed: int = 0) -> bytes:
    """Create a small test JPEG image.
    
    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        seed: Seed for generating different colored images.
        
    Returns:
        bytes: JPEG image data.
    """
    # Create a simple RGB image with color based on seed
    color = ((seed * 50) % 256, (seed * 100) % 256, (seed * 150) % 256)
    img = Image.new("RGB", (width, height), color=color)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    
    return img_bytes.read()


def test_upload_image_success():
    """Test successful image upload."""
    client = TestClient(app)
    repository = get_image_repository()
    
    # Create test image data
    test_image = create_test_jpeg()
    
    # Upload the image
    response = client.post(
        "/images/",
        files={"file": ("test.jpg", test_image, "image/jpeg")}
    )
    
    # Check response status
    assert response.status_code == 200
    
    # Check response body
    data = response.json()
    assert "image_id" in data
    
    # Verify the image_id is a valid SHA-256 hash
    image_id = data["image_id"]
    assert len(image_id) == 64
    assert all(c in "0123456789abcdef" for c in image_id)
    
    # Verify the ID matches the expected hash
    expected_id = hashlib.sha256(test_image).hexdigest()
    assert image_id == expected_id
    
    # Verify the mapping was stored in repository
    assert repository.exists(image_id)
    file_path = repository.get_path(image_id)
    assert file_path.endswith(".jpg")
    
    # Verify the file exists on disk
    assert Path(file_path).exists()


def test_upload_image_empty_file():
    """Test uploading an empty file."""
    client = TestClient(app)
    
    # Upload empty file
    response = client.post(
        "/images/",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    
    # Should fail with 400 Bad Request
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_duplicate_image():
    """Test uploading the same image twice returns 409 Conflict."""
    client = TestClient(app)
    
    # Create test image
    test_image = create_test_jpeg(seed=42)
    
    # Upload first time
    response1 = client.post(
        "/images/",
        files={"file": ("test1.jpg", test_image, "image/jpeg")}
    )
    assert response1.status_code == 200
    image_id1 = response1.json()["image_id"]
    
    # Upload same image again
    response2 = client.post(
        "/images/",
        files={"file": ("test2.jpg", test_image, "image/jpeg")}
    )
    
    # Should fail with 409 Conflict
    assert response2.status_code == 409
    detail = response2.json()["detail"]
    assert "already exists" in detail
    assert image_id1 in detail


def test_upload_multiple_different_images():
    """Test uploading multiple different images generates different IDs."""
    client = TestClient(app)
    repository = get_image_repository()
    
    image_ids = []
    
    # Upload 3 different images
    for i in range(3):
        test_image = create_test_jpeg(width=10 + i, height=10 + i, seed=i)
        response = client.post(
            "/images/",
            files={"file": (f"test_{i}.jpg", test_image, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        image_ids.append(data["image_id"])
    
    # Verify all IDs are unique
    assert len(set(image_ids)) == 3
    
    # Verify all mappings exist
    for image_id in image_ids:
        assert repository.exists(image_id)


def test_upload_large_filename():
    """Test uploading a file with a very long filename."""
    client = TestClient(app)
    
    # Create test image
    test_image = create_test_jpeg(seed=99)
    
    # Create a long filename
    long_filename = "very_" * 50 + "long_filename.jpg"
    
    # Upload the image
    response = client.post(
        "/images/",
        files={"file": (long_filename, test_image, "image/jpeg")}
    )
    
    # Should still succeed
    assert response.status_code == 200
    data = response.json()
    assert "image_id" in data
    
    # Verify it's a valid SHA-256 hash
    image_id = data["image_id"]
    assert len(image_id) == 64


def test_upload_png_as_jpeg():
    """Test uploading a PNG image (will be saved as .jpg by storage)."""
    client = TestClient(app)
    repository = get_image_repository()
    
    # Create a PNG image
    img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    png_content = img_bytes.read()
    
    # Upload as PNG
    response = client.post(
        "/images/",
        files={"file": ("test.png", png_content, "image/png")}
    )
    
    # Should succeed
    assert response.status_code == 200
    data = response.json()
    
    # Verify ID is based on PNG content
    expected_id = hashlib.sha256(png_content).hexdigest()
    assert data["image_id"] == expected_id
    
    # File should be saved with .jpg extension
    file_path = repository.get_path(data["image_id"])
    assert file_path.endswith(".jpg")


def test_repository_isolation():
    """Test that repository is properly isolated between tests."""
    repository = get_image_repository()
    
    # Repository should be empty at the start of the test
    assert repository.count() == 0


def test_deterministic_ids():
    """Test that the same image content always produces the same ID."""
    client = TestClient(app)
    _ = get_image_repository()
    
    # Create a specific image
    test_image = create_test_jpeg(width=20, height=20, seed=123)
    
    # Upload it
    response = client.post(
        "/images/",
        files={"file": ("test.jpg", test_image, "image/jpeg")}
    )
    assert response.status_code == 200
    
    # The ID should be deterministic
    expected_id = hashlib.sha256(test_image).hexdigest()
    assert response.json()["image_id"] == expected_id 