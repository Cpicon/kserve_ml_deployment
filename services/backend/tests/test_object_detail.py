"""Tests for the object detail endpoint."""

import logging
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from aiq_circular_detection.main import app
from aiq_circular_detection.models.db import CircularObject


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_repos():
    """Create mock repositories."""
    # Create the mocks
    image_repo = MagicMock()
    object_repo = MagicMock()
    
    # Override the dependency functions
    def override_get_image_repository():
        return image_repo
    
    def override_get_object_db_repository():
        return object_repo
    
    from aiq_circular_detection.dependencies import get_image_repository, get_object_db_repository
    
    app.dependency_overrides[get_image_repository] = override_get_image_repository
    app.dependency_overrides[get_object_db_repository] = override_get_object_db_repository
    
    yield image_repo, object_repo
    
    # Clean up
    app.dependency_overrides.clear()


def test_get_object_detail_success(client, mock_repos, caplog):
    """Test successful retrieval of object details."""
    image_repo, object_repo = mock_repos
    
    # Test data
    image_id = "a" * 64  # Valid SHA-256 hash
    object_id = str(uuid4())
    
    # Mock image exists
    image_repo.exists.return_value = True
    
    # Mock object data
    mock_object = MagicMock(spec=CircularObject)
    mock_object.id = UUID(object_id)
    mock_object.image_id = image_id
    mock_object.bbox = [10.5, 20.5, 30.5, 40.5]
    mock_object.centroid = {"x": 20.5, "y": 30.5}
    mock_object.radius = 10.5
    
    object_repo.get_object.return_value = mock_object
    
    # Make request with logging capture
    with caplog.at_level(logging.INFO):
        response = client.get(f"/images/{image_id}/objects/{object_id}")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["object_id"] == object_id
    assert data["bbox"] == [10, 20, 30, 40]  # Should be converted to int
    assert data["centroid"] == [20.5, 30.5]
    assert data["radius"] == 10.5
    
    # Verify logging
    assert f"Returning object {object_id} for image {image_id}." in caplog.text
    
    # Verify repository calls
    image_repo.exists.assert_called_once_with(image_id)
    object_repo.get_object.assert_called_once_with(UUID(object_id))


def test_get_object_detail_image_not_found(client, mock_repos, caplog):
    """Test 404 when image doesn't exist."""
    image_repo, object_repo = mock_repos
    
    # Test data
    image_id = "a" * 64
    object_id = str(uuid4())
    
    # Mock image doesn't exist
    image_repo.exists.return_value = False
    
    # Make request with logging capture
    with caplog.at_level(logging.WARNING):
        response = client.get(f"/images/{image_id}/objects/{object_id}")
    
    # Verify response
    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"
    
    # Verify logging
    assert f"Image not found: {image_id}" in caplog.text
    
    # Verify repository calls
    image_repo.exists.assert_called_once_with(image_id)
    object_repo.get_object.assert_not_called()


def test_get_object_detail_object_not_found(client, mock_repos, caplog):
    """Test 404 when object doesn't exist."""
    image_repo, object_repo = mock_repos
    
    # Test data
    image_id = "a" * 64
    object_id = str(uuid4())
    
    # Mock image exists but object doesn't
    image_repo.exists.return_value = True
    object_repo.get_object.return_value = None
    
    # Make request with logging capture
    with caplog.at_level(logging.WARNING):
        response = client.get(f"/images/{image_id}/objects/{object_id}")
    
    # Verify response
    assert response.status_code == 404
    assert response.json()["detail"] == "Object not found"
    
    # Verify logging
    assert f"Object not found: {object_id}" in caplog.text
    
    # Verify repository calls
    image_repo.exists.assert_called_once_with(image_id)
    object_repo.get_object.assert_called_once_with(UUID(object_id))


def test_get_object_detail_wrong_image(client, mock_repos, caplog):
    """Test 404 when object belongs to different image."""
    image_repo, object_repo = mock_repos
    
    # Test data
    image_id = "a" * 64
    different_image_id = "b" * 64
    object_id = str(uuid4())
    
    # Mock image exists
    image_repo.exists.return_value = True
    
    # Mock object belongs to different image
    mock_object = MagicMock(spec=CircularObject)
    mock_object.id = UUID(object_id)
    mock_object.image_id = different_image_id  # Different image!
    mock_object.bbox = [10.5, 20.5, 30.5, 40.5]
    mock_object.centroid = {"x": 20.5, "y": 30.5}
    mock_object.radius = 10.5
    
    object_repo.get_object.return_value = mock_object
    
    # Make request with logging capture
    with caplog.at_level(logging.WARNING):
        response = client.get(f"/images/{image_id}/objects/{object_id}")
    
    # Verify response
    assert response.status_code == 404
    assert response.json()["detail"] == "Object not found"
    
    # Verify logging
    assert f"Object {object_id} does not belong to image {image_id}" in caplog.text
    
    # Verify repository calls
    image_repo.exists.assert_called_once_with(image_id)
    object_repo.get_object.assert_called_once_with(UUID(object_id))


def test_get_object_detail_invalid_object_id(client, mock_repos, caplog):
    """Test 404 when object ID is invalid UUID."""
    image_repo, object_repo = mock_repos
    
    # Test data
    image_id = "a" * 64
    invalid_object_id = "not-a-uuid"
    
    # Mock image exists
    image_repo.exists.return_value = True
    
    # Make request with logging capture
    with caplog.at_level(logging.WARNING):
        response = client.get(f"/images/{image_id}/objects/{invalid_object_id}")
    
    # Verify response
    assert response.status_code == 404
    assert response.json()["detail"] == "Object not found"
    
    # Verify logging
    assert f"Invalid object ID format: {invalid_object_id}" in caplog.text
    
    # Verify repository calls
    image_repo.exists.assert_called_once_with(image_id)
    object_repo.get_object.assert_not_called()


def test_list_objects_logging(client, mock_repos, caplog):
    """Test logging for list objects endpoint."""
    image_repo, object_repo = mock_repos
    
    # Test data
    image_id = "a" * 64
    object_id1 = uuid4()
    object_id2 = uuid4()
    
    # Mock image exists
    image_repo.exists.return_value = True
    
    # Mock objects
    mock_objects = []
    for obj_id in [object_id1, object_id2]:
        mock_obj = MagicMock(spec=CircularObject)
        mock_obj.id = obj_id
        mock_obj.image_id = image_id
        mock_obj.bbox = [10.5, 20.5, 30.5, 40.5]
        mock_objects.append(mock_obj)
    
    object_repo.list_objects.return_value = mock_objects
    
    # Make request with logging capture
    with caplog.at_level(logging.INFO):
        response = client.get(f"/images/{image_id}/objects")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["objects"]) == 2
    
    # Verify logging
    assert f"Listing 2 objects for image {image_id}." in caplog.text
    
    # Verify repository calls
    image_repo.exists.assert_called_once_with(image_id)
    object_repo.list_objects.assert_called_once_with(image_id=image_id)


def test_list_objects_empty_logging(client, mock_repos, caplog):
    """Test logging for list objects endpoint when no objects found."""
    image_repo, object_repo = mock_repos
    
    # Test data
    image_id = "a" * 64
    
    # Mock image exists but no objects
    image_repo.exists.return_value = True
    object_repo.list_objects.return_value = []
    
    # Make request with logging capture
    with caplog.at_level(logging.INFO):
        response = client.get(f"/images/{image_id}/objects")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["objects"] == []
    
    # Verify logging
    assert f"Listing 0 objects for image {image_id}." in caplog.text


def test_object_detail_response_schema():
    """Test ObjectDetailResponse schema validation."""
    from aiq_circular_detection.schemas.object import ObjectDetailResponse
    
    # Valid data
    valid_data = {
        "object_id": str(uuid4()),
        "bbox": [10, 20, 30, 40],
        "centroid": (20.5, 30.5),
        "radius": 10.5
    }
    
    # Create instance
    response = ObjectDetailResponse(**valid_data)
    
    # Verify fields
    assert str(response.object_id) == valid_data["object_id"]
    assert response.bbox == valid_data["bbox"]
    assert response.centroid == valid_data["centroid"]
    assert response.radius == valid_data["radius"]
    
    # Test invalid data
    with pytest.raises(ValueError):
        # Invalid bbox length
        ObjectDetailResponse(
            object_id=str(uuid4()),
            bbox=[10, 20, 30],  # Only 3 elements
            centroid=(20.5, 30.5),
            radius=10.5
        )
    
    with pytest.raises(ValueError):
        # Invalid radius (negative)
        ObjectDetailResponse(
            object_id=str(uuid4()),
            bbox=[10, 20, 30, 40],
            centroid=(20.5, 30.5),
            radius=-10.5
        ) 