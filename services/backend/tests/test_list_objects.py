"""Tests for the GET /images/{image_id}/objects endpoint."""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aiq_circular_detection.db import get_db
from aiq_circular_detection.dependencies import get_image_repository
from aiq_circular_detection.main import app
from aiq_circular_detection.models.db import Base, CircularObject, Image
from aiq_circular_detection.repositories.image import ImageDBRepository, InMemoryImageRepository


@pytest.fixture
def db_engine():
    """Create an in-memory SQLite database engine for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def override_get_db(db_session):
    """Override the get_db dependency to use the test session."""
    def _get_test_db():
        try:
            yield db_session
        finally:
            pass  # Don't close the session here, it's handled by the fixture
    
    app.dependency_overrides[get_db] = _get_test_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def override_get_image_repository(db_session):
    """Override the get_image_repository dependency to use the database repository."""
    # Create a single instance for all requests in this test
    image_repo = ImageDBRepository(db_session)
    
    def _get_test_image_repository():
        return image_repo
    
    app.dependency_overrides[get_image_repository] = _get_test_image_repository
    yield
    # Clear is handled by override_get_db fixture


@pytest.fixture
def client(override_get_db, override_get_image_repository, db_session):
    """Create a test client with the overridden database and repository."""
    # db_session is included to ensure tables are created even if not used directly
    return TestClient(app)


@pytest.fixture
def client_in_memory():
    """Create a test client with in-memory repository for tests that don't need persistence."""
    # Override to use in-memory repository
    def _get_test_image_repository():
        return InMemoryImageRepository()
    
    app.dependency_overrides[get_image_repository] = _get_test_image_repository
    
    yield TestClient(app)
    
    app.dependency_overrides.clear()


@pytest.fixture
def test_image(db_session):
    """Create a test image in the database."""
    image = Image(
        id="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
        path="data/test-image.jpg"
    )
    db_session.add(image)
    db_session.commit()
    db_session.refresh(image)
    return image


@pytest.fixture
def test_circular_objects(db_session, test_image):
    """Create test circular objects in the database."""
    objects = []
    
    # Create 3 circular objects with different properties
    object_data = [
        {
            "bbox": [10.5, 20.5, 30.5, 40.5],
            "centroid": {"x": 20.5, "y": 30.5},
            "radius": 10.0
        },
        {
            "bbox": [50.2, 60.7, 150.8, 160.3],
            "centroid": {"x": 100.5, "y": 110.5},
            "radius": 50.3
        },
        {
            "bbox": [200.1, 210.9, 250.4, 260.6],
            "centroid": {"x": 225.25, "y": 235.75},
            "radius": 25.25
        }
    ]
    
    for data in object_data:
        obj = CircularObject(
            image_id=test_image.id,
            bbox=data["bbox"],
            centroid=data["centroid"],
            radius=data["radius"]
        )
        db_session.add(obj)
        objects.append(obj)
    
    db_session.commit()
    
    # Refresh all objects to get their IDs
    for obj in objects:
        db_session.refresh(obj)
    
    return objects


class TestListObjectsEndpoint:
    """Test cases for the GET /images/{image_id}/objects endpoint."""
    
    def test_list_objects_success(self, client, test_image, test_circular_objects):
        """Test successfully listing objects for an image."""
        response = client.get(f"/images/{test_image.id}/objects")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check the response structure
        assert "image_id" in data
        assert "count" in data
        assert "objects" in data
        
        # Verify metadata
        assert data["image_id"] == test_image.id
        assert data["count"] == 3
        
        # Get the objects list
        objects = data["objects"]
        
        # Should return a list of ObjectSummary
        assert isinstance(objects, list)
        assert len(objects) == 3
        
        # Verify each object has correct structure
        for i, obj_summary in enumerate(objects):
            # Check required fields
            assert "object_id" in obj_summary
            assert "bbox" in obj_summary
            
            # Verify object_id is a valid UUID string
            assert isinstance(obj_summary["object_id"], str)
            uuid.UUID(obj_summary["object_id"])  # Should not raise
            
            # Verify bbox is list of integers
            assert isinstance(obj_summary["bbox"], list)
            assert len(obj_summary["bbox"]) == 4
            assert all(isinstance(coord, int) for coord in obj_summary["bbox"])
        
        # Verify the actual data matches (note: order might vary)
        returned_ids = {obj["object_id"] for obj in objects}
        expected_ids = {str(obj.id) for obj in test_circular_objects}
        assert returned_ids == expected_ids
        
        # Check that bbox values are properly converted to integers
        for obj_summary in objects:
            # Find the corresponding test object
            test_obj = next(
                obj for obj in test_circular_objects 
                if str(obj.id) == obj_summary["object_id"]
            )
            
            # Verify bbox conversion (float to int)
            expected_bbox = [int(coord) for coord in test_obj.bbox]
            assert obj_summary["bbox"] == expected_bbox
    
    def test_list_objects_empty_image(self, client, test_image):
        """Test listing objects for an image with no objects."""
        # test_image exists but has no objects (no test_circular_objects fixture)
        response = client.get(f"/images/{test_image.id}/objects")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["image_id"] == test_image.id
        assert data["count"] == 0
        assert data["objects"] == []
    
    def test_list_objects_nonexistent_image(self, client_in_memory):
        """Test listing objects for non-existent image returns 404."""
        fake_image_id = "b" * 64  # Valid SHA-256 format but doesn't exist
        
        response = client_in_memory.get(f"/images/{fake_image_id}/objects")
        
        # Should return 404 with error message
        assert response.status_code == 404
        data = response.json()
        assert data == {"detail": "Image not found"}
    
    def test_list_objects_invalid_image_id_format(self, client_in_memory):
        """Test listing objects with invalid image ID format returns 404."""
        invalid_id = "not-a-valid-hash"
        
        response = client_in_memory.get(f"/images/{invalid_id}/objects")
        
        # Should return 404 as the image won't exist
        assert response.status_code == 404
        assert response.json() == {"detail": "Image not found"}
    
    def test_list_objects_missing_image_id(self, client_in_memory):
        """Test listing objects for missing image_id returns 404 with proper error message."""
        # Use a valid SHA-256 hash format that doesn't exist in the database
        missing_image_id = "0" * 64  # Valid format but doesn't exist
        
        response = client_in_memory.get(f"/images/{missing_image_id}/objects")
        
        # Assert 404 status code
        assert response.status_code == 404
        
        # Assert the JSON response matches exactly
        error_response = response.json()
        assert error_response == {"detail": "Image not found"}
        
        # Verify it's proper JSON
        assert response.headers["content-type"] == "application/json"
    
    def test_list_objects_correct_filtering(self, client, db_session):
        """Test that objects are correctly filtered by image ID."""
        # Create two images
        image1 = Image(
            id="a" * 64,
            path="data/image1.jpg"
        )
        image2 = Image(
            id="b" * 64,
            path="data/image2.jpg"
        )
        db_session.add_all([image1, image2])
        db_session.commit()
        
        # Create objects for each image
        obj1 = CircularObject(
            image_id=image1.id,
            bbox=[1.0, 2.0, 3.0, 4.0],
            centroid={"x": 2.0, "y": 3.0},
            radius=1.0
        )
        obj2 = CircularObject(
            image_id=image2.id,
            bbox=[5.0, 6.0, 7.0, 8.0],
            centroid={"x": 6.0, "y": 7.0},
            radius=2.0
        )
        db_session.add_all([obj1, obj2])
        db_session.commit()
        db_session.refresh(obj1)
        db_session.refresh(obj2)
        
        # Get objects for image1
        response = client.get(f"/images/{image1.id}/objects")
        assert response.status_code == 200
        data = response.json()
        
        # Check response for image1
        assert data["image_id"] == image1.id
        assert data["count"] == 1
        assert len(data["objects"]) == 1
        assert data["objects"][0]["object_id"] == str(obj1.id)
        assert data["objects"][0]["bbox"] == [1, 2, 3, 4]
        
        # Get objects for image2
        response = client.get(f"/images/{image2.id}/objects")
        assert response.status_code == 200
        data = response.json()
        
        # Check response for image2
        assert data["image_id"] == image2.id
        assert data["count"] == 1
        assert len(data["objects"]) == 1
        assert data["objects"][0]["object_id"] == str(obj2.id)
        assert data["objects"][0]["bbox"] == [5, 6, 7, 8]
    
    def test_list_objects_bbox_rounding(self, client, db_session, test_image):
        """Test that bbox coordinates are properly rounded to integers."""
        # Create object with precise float coordinates
        obj = CircularObject(
            image_id=test_image.id,
            bbox=[10.1, 20.9, 30.4, 40.6],  # Various decimal places
            centroid={"x": 20.25, "y": 30.75},
            radius=10.25
        )
        db_session.add(obj)
        db_session.commit()
        db_session.refresh(obj)
        
        response = client.get(f"/images/{test_image.id}/objects")
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["image_id"] == test_image.id
        assert data["count"] == 1
        assert len(data["objects"]) == 1
        
        # int() truncates towards zero, not rounds
        assert data["objects"][0]["bbox"] == [10, 20, 30, 40]
    
    def test_endpoint_response_format(self, client, test_image, test_circular_objects):
        """Test that the endpoint returns proper JSON format."""
        response = client.get(f"/images/{test_image.id}/objects")
        
        # Check response headers
        assert response.headers["content-type"] == "application/json"
        
        # Verify it's valid JSON (will raise if not)
        data = response.json()
        
        # Check it matches the expected schema
        assert isinstance(data, dict)
        
        # Check top-level required fields
        assert set(data.keys()) == {"image_id", "count", "objects"}
        
        # Check field types
        assert isinstance(data["image_id"], str)
        assert isinstance(data["count"], int)
        assert isinstance(data["objects"], list)
        
        # Check image_id format (SHA-256 hash)
        assert len(data["image_id"]) == 64
        assert all(c in "0123456789abcdef" for c in data["image_id"])
        
        # Check count matches objects length
        assert data["count"] == len(data["objects"])
        
        if data["objects"]:  # If there are objects
            # Each item should have exactly these fields
            for item in data["objects"]:
                assert set(item.keys()) == {"object_id", "bbox"}
    
    def test_openapi_documentation(self, client):
        """Test that the endpoint is properly documented in OpenAPI schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi = response.json()
        
        # Check that the endpoint is documented
        path = "/images/{image_id}/objects"
        assert path in openapi["paths"]
        
        # Check the GET operation
        get_operation = openapi["paths"][path]["get"]
        assert "tags" in get_operation
        assert "objects" in get_operation["tags"]
        
        # Check response schema
        assert "responses" in get_operation
        assert "200" in get_operation["responses"]
        
        # The response should reference the ObjectListSummaryResponse schema
        response_content = get_operation["responses"]["200"]["content"]["application/json"]
        assert "schema" in response_content
        
        # Check that it references ObjectListSummaryResponse
        schema_ref = response_content["schema"]
        if "$ref" in schema_ref:
            # Should reference ObjectListSummaryResponse
            assert "ObjectListSummaryResponse" in schema_ref["$ref"]
        
        # Check that the schema is defined in components
        assert "components" in openapi
        assert "schemas" in openapi["components"]
        assert "ObjectListSummaryResponse" in openapi["components"]["schemas"]
        
        # Verify the schema has expected properties
        response_schema = openapi["components"]["schemas"]["ObjectListSummaryResponse"]
        assert "properties" in response_schema
        assert "image_id" in response_schema["properties"]
        assert "count" in response_schema["properties"]
        assert "objects" in response_schema["properties"]