"""Unit tests for SQLAlchemy models."""

import uuid

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from aiq_circular_detection.models import Base, CircularObject, Image


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    
    # Enable foreign key constraints in SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def test_session(test_engine):
    """Create a test database session."""
    SessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    yield session
    session.close()


class TestImageModel:
    """Test cases for the Image model."""
    
    def test_create_image(self, test_session: Session):
        """Test creating an Image instance."""
        # Create an image with SHA-256 hash ID
        image_id = "a" * 64  # Valid SHA-256 hash format
        image = Image(
            id=image_id,
            path="data/images/test.jpg"
        )
        
        test_session.add(image)
        test_session.commit()
        
        # Retrieve and verify
        retrieved = test_session.query(Image).filter_by(id=image_id).first()
        assert retrieved is not None
        assert retrieved.id == image_id
        assert retrieved.path == "data/images/test.jpg"
        assert retrieved.circular_objects == []
    
    def test_image_repr(self):
        """Test Image string representation."""
        image_id = "abcdef1234567890" + "a" * 48
        image = Image(id=image_id, path="test.jpg")
        
        repr_str = repr(image)
        assert "Image" in repr_str
        assert "abcdef12..." in repr_str  # First 8 chars of ID
        assert "test.jpg" in repr_str
    
    def test_unique_path_constraint(self, test_session: Session):
        """Test that path must be unique."""
        # Create first image
        image1 = Image(id="a" * 64, path="test.jpg")
        test_session.add(image1)
        test_session.commit()
        
        # Try to create second image with same path
        image2 = Image(id="b" * 64, path="test.jpg")
        test_session.add(image2)
        
        with pytest.raises(Exception):  # SQLAlchemy will raise IntegrityError
            test_session.commit()
    
    def test_cascade_delete(self, test_session: Session):
        """Test that deleting an image cascades to circular objects."""
        # Create image with circular objects
        image = Image(id="a" * 64, path="test.jpg")
        
        circle1 = CircularObject(
            image=image,
            bbox=[10, 20, 30, 40],
            centroid={"x": 20.0, "y": 30.0},
            radius=10.0
        )
        circle2 = CircularObject(
            image=image,
            bbox=[50, 60, 70, 80],
            centroid={"x": 60.0, "y": 70.0},
            radius=10.0
        )
        
        test_session.add(image)
        test_session.add(circle1)
        test_session.add(circle2)
        test_session.commit()
        
        # Verify objects exist
        assert test_session.query(CircularObject).count() == 2
        
        # Delete image
        test_session.delete(image)
        test_session.commit()
        
        # Verify circular objects were deleted
        assert test_session.query(CircularObject).count() == 0


class TestCircularObjectModel:
    """Test cases for the CircularObject model."""
    
    def test_create_circular_object(self, test_session: Session):
        """Test creating a CircularObject instance."""
        # Create parent image first
        image = Image(id="a" * 64, path="test.jpg")
        test_session.add(image)
        test_session.commit()
        
        # Create circular object
        circle = CircularObject(
            image_id=image.id,
            bbox=[10.5, 20.5, 30.5, 40.5],
            centroid={"x": 20.5, "y": 30.5},
            radius=10.0
        )
        
        test_session.add(circle)
        test_session.commit()
        
        # Verify auto-generated UUID
        assert circle.id is not None
        assert isinstance(circle.id, uuid.UUID)
        
        # Retrieve and verify
        retrieved = test_session.query(CircularObject).filter_by(id=circle.id).first()
        assert retrieved is not None
        assert retrieved.image_id == image.id
        assert retrieved.bbox == [10.5, 20.5, 30.5, 40.5]
        assert retrieved.centroid == {"x": 20.5, "y": 30.5}
        assert retrieved.radius == 10.0
    
    def test_circular_object_repr(self):
        """Test CircularObject string representation."""
        obj_id = uuid.uuid4()
        image_id = "abcdef1234567890" + "a" * 48
        circle = CircularObject(
            id=obj_id,
            image_id=image_id,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2, "y": 3},
            radius=1.0
        )
        
        repr_str = repr(circle)
        assert "CircularObject" in repr_str
        assert str(obj_id) in repr_str
        assert "abcdef12..." in repr_str
        assert "{'x': 2, 'y': 3}" in repr_str
        assert "1.0" in repr_str
    
    def test_bbox_list_property(self):
        """Test bbox_list property."""
        circle = CircularObject(
            id=uuid.uuid4(),
            image_id="a" * 64,
            bbox=[1.0, 2.0, 3.0, 4.0],
            centroid={"x": 2.0, "y": 3.0},
            radius=1.0
        )
        
        assert circle.bbox_list == [1.0, 2.0, 3.0, 4.0]
        
        # Test with invalid bbox
        circle.bbox = "invalid"
        assert circle.bbox_list == []
    
    def test_centroid_dict_property(self):
        """Test centroid_dict property."""
        circle = CircularObject(
            id=uuid.uuid4(),
            image_id="a" * 64,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2.5, "y": 3.5},
            radius=1.0
        )
        
        assert circle.centroid_dict == {"x": 2.5, "y": 3.5}
        
        # Test with invalid centroid
        circle.centroid = "invalid"
        assert circle.centroid_dict == {"x": 0.0, "y": 0.0}
    
    def test_relationship_navigation(self, test_session: Session):
        """Test navigating relationships between Image and CircularObject."""
        # Create image
        image = Image(id="a" * 64, path="test.jpg")
        
        # Create multiple circular objects
        circles = [
            CircularObject(
                image=image,
                bbox=[i*10, i*10, i*10+20, i*10+20],
                centroid={"x": i*10+10, "y": i*10+10},
                radius=10.0
            )
            for i in range(3)
        ]
        
        test_session.add(image)
        test_session.add_all(circles)
        test_session.commit()
        
        # Test navigation from image to circles
        retrieved_image = test_session.query(Image).first()
        assert len(retrieved_image.circular_objects) == 3
        
        # Test navigation from circle to image
        retrieved_circle = test_session.query(CircularObject).first()
        assert retrieved_circle.image == retrieved_image
        assert retrieved_circle.image.path == "test.jpg"
    
    def test_json_fields_storage(self, test_session: Session):
        """Test that JSON fields properly store complex data."""
        image = Image(id="a" * 64, path="test.jpg")
        
        # Test with nested data
        complex_bbox = [10.123456, 20.789012, 30.345678, 40.901234]
        complex_centroid = {"x": 20.123456789, "y": 30.987654321}
        
        circle = CircularObject(
            image=image,
            bbox=complex_bbox,
            centroid=complex_centroid,
            radius=15.123456789
        )
        
        test_session.add(image)
        test_session.add(circle)
        test_session.commit()
        
        # Retrieve and verify JSON data integrity
        retrieved = test_session.query(CircularObject).first()
        assert retrieved.bbox == complex_bbox
        assert retrieved.centroid == complex_centroid
        assert abs(retrieved.radius - 15.123456789) < 0.000001
    
    def test_foreign_key_constraint(self, test_session: Session):
        """Test foreign key constraint enforcement."""
        # Try to create circular object with non-existent image_id
        circle = CircularObject(
            image_id="nonexistent" + "x" * 54,  # 64 chars but doesn't exist
            bbox=[1, 2, 3, 4],
            centroid={"x": 2, "y": 3},
            radius=1.0
        )
        
        test_session.add(circle)
        with pytest.raises(IntegrityError):  # Foreign key violation
            test_session.commit()


class TestPydanticCompatibility:
    """Test Pydantic compatibility features."""
    
    def test_image_dict_structure(self):
        """Test ImageDict structure."""
        from aiq_circular_detection.models.db import ImageDict
        
        # ImageDict should be a dict subclass with type hints
        assert issubclass(ImageDict, dict)
        
        # Check annotations
        annotations = ImageDict.__annotations__
        assert "id" in annotations
        assert "path" in annotations
        assert "circular_objects" in annotations
    
    def test_circular_object_dict_structure(self):
        """Test CircularObjectDict structure."""
        from aiq_circular_detection.models.db import CircularObjectDict
        
        # CircularObjectDict should be a dict subclass with type hints
        assert issubclass(CircularObjectDict, dict)
        
        # Check annotations
        annotations = CircularObjectDict.__annotations__
        assert "id" in annotations
        assert "image_id" in annotations
        assert "bbox" in annotations
        assert "centroid" in annotations
        assert "radius" in annotations 