"""Unit tests for CircularObjectDBRepository."""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aiq_circular_detection.models.db import Base, Image
from aiq_circular_detection.repositories.object_db import CircularObjectDBRepository


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
def test_image(db_session):
    """Create a test image in the database."""
    image = Image(
        id="a" * 64,  # Valid SHA-256 hash format
        path="data/test-image.jpg"
    )
    db_session.add(image)
    db_session.commit()
    return image


@pytest.fixture
def repository(db_session):
    """Create a CircularObjectDBRepository instance for testing."""
    return CircularObjectDBRepository(db_session)


class TestCircularObjectDBRepository:
    """Test cases for CircularObjectDBRepository."""
    
    def test_create_object_success(self, repository, test_image):
        """Test successful circular object creation."""
        bbox = [10.0, 20.0, 30.0, 40.0]
        centroid = {"x": 20.0, "y": 30.0}
        radius = 10.0
        
        # Create object
        obj = repository.create_object(
            image_id=test_image.id,
            bbox=bbox,
            centroid=centroid,
            radius=radius
        )
        
        # Verify object properties
        assert obj.id is not None
        assert isinstance(obj.id, uuid.UUID)
        assert obj.image_id == test_image.id
        assert obj.bbox == bbox
        assert obj.centroid == centroid
        assert obj.radius == radius
    
    def test_create_object_invalid_image(self, repository):
        """Test creating object with non-existent image."""
        with pytest.raises(ValueError, match="Image with ID .* does not exist"):
            repository.create_object(
                image_id="nonexistent" * 8,  # 64 chars
                bbox=[1, 2, 3, 4],
                centroid={"x": 2.5, "y": 3.5},
                radius=1.5
            )
    
    def test_create_object_invalid_bbox(self, repository, test_image):
        """Test creating object with invalid bbox."""
        # Wrong number of elements
        with pytest.raises(ValueError, match="bbox must be a list of 4 floats"):
            repository.create_object(
                image_id=test_image.id,
                bbox=[1, 2, 3],  # Only 3 elements
                centroid={"x": 2.0, "y": 3.0},
                radius=1.0
            )
        
        # Not a list
        with pytest.raises(ValueError, match="bbox must be a list of 4 floats"):
            repository.create_object(
                image_id=test_image.id,
                bbox="not a list",
                centroid={"x": 2.0, "y": 3.0},
                radius=1.0
            )
    
    def test_create_object_invalid_centroid(self, repository, test_image):
        """Test creating object with invalid centroid."""
        # Missing 'x' key
        with pytest.raises(ValueError, match="centroid must be a dict with 'x' and 'y' keys"):
            repository.create_object(
                image_id=test_image.id,
                bbox=[1, 2, 3, 4],
                centroid={"y": 3.0},  # Missing 'x'
                radius=1.0
            )
        
        # Not a dict
        with pytest.raises(ValueError, match="centroid must be a dict with 'x' and 'y' keys"):
            repository.create_object(
                image_id=test_image.id,
                bbox=[1, 2, 3, 4],
                centroid="not a dict",
                radius=1.0
            )
    
    def test_create_object_invalid_radius(self, repository, test_image):
        """Test creating object with invalid radius."""
        # Zero radius
        with pytest.raises(ValueError, match="radius must be positive"):
            repository.create_object(
                image_id=test_image.id,
                bbox=[1, 2, 3, 4],
                centroid={"x": 2.5, "y": 3.5},
                radius=0
            )
        
        # Negative radius
        with pytest.raises(ValueError, match="radius must be positive"):
            repository.create_object(
                image_id=test_image.id,
                bbox=[1, 2, 3, 4],
                centroid={"x": 2.5, "y": 3.5},
                radius=-5.0
            )
    
    def test_get_object_exists(self, repository, test_image):
        """Test retrieving an existing object."""
        # Create object
        created_obj = repository.create_object(
            image_id=test_image.id,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2.5, "y": 3.5},
            radius=1.5
        )
        
        # Retrieve object
        retrieved_obj = repository.get_object(created_obj.id)
        
        assert retrieved_obj is not None
        assert retrieved_obj.id == created_obj.id
        assert retrieved_obj.image_id == created_obj.image_id
        assert retrieved_obj.bbox == created_obj.bbox
        assert retrieved_obj.centroid == created_obj.centroid
        assert retrieved_obj.radius == created_obj.radius
    
    def test_get_object_not_exists(self, repository):
        """Test retrieving non-existent object."""
        fake_id = uuid.uuid4()
        
        obj = repository.get_object(fake_id)
        
        assert obj is None
    
    def test_list_objects_all(self, repository, test_image):
        """Test listing all objects."""
        # Create multiple objects
        obj1 = repository.create_object(
            image_id=test_image.id,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2.5, "y": 3.5},
            radius=1.5
        )
        obj2 = repository.create_object(
            image_id=test_image.id,
            bbox=[5, 6, 7, 8],
            centroid={"x": 6.5, "y": 7.5},
            radius=2.5
        )
        
        # List all objects
        objects = repository.list_objects()
        
        assert len(objects) == 2
        assert any(o.id == obj1.id for o in objects)
        assert any(o.id == obj2.id for o in objects)
    
    def test_list_objects_by_image(self, repository, test_image, db_session):
        """Test listing objects filtered by image."""
        # Create another image
        other_image = Image(id="b" * 64, path="data/other-image.jpg")
        db_session.add(other_image)
        db_session.commit()
        
        # Create objects for different images
        obj1 = repository.create_object(
            image_id=test_image.id,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2.5, "y": 3.5},
            radius=1.5
        )
        obj2 = repository.create_object(
            image_id=other_image.id,
            bbox=[5, 6, 7, 8],
            centroid={"x": 6.5, "y": 7.5},
            radius=2.5
        )
        
        # List objects for specific image
        objects = repository.list_objects(image_id=test_image.id)
        
        assert len(objects) == 1
        assert objects[0].id == obj1.id
        assert objects[0].image_id == test_image.id
    
    def test_list_objects_with_limit_offset(self, repository, test_image):
        """Test listing objects with pagination."""
        # Create 5 objects
        created_objects = []
        for i in range(5):
            obj = repository.create_object(
                image_id=test_image.id,
                bbox=[i, i+1, i+2, i+3],
                centroid={"x": i+1.5, "y": i+2.5},
                radius=i+0.5
            )
            created_objects.append(obj)
        
        # Test limit
        objects = repository.list_objects(limit=2)
        assert len(objects) == 2
        
        # Test offset
        objects = repository.list_objects(offset=2)
        assert len(objects) == 3
        
        # Test limit + offset
        objects = repository.list_objects(limit=2, offset=2)
        assert len(objects) == 2
    
    def test_delete_object_success(self, repository, test_image):
        """Test successful object deletion."""
        # Create object
        obj = repository.create_object(
            image_id=test_image.id,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2.5, "y": 3.5},
            radius=1.5
        )
        
        # Delete object
        result = repository.delete_object(obj.id)
        
        assert result is True
        assert repository.get_object(obj.id) is None
        assert repository.count() == 0
    
    def test_delete_object_not_exists(self, repository):
        """Test deleting non-existent object."""
        fake_id = uuid.uuid4()
        
        result = repository.delete_object(fake_id)
        
        assert result is False
    
    def test_count_all(self, repository, test_image):
        """Test counting all objects."""
        assert repository.count() == 0
        
        # Add objects
        repository.create_object(
            image_id=test_image.id,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2.5, "y": 3.5},
            radius=1.5
        )
        assert repository.count() == 1
        
        repository.create_object(
            image_id=test_image.id,
            bbox=[5, 6, 7, 8],
            centroid={"x": 6.5, "y": 7.5},
            radius=2.5
        )
        assert repository.count() == 2
    
    def test_count_by_image(self, repository, test_image, db_session):
        """Test counting objects by image."""
        # Create another image
        other_image = Image(id="b" * 64, path="data/other-image.jpg")
        db_session.add(other_image)
        db_session.commit()
        
        # Create objects for different images
        repository.create_object(
            image_id=test_image.id,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2.5, "y": 3.5},
            radius=1.5
        )
        repository.create_object(
            image_id=test_image.id,
            bbox=[5, 6, 7, 8],
            centroid={"x": 6.5, "y": 7.5},
            radius=2.5
        )
        repository.create_object(
            image_id=other_image.id,
            bbox=[9, 10, 11, 12],
            centroid={"x": 10.5, "y": 11.5},
            radius=3.5
        )
        
        # Count for specific image
        assert repository.count(image_id=test_image.id) == 2
        assert repository.count(image_id=other_image.id) == 1
        assert repository.count() == 3  # Total
    
    def test_update_object_success(self, repository, test_image):
        """Test successful object update."""
        # Create object
        obj = repository.create_object(
            image_id=test_image.id,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2.5, "y": 3.5},
            radius=1.5
        )
        
        # Update object
        new_bbox = [10, 20, 30, 40]
        new_centroid = {"x": 20.0, "y": 30.0}
        new_radius = 15.0
        
        updated_obj = repository.update_object(
            object_id=obj.id,
            bbox=new_bbox,
            centroid=new_centroid,
            radius=new_radius
        )
        
        assert updated_obj is not None
        assert updated_obj.bbox == new_bbox
        assert updated_obj.centroid == new_centroid
        assert updated_obj.radius == new_radius
    
    def test_update_object_partial(self, repository, test_image):
        """Test partial object update."""
        # Create object
        original_bbox = [1, 2, 3, 4]
        original_centroid = {"x": 2.5, "y": 3.5}
        original_radius = 1.5
        
        obj = repository.create_object(
            image_id=test_image.id,
            bbox=original_bbox,
            centroid=original_centroid,
            radius=original_radius
        )
        
        # Update only radius
        new_radius = 5.0
        updated_obj = repository.update_object(
            object_id=obj.id,
            radius=new_radius
        )
        
        assert updated_obj is not None
        assert updated_obj.bbox == original_bbox
        assert updated_obj.centroid == original_centroid
        assert updated_obj.radius == new_radius
    
    def test_update_object_not_exists(self, repository):
        """Test updating non-existent object."""
        fake_id = uuid.uuid4()
        
        updated_obj = repository.update_object(
            object_id=fake_id,
            radius=5.0
        )
        
        assert updated_obj is None
    
    def test_update_object_invalid_values(self, repository, test_image):
        """Test updating object with invalid values."""
        # Create object
        obj = repository.create_object(
            image_id=test_image.id,
            bbox=[1, 2, 3, 4],
            centroid={"x": 2.5, "y": 3.5},
            radius=1.5
        )
        
        # Invalid bbox
        with pytest.raises(ValueError, match="bbox must be a list of 4 floats"):
            repository.update_object(obj.id, bbox=[1, 2, 3])  # Wrong length
        
        # Invalid centroid
        with pytest.raises(ValueError, match="centroid must be a dict with 'x' and 'y' keys"):
            repository.update_object(obj.id, centroid={"x": 1.0})  # Missing 'y'
        
        # Invalid radius
        with pytest.raises(ValueError, match="radius must be positive"):
            repository.update_object(obj.id, radius=-1.0)
    
    def test_transaction_rollback(self, repository, test_image, db_session):
        """Test that transactions are properly rolled back on error."""
        # Mock a database error during commit
        with patch.object(db_session, 'commit', side_effect=Exception("DB error")):
            with pytest.raises(Exception, match="DB error"):
                repository.create_object(
                    image_id=test_image.id,
                    bbox=[1, 2, 3, 4],
                    centroid={"x": 2.5, "y": 3.5},
                    radius=1.5
                )
        
        # Verify no object was saved
        assert repository.count() == 0 