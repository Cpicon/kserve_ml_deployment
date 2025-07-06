"""Tests for database configuration and session management."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import MetaData, create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker
import config
from aiq_circular_detection.db.database import get_db
from aiq_circular_detection.models import Base, CircularObject, Image
import aiq_circular_detection.db.database as db_module

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.database_url = "sqlite:///:memory:"
    settings.database_echo = False
    return settings

@pytest.fixture
def temp_engine(mock_settings):
    """Create a temporary SQLite engine for testing."""
    engine = create_engine(
        mock_settings.database_url,
        connect_args={"check_same_thread": False}
    )
    
    # Enable foreign key constraints
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    yield engine
    
    engine.dispose()


@pytest.fixture
def temp_session(temp_engine):
    """Create a temporary database session."""
    SessionLocal = sessionmaker(bind=temp_engine)
    session = SessionLocal()
    yield session
    session.close()

class TestDatabaseSetup:
    """Test database setup and configuration."""
    
    def test_create_all_tables(self, temp_engine):
        """Test that all tables can be created in a temp SQLite file."""
        # Create all tables
        Base.metadata.create_all(bind=temp_engine)
        
        # Get inspector to check tables
        inspector = inspect(temp_engine)
        tables = inspector.get_table_names()
        
        # Verify expected tables exist
        assert "images" in tables
        assert "circular_objects" in tables
        assert len(tables) == 2
    
    def test_reflect_tables(self, temp_engine):
        """Test reflecting tables from the database."""
        # Create tables first
        Base.metadata.create_all(bind=temp_engine)
        
        # Create a new metadata object and reflect
        metadata = MetaData()
        metadata.reflect(bind=temp_engine)
        
        # Check reflected tables
        assert "images" in metadata.tables
        assert "circular_objects" in metadata.tables
        
        # Verify image table structure
        images_table = metadata.tables["images"]
        assert "id" in images_table.columns
        assert "path" in images_table.columns
        assert images_table.columns["id"].primary_key
        # For SQLite, String type length is stored directly as an integer
        assert images_table.columns["id"].type.length == 64  # SHA-256 length
        
        # Verify circular_objects table structure
        circular_objects_table = metadata.tables["circular_objects"]
        assert "id" in circular_objects_table.columns
        assert "image_id" in circular_objects_table.columns
        assert "bbox" in circular_objects_table.columns
        assert "centroid" in circular_objects_table.columns
        assert "radius" in circular_objects_table.columns
        assert circular_objects_table.columns["id"].primary_key
    
    def test_foreign_key_constraint(self, temp_engine):
        """Test that foreign key constraints are properly set up."""
        Base.metadata.create_all(bind=temp_engine)
        
        inspector = inspect(temp_engine)
        
        # Get foreign keys for circular_objects table
        foreign_keys = inspector.get_foreign_keys("circular_objects")
        
        # Should have one foreign key
        assert len(foreign_keys) == 1
        
        # Check foreign key details
        fk = foreign_keys[0]
        assert fk["constrained_columns"] == ["image_id"]
        assert fk["referred_table"] == "images"
        assert fk["referred_columns"] == ["id"]
    
    def test_indexes(self, temp_engine):
        """Test that indexes are created properly."""
        Base.metadata.create_all(bind=temp_engine)
        
        inspector = inspect(temp_engine)
        
        # Check indexes on circular_objects table
        indexes = inspector.get_indexes("circular_objects")
        
        # Should have at least one index on image_id
        image_id_indexes = [idx for idx in indexes if "image_id" in idx["column_names"]]
        assert len(image_id_indexes) >= 1


class TestDatabaseSession:
    """Test database session management."""
    
    def test_get_db_dependency(self, temp_engine, monkeypatch, mock_settings):
        """Test the get_db dependency function."""
        # Update mock settings with temp database URL
        mock_settings.database_url = str(temp_engine.url)
        
        # Monkeypatch get_settings to return our mock
        monkeypatch.setattr(config, "get_settings", lambda: mock_settings)
        
        # Create a mock SessionLocal
        mock_session_local = sessionmaker(bind=temp_engine)
        
        # Monkeypatch the SessionLocal in the database module
        monkeypatch.setattr(db_module, "SessionLocal", mock_session_local)
        
        # Create tables
        Base.metadata.create_all(bind=temp_engine)
        
        # Test the dependency
        db_gen = get_db()
        session = next(db_gen)
        
        try:
            # Verify it's a valid session
            assert isinstance(session, Session)
            
            # Test we can query with it
            images = session.query(Image).all()
            assert images == []
            
        finally:
            # Ensure cleanup happens
            try:
                next(db_gen)
            except StopIteration:
                pass
    
    def test_session_rollback_on_error(self, temp_engine, monkeypatch, mock_settings):
        """Test that sessions are properly rolled back on error."""
        # Update mock settings with temp database URL
        mock_settings.database_url = str(temp_engine.url)
        monkeypatch.setattr(config, "get_settings", lambda: mock_settings)
        mock_session_local = sessionmaker(bind=temp_engine)
        monkeypatch.setattr(db_module, "SessionLocal", mock_session_local)
        
        Base.metadata.create_all(bind=temp_engine)
        
        # Use the dependency
        db_gen = get_db()
        session = next(db_gen)
        
        # Add an image
        image = Image(id="a" * 64, path="test.jpg")
        session.add(image)
        
        # Don't commit, just close
        try:
            next(db_gen)
        except StopIteration:
            pass
        
        # Verify the image wasn't committed
        new_session = mock_session_local()
        assert new_session.query(Image).count() == 0
        new_session.close()


class TestDatabaseOperations:
    """Test database operations with the configured setup."""
    
    def test_init_db_creates_tables(self, temp_engine, monkeypatch, mock_settings):
        """Test that init_db creates all tables."""
        # Update mock settings with temp database URL
        mock_settings.database_url = str(temp_engine.url)
        
        # Monkeypatch get_settings
        monkeypatch.setattr(config, "get_settings", lambda: mock_settings)
        monkeypatch.setattr(db_module, "engine", temp_engine)
        monkeypatch.setattr(db_module, "Base", Base)
        
        # Verify no tables exist initially
        inspector = inspect(temp_engine)
        assert len(inspector.get_table_names()) == 0
        
        # Initialize database
        db_module.init_db()
        
        # Create fresh inspector to get updated state
        inspector = inspect(temp_engine)
        tables = inspector.get_table_names()
        assert "images" in tables
        assert "circular_objects" in tables
    
    def test_drop_db_removes_tables(self, temp_engine, monkeypatch, mock_settings):
        """Test that drop_db removes all tables."""
        # Update mock settings with temp database URL
        mock_settings.database_url = str(temp_engine.url)
        
        # Monkeypatch get_settings
        monkeypatch.setattr(config, "get_settings", lambda: mock_settings)
        monkeypatch.setattr(db_module, "engine", temp_engine)
        monkeypatch.setattr(db_module, "Base", Base)
        
        # Create tables first
        Base.metadata.create_all(bind=temp_engine)
        
        # Verify tables exist
        inspector = inspect(temp_engine)
        assert len(inspector.get_table_names()) > 0
        
        # Drop database
        db_module.drop_db()
        
        # Create fresh inspector to get updated state
        inspector = inspect(temp_engine)
        assert len(inspector.get_table_names()) == 0
    
    def test_crud_operations(self, temp_session):
        """Test basic CRUD operations with the database setup."""
        # Create tables
        Base.metadata.create_all(bind=temp_session.bind)
        
        # Create an image
        image = Image(
            id="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            path="data/test_image.jpg"
        )
        temp_session.add(image)
        temp_session.commit()
        
        # Create circular objects
        circle1 = CircularObject(
            image_id=image.id,
            bbox=[10, 20, 30, 40],
            centroid={"x": 20, "y": 30},
            radius=10.0
        )
        circle2 = CircularObject(
            image_id=image.id,
            bbox=[50, 60, 70, 80],
            centroid={"x": 60, "y": 70},
            radius=10.0
        )
        
        temp_session.add_all([circle1, circle2])
        temp_session.commit()
        
        # Query and verify
        retrieved_image = temp_session.query(Image).first()
        assert retrieved_image.id == image.id
        assert len(retrieved_image.circular_objects) == 2
        
        # Test relationship navigation
        circles = temp_session.query(CircularObject).all()
        assert len(circles) == 2
        assert all(c.image == retrieved_image for c in circles) 