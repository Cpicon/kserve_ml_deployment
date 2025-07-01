"""Integration tests for database configuration with environment variables."""

import os
import tempfile
from pathlib import Path

from config import get_settings


def test_database_url_from_environment():
    """Test that DATABASE_URL environment variable overrides default."""
    # Save original env var if it exists
    original = os.environ.get("DATABASE_URL")
    
    try:
        # Set a custom database URL
        test_db_url = "postgresql://testuser:testpass@testhost:5432/testdb"
        os.environ["DATABASE_URL"] = test_db_url
        
        # Get fresh settings (clear cache)
        get_settings.cache_clear()
        settings = get_settings()
        
        assert settings.database_url == test_db_url
        
    finally:
        # Restore original env var
        if original is not None:
            os.environ["DATABASE_URL"] = original
        else:
            os.environ.pop("DATABASE_URL", None)
        
        # Clear cache again for other tests
        get_settings.cache_clear()


def test_default_database_url():
    """Test default database URL when no environment variable is set."""
    # Save original env var if it exists
    original = os.environ.get("DATABASE_URL")
    
    try:
        # Remove DATABASE_URL if it exists
        os.environ.pop("DATABASE_URL", None)
        
        # Get fresh settings
        get_settings.cache_clear()
        settings = get_settings()
        
        assert settings.database_url == "sqlite:///data/db.sqlite3"
        
    finally:
        # Restore original env var
        if original is not None:
            os.environ["DATABASE_URL"] = original
        
        # Clear cache again
        get_settings.cache_clear()


def test_sqlite_data_directory_creation():
    """Test that SQLite data directory is created automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set DATABASE_URL to a non-existent directory
        db_path = Path(tmpdir) / "subdir" / "test.db"
        db_url = f"sqlite:///{db_path}"
        
        original = os.environ.get("DATABASE_URL")
        try:
            os.environ["DATABASE_URL"] = db_url
            get_settings.cache_clear()
            
            # Import database module which should create the directory
            import importlib

            import aiq_circular_detection.db.database as db_module
            importlib.reload(db_module)
            
            # Check that parent directory was created
            assert db_path.parent.exists()
            
        finally:
            if original is not None:
                os.environ["DATABASE_URL"] = original
            else:
                os.environ.pop("DATABASE_URL", None)
            get_settings.cache_clear() 