"""Tests for configuration settings."""
import logging
from pathlib import Path

import pytest

from config import Settings, get_settings


class TestSettings:
    """Test Settings configuration."""
    
    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        
        # Application metadata
        assert settings.app_name == "AIQ Circular Detection Service"
        assert settings.app_version == "0.1.0"
        assert settings.environment == "local"
        assert settings.debug is False
        
        # API Configuration
        assert settings.api_prefix == "/api/v1"
        assert settings.cors_origins == ["*"]
        
        # Storage Configuration
        assert settings.storage_type == "local"
        assert settings.storage_root == Path("data/images")
        
        # Database Configuration
        assert settings.database_url == "sqlite:///data/db.sqlite3"
        assert settings.database_echo is False
        
        # Model Service Configuration
        assert settings.model_service_url is None
        assert settings.model_service_timeout == 30
        assert settings.model_service_retries == 3
        
        # Logging Configuration
        assert settings.log_level == "INFO"
        assert settings.log_json is False
        assert settings.log_file is None
        
        # Performance Configuration
        assert settings.max_upload_size == 10 * 1024 * 1024
        assert settings.worker_count == 1
    
    def test_env_override(self, monkeypatch):
        """Test environment variable overrides."""
        # Set environment variables
        monkeypatch.setenv("APP_NAME", "Test App")
        monkeypatch.setenv("ENVIRONMENT", "dev")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("STORAGE_ROOT", "/custom/storage")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_JSON", "true")
        monkeypatch.setenv("MAX_UPLOAD_SIZE", "5242880")  # 5MB
        
        settings = Settings()
        
        assert settings.app_name == "Test App"
        assert settings.environment == "dev"
        assert settings.debug is True
        assert settings.storage_root == Path("/custom/storage")
        assert settings.log_level == "DEBUG"
        assert settings.log_json is True
        assert settings.max_upload_size == 5242880
    
    def test_storage_root_validator(self):
        """Test storage root path validator."""
        # String input
        settings = Settings(storage_root="custom/path")
        assert isinstance(settings.storage_root, Path)
        assert settings.storage_root == Path("custom/path")
        
        # Path input
        settings = Settings(storage_root=Path("/absolute/path"))
        assert isinstance(settings.storage_root, Path)
        assert settings.storage_root == Path("/absolute/path")
    
    def test_log_level_numeric(self):
        """Test numeric log level property."""
        settings = Settings(log_level="DEBUG")
        assert settings.log_level_numeric == logging.DEBUG
        
        settings = Settings(log_level="INFO")
        assert settings.log_level_numeric == logging.INFO
        
        settings = Settings(log_level="ERROR")
        assert settings.log_level_numeric == logging.ERROR
    
    def test_get_storage_path(self):
        """Test get_storage_path method."""
        settings = Settings(storage_root="data/images")
        
        # Single path component
        path = settings.get_storage_path("subfolder")
        assert path == Path("data/images/subfolder")
        
        # Multiple path components
        path = settings.get_storage_path("year", "2024", "january")
        assert path == Path("data/images/year/2024/january")
        
        # No components
        path = settings.get_storage_path()
        assert path == Path("data/images")
    
    def test_configure_logging_console(self):
        """Test logging configuration with console output."""
        settings = Settings(log_level="INFO", log_json=False)
        settings.configure_logging()
        
        # Verify logging is configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0
        
        # Check that at least one handler is a StreamHandler
        stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) > 0
        
        # Check formatter is not JSON
        formatter = stream_handlers[0].formatter
        assert formatter is not None
        assert formatter._fmt == settings.log_format
    
    def test_configure_logging_json(self, capsys):
        """Test logging configuration with JSON output."""
        settings = Settings(log_level="INFO", log_json=True)
        settings.configure_logging()
        
        # Test logging works
        logger = logging.getLogger("test_logger")
        logger.info("Test JSON message")
        
        # Capture stdout since JSON formatter outputs there
        captured = capsys.readouterr()
        
        # Check that output contains JSON structure
        assert '"message": "Test JSON message"' in captured.out
        assert '"level": "INFO"' in captured.out
        assert '"logger": "test_logger"' in captured.out
    
    def test_configure_logging_file(self, tmp_path):
        """Test logging configuration with file output."""
        log_file = tmp_path / "test.log"
        settings = Settings(log_level="INFO", log_file=log_file)
        settings.configure_logging()
        
        # Test logging works
        logger = logging.getLogger("test_logger")
        logger.info("Test file message")
        
        # Check log file exists and contains message
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Test file message" in log_content
    
    def test_configure_logging_debug_mode(self):
        """Test logging configuration in debug mode."""
        settings = Settings(debug=True)
        settings.configure_logging()
        
        # Check that debug mode sets appropriate log levels
        aiq_logger = logging.getLogger("aiq_circular_detection")
        assert aiq_logger.level == logging.DEBUG
    
    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        # Clear cache first
        get_settings.cache_clear()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2  # Same instance
    
    @pytest.mark.parametrize("env", ["local", "dev", "stage", "prod"])
    def test_valid_environments(self, env):
        """Test valid environment values."""
        settings = Settings(environment=env)
        assert settings.environment == env
    
    def test_invalid_environment(self):
        """Test invalid environment value raises error."""
        with pytest.raises(ValueError):
            Settings(environment="invalid")
    
    @pytest.mark.parametrize("storage_type", ["local", "s3", "azure"])
    def test_valid_storage_types(self, storage_type):
        """Test valid storage type values."""
        settings = Settings(storage_type=storage_type)
        assert settings.storage_type == storage_type
    
    def test_env_file_loading(self, tmp_path, monkeypatch):
        """Test loading settings from .env file."""
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("""
APP_NAME=Env File App
ENVIRONMENT=dev
STORAGE_ROOT=/env/file/storage
LOG_LEVEL=WARNING
""")
        
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        settings = Settings()
        
        assert settings.app_name == "Env File App"
        assert settings.environment == "dev"
        assert settings.storage_root == Path("/env/file/storage")
        assert settings.log_level == "WARNING" 