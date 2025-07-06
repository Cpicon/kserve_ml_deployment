"""Application settings and configuration."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application metadata
    app_name: str = Field(
        default="AIQ Circular Detection Service",
        description="Application name"
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version"
    )
    environment: Literal["local", "dev", "stage", "prod"] = Field(
        default="local",
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # API Configuration
    api_prefix: str = Field(
        default="/api/v1",
        description="API route prefix"
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    # Storage Configuration (for image files)
    storage_type: Literal["local", "s3", "azure"] = Field(
        default="local",
        description="Storage backend type for image files"
    )
    storage_root: Path = Field(
        default=Path("data/images"),
        description="Root directory for local storage"
    )
    
    # Metadata Storage Configuration
    metadata_storage: Literal["memory", "database"] = Field(
        default="memory",
        description="Storage backend for image metadata (ID to path mapping)"
    )
    
    @field_validator("storage_root", mode="before")
    @classmethod
    def resolve_storage_path(cls, v: str | Path) -> Path:
        """Ensure storage root is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    # Database Configuration (for metadata)
    database_url: str = Field(
        default="sqlite:///data/db.sqlite3",
        description="Database connection URL"
    )
    database_echo: bool = Field(
        default=False,
        description="Echo SQL statements (for debugging)"
    )
    
    # Model Service Configuration
    model_service_url: Optional[str] = Field(
        default=None,
        description="URL of the KServe model service"
    )
    model_service_timeout: int = Field(
        default=30,
        description="Timeout for model service requests (seconds)"
    )
    model_service_retries: int = Field(
        default=3,
        description="Number of retries for model service requests"
    )
    
    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    log_json: bool = Field(
        default=False,
        description="Output logs in JSON format"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Path to log file (if None, logs to stdout only)"
    )
    
    # Performance Configuration
    max_upload_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file upload size in bytes"
    )
    worker_count: int = Field(
        default=1,
        description="Number of worker processes"
    )
    
    # Security Configuration
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for security features"
    )
    
    @property
    def log_level_numeric(self) -> int:
        """Get numeric log level."""
        return getattr(logging, self.log_level)
    
    def configure_logging(self) -> None:
        """Configure logging based on settings."""
        import sys
        
        # Base logging configuration
        handlers: list[logging.Handler] = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        handlers.append(console_handler)
        
        # File handler if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            handlers.append(file_handler)
        
        # Configure formatter
        if self.log_json:
            # JSON formatter for structured logging
            import json
            
            class JSONFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    log_obj = {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno,
                    }
                    if record.exc_info:
                        log_obj["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_obj)
            
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(self.log_format)
        
        # Apply formatter to all handlers
        for handler in handlers:
            handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=self.log_level_numeric,
            handlers=handlers,
            force=True,
        )
        
        # Set specific logger levels
        if self.debug:
            logging.getLogger("aiq_circular_detection").setLevel(logging.DEBUG)
        else:
            # Reduce noise from third-party libraries
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
    
    def get_storage_path(self, *paths: str) -> Path:
        """Get a path relative to storage root."""
        full_path = self.storage_root
        for path in paths:
            full_path = full_path / path
        return full_path


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings() 