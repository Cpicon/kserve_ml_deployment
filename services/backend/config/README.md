# Configuration Module

A comprehensive Pydantic-based configuration system for the AIQ Circular Detection service.

## Features

- **Environment-based Configuration**: Load settings from environment variables or `.env` files
- **Type Safety**: Full type validation with Pydantic
- **Comprehensive Logging Setup**: Configurable logging with console/file output and JSON formatting
- **Flexible Storage Configuration**: Support for different storage backends
- **Performance Tuning**: Configurable limits and worker settings
- **Environment-specific Settings**: Different configurations for local/dev/stage/prod

## Configuration Options

### Application Settings
- `APP_NAME`: Application name (default: "AIQ Circular Detection Service")
- `APP_VERSION`: Application version (default: "0.1.0")
- `ENVIRONMENT`: Deployment environment - local/dev/stage/prod (default: "local")
- `DEBUG`: Enable debug mode (default: false)

### API Configuration
- `API_PREFIX`: API route prefix (default: "/api/v1")
- `CORS_ORIGINS`: Comma-separated allowed CORS origins (default: "*")

### Storage Configuration
- `STORAGE_TYPE`: Storage backend type - local/s3/azure (default: "local")
- `STORAGE_ROOT`: Root directory for local storage (default: "data/images")

### Database Configuration
- `DATABASE_URL`: Database connection URL (default: "sqlite:///data/db.sqlite3")
- `DATABASE_ECHO`: Echo SQL statements for debugging (default: false)

### Model Service Configuration
- `MODEL_SERVICE_URL`: URL of the KServe model service
- `MODEL_SERVICE_TIMEOUT`: Timeout for model requests in seconds (default: 30)
- `MODEL_SERVICE_RETRIES`: Number of retries for model requests (default: 3)

### Logging Configuration
- `LOG_LEVEL`: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (default: "INFO")
- `LOG_FORMAT`: Log message format (default: "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
- `LOG_JSON`: Output logs in JSON format (default: false)
- `LOG_FILE`: Path to log file (optional)

### Performance Configuration
- `MAX_UPLOAD_SIZE`: Maximum file upload size in bytes (default: 10MB)
- `WORKER_COUNT`: Number of worker processes (default: 1)

### Security Configuration
- `SECRET_KEY`: Secret key for security features (default: "change-me-in-production")

## Usage

### Basic Usage

```python
from config import get_settings

# Get cached settings instance
settings = get_settings()

# Configure logging
settings.configure_logging()

# Access configuration values
print(f"App: {settings.app_name}")
print(f"Storage root: {settings.storage_root}")
```

### Environment Variables

Set configuration via environment variables:

```bash
export ENVIRONMENT=dev
export LOG_LEVEL=DEBUG
export STORAGE_ROOT=/custom/storage
export LOG_JSON=true
```

### Using .env File

Create a `.env` file in the project root:

```env
APP_NAME=My Custom App
ENVIRONMENT=dev
DEBUG=true
STORAGE_ROOT=/data/custom
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://user:pass@localhost/mydb
```

### Logging Configuration

The settings provide a `configure_logging()` method that sets up Python logging:

```python
settings = get_settings()
settings.configure_logging()

# Now use logging anywhere
import logging
logger = logging.getLogger(__name__)
logger.info("Application started")
```

Features:
- Console output (always enabled)
- Optional file output
- JSON formatting option for structured logging
- Debug mode automatically sets debug level for app loggers
- Reduces noise from third-party libraries

### Storage Path Helper

Use the `get_storage_path()` helper to build paths:

```python
settings = get_settings()

# Get paths relative to storage root
archive_path = settings.get_storage_path("archive", "2024", "images")
# Returns: Path("data/images/archive/2024/images")
```

## Testing

The module includes comprehensive tests covering:
- Default values
- Environment variable overrides
- Validation
- Logging configuration
- Path helpers

Run tests:
```bash
uv run pytest tests/test_config.py -v
```

## Best Practices

1. **Production Settings**: Always override sensitive defaults in production:
   - `SECRET_KEY`
   - `DATABASE_URL`
   - `CORS_ORIGINS`

2. **Logging**: 
   - Use JSON logging in production for better log aggregation
   - Configure appropriate log levels per environment
   - Use file logging for persistent logs

3. **Environment Separation**:
   - Use different `.env` files for different environments
   - Keep production secrets in secure secret management systems

4. **Performance Tuning**:
   - Adjust `WORKER_COUNT` based on available CPU cores
   - Set appropriate `MAX_UPLOAD_SIZE` for your use case
   - Configure `MODEL_SERVICE_TIMEOUT` based on model complexity 