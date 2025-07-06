# Backend Service

A production-ready FastAPI backend that manages circular-object detection workflows on images. This service is intended to run locally (with SQLite) or as a container in Kubernetes, and integrates with a KServe-deployed PyTorch model for inference.

## Features

* **Image management** – upload images, store metadata, retrieve detections.
* **Model integration** – pluggable client to call external KServe model with automatic circle detection on upload.
* **Dual operation modes** – "dummy" mode for development/testing, "real" mode for production inference.
* **REST API** – documented automatically via OpenAPI/Swagger (`/docs`).
* **CI/CD** – GitHub Actions workflow runs Ruff linting and tests on every push.
* **Container-ready** – lightweight image based on Python 3.12-slim.
* **Configuration management** – Pydantic-based settings with environment variable support.
* **Production logging** – Configurable logging with JSON output.

## Project layout

```
services/backend/
├── aiq_circular_detection/            # Application code
│   ├── __init__.py     # Package entrypoint
│   ├── main.py         # FastAPI app with endpoints
│   ├── model_client.py # ML model client for inference
│   ├── repositories/   # Data repositories (image, objects)
│   ├── schemas/        # Pydantic models for API
│   └── storage/        # Storage abstraction layer
├── config/             # Configuration module
│   ├── settings.py     # Pydantic settings
│   └── README.md       # Config documentation
├── tests/              # Pytest test suite
├── data/               # (Ignored) local files & SQLite DB
├── pyproject.toml      # Dependencies & tool config
├── start-dev.sh        # Development startup script
└── README.md           # You are here
```

## Quick start (local)
### Requirements
* Python 3.12 or higher
* `uv` package manager (install via `pip install uv` or follow [installation instructions](https://astral.sh/uv/))
```bash
# Create venv and install deps (requires Python ≥3.12)
uv venv

# Use `uv` for fast installs
uv sync --active --dev  # install all dependencies in dev mode

# Run the dev server (option 1: using the startup script)
./start-dev.sh

# Available endpoints:
# ➜ visit http://localhost:8000/docs → Swagger UI
# - http://localhost:8000/health - Health check
# - http://localhost:8000/config - Current configuration
# - http://localhost:8000/docs   - API documentation
```

## Configuration

The service supports the following key environment variables:

```bash
# Model client configuration
MODE=dummy                            # Use "real" for actual model inference
MODEL_SERVER_URL=http://localhost:8080 # Required for real mode
MODEL_NAME=circular-detector          # Model name for inference endpoint

# Storage configuration
METADATA_STORAGE=memory              # Use "database" for persistence
DATABASE_URL=sqlite:///data/db.sqlite3

# See env.example for all available options
```

## Model Integration

When an image is uploaded, the service automatically:
1. Stores the image and metadata
2. Calls the model client for circle detection
3. Saves detected circles to the database
4. Returns both image ID and detection results
In **dummy mode** (default), it returns fixed test data. In **real mode**, it calls the configured model server.

## Running tests & lint

```bash
pytest -q          # run smoke tests
ruff check .       # lint/format check
```

## Docker

### Building the Image

```bash
# Build the Docker image
docker build -t aiq-circular-detection .
```

### Running with Docker

```bash
# Run with default settings (dummy mode)
docker run -d --name aiq-api -p 8000:8000 aiq-circular-detection

# Run with environment variables
docker run -d --name aiq-api \
  -p 8000:8000 \
  -e MODE=dummy \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/data:/data \
  -v $(pwd)/logs:/logs \
  aiq-circular-detection

# Check logs
docker logs -f aiq-api

# Stop and remove
docker stop aiq-api && docker rm aiq-api
```

### Running with Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down

# Rebuild and start (after code changes)
docker-compose up -d --build
```

The service will be available at:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Volumes

The Docker setup uses two volumes for persistence:
- `./data`: SQLite database and uploaded images
- `./logs`: Application logs

These directories are created automatically and persist data between container restarts.

---

Made with ❤️ 