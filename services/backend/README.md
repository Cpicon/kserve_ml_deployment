# Backend Service

A production-ready FastAPI backend that manages circular-object detection workflows on images. This service is intended to run locally (with SQLite) or as a container in Kubernetes, and integrates with a KServe-deployed PyTorch model for inference.

## Features

* **Image management** – upload images, store metadata, retrieve detections.
* **Model integration** – pluggable client to call external KServe model.
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
uv sync --dev  # install all dependencies in dev mode

# Run the dev server (option 1: using the startup script)
./start-dev.sh

# Run the dev server (option 2: manual)
uv run fastapi dev aiq_circular_detection
# ➜ visit http://localhost:8000/docs → Swagger UI

# Available endpoints:
# - http://localhost:8000/health - Health check
# - http://localhost:8000/config - Current configuration
# - http://localhost:8000/docs   - API documentation
```

## Running tests & lint

```bash
pytest -q          # run smoke tests
ruff check .       # lint/format check
```

## Docker

```bash
docker build -t backend-service:latest .
docker run -p 8000:80 backend-service:latest
```

---

Made with ❤️ 