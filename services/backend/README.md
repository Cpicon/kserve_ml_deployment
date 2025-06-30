# Backend Service

A production-ready FastAPI backend that manages circular-object detection workflows on images. This service is intended to run locally (with SQLite) or as a container in Kubernetes, and integrates with a KServe-deployed PyTorch model for inference.

## Features

* **Image management** – upload images, store metadata, retrieve detections.
* **Model integration** – pluggable client to call external KServe model.
* **REST API** – documented automatically via OpenAPI/Swagger (`/docs`).
* **CI/CD** – GitHub Actions workflow runs Ruff linting and tests on every push.
* **Container-ready** – lightweight image based on Python 3.12-slim.

## Project layout

```
services/backend/
├── aiq_circular_detection/            # Application code
│   ├── __init__.py # entrypoint for the package
│   └── main.py     # FastAPI app w/ health endpoint
├── tests/          # Pytest test suite
├── data/           # (Ignored) local files & SQLite DB
├── config/         # Future configuration files
├── pyproject.toml  # Dependencies & tool config
└── README.md       # You are here
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

# Run the dev server
uv run --active fastapi dev aiq_circular_detection
# ➜ visit http://localhost:8000/docs → Swagger UI
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