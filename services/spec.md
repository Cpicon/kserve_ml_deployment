# Backend Service
This project is a Python-based, production-grade FastAPI service that manages circular-object detection workflows on images, structured to showcase scalable ML infrastructure skills. Its core responsibilities are:
	•	Image Management:
	•	Accepts image uploads (POST /images/), stores them on a configurable persistent layer (filesystem/SQLite), and assigns each upload a unique UUID.
	•	Maintains metadata about each upload and its detected objects in a SQL database via SQLAlchemy and Pydantic schemas.
	•	Object Detection Integration:
	•	Delegates inference to an external KServe-deployed PyTorch model over HTTP, using a pluggable ModelClient abstraction (with both dummy and real clients).
	•	On upload, invokes the model to detect circular objects (bounding box, centroid, radius) and persists those results.
	•	RESTful Endpoints:
	•	List Objects (GET /images/{image_id}/objects): returns all detected object IDs and bounding boxes.
	•	Object Detail (GET /images/{image_id}/objects/{object_id}): returns full geometric data (bbox, centroid, radius) for a single object.
	•	Architecture & Best Practices:
	•	Follows SOLID design with clear separation between storage, database repositories, model client, and API layers.
	•	Fully test-driven: each module has pytest coverage, including error paths and logging.
	•	Containerized with a non-root Alpine-based Docker image.
	•	Kubernetes Deployment:
	•	Provides Helm/K8s manifests (for Kind/Knative/KServe) to demonstrate autoscaling (scale-to-zero), resource optimization, and in-cluster HTTP routing.
	•	Integrates Prometheus and Grafana dashboards for live monitoring of latency, throughput, and resource usage.

This blueprint ensures that the LLM can generate each component—from scaffolding and CI, through storage, API, model integration, testing, containerization, and Kubernetes manifests—in small, composable, testable prompts.
High-Level Blueprint
	1.	Project CI/CD
– Add CI configuration (pytest, linters)
	2.	Storage & Persistence Layer
– Define where and how images and metadata live (e.g. filesystem + SQLite)
– Abstract into a Python “storage” module
	3.	Image Upload Endpoint
– Implement POST /images/ to accept and store raw images
– Generate a unique image ID
	4.	Metadata & Object Registry
– Define data models (Pydantic schemas + ORM models) for “Image” and “CircularObject”
– On upload, register zero or more circular objects (placeholder)
	5.	List Objects Endpoint
– Implement GET /images/{image_id}/objects
– Return list of { object_id, bbox }
	6.	Object Detail Endpoint
– Implement GET /images/{image_id}/objects/{object_id}
– Return { bbox, centroid, radius }
	7.	Model-Evaluation Strategy
– Create a “model client” abstraction that calls the KServe model (using the provided helper script pattern)
– Write integration tests with dummy response
	8.	Containerization & Deployment
– Write a Dockerfile for the FastAPI app
– Add Kubernetes manifests (ConfigMap, Deployment, Service) under environments/

⸻
Fine-Grained Steps:

Chunk 1: Scaffold & CI
	2.	Create src/main.py with an empty FastAPI instance.
	3.	Add pyproject.toml and include FastAPI, Uvicorn, SQLAlchemy, Pydantic, pytest, ruff.
	4.	Write a basic smoke test: import the FastAPI app, assert it starts.
	5.	Configure pytest in pyproject.toml; add a GitHub Actions workflow that runs linters and tests on push.

Chunk 2: Storage Module
	1.	Under src/storage/, define an interface StorageClient with methods save_image(bytes) -> image_path and read_image(path) -> bytes.
	2.	Provide a default filesystem implementation LocalStorageClient that writes to services/backend/data/images/.
	3.	Write unit tests for LocalStorageClient using tmp paths (pytest tmp_path).
	4.	Add configuration support (via Pydantic settings) for storage root.

Chunk 3: Upload API
	1.	Define Pydantic schema ImageUploadResponse with image_id: UUID.
	2.	Implement POST /images/ endpoint in main.py to:
	•	Receive UploadFile via FastAPI
	•	Call storage_client.save_image()
	•	Generate a new image_id (UUID)
	•	Persist Image record (for now, in-memory store or stub)
	•	Return ImageUploadResponse
	3.	Write tests for POST /images/ using FastAPI’s TestClient, asserting 200 OK and valid image_id.
	4.	Integrate logging: log receipt of upload with image size and generated ID.

Chunk 4: Data Models & DB
	1.	Under src/models/, define SQLAlchemy models for Image and CircularObject (with columns: id, image_id, bbox, centroid, radius).
	2.	Create a database.py that sets up an SQLite engine (file data/db.sqlite3) and session factory.
	3.	Write migrations or a simple create_all() call on startup.
	4.	Write unit tests to insert and query these models via a test database in tmp_path.

Chunk 5: List-Objects API
	1.	Define Pydantic schema ObjectSummary with object_id: UUID and bbox: List[int].
	2.	In main.py, add GET /images/{image_id}/objects:
	•	Query DB for all CircularObject rows matching image_id.
	•	Return list of ObjectSummary.
	3.	Write tests:
	•	Seed the test DB with known objects
	•	Call endpoint and verify JSON matches expected summaries.

Chunk 6: Object-Detail API
	1.	Define Pydantic schema ObjectDetail with bbox: List[int], centroid: Tuple[float, float], radius: float.
	2.	Add GET /images/{image_id}/objects/{object_id} to:
	•	Fetch the specific CircularObject.
	•	Return ObjectDetail.
	3.	Write tests covering: object exists, object not found (404), wrong image_id (404).

Chunk 7: Model Client & Eval Tests
	1.	Under src/model_client.py, define ModelClient with method detect_circles(image_path) -> List[Circle].
	2.	Use HTTP calls (e.g., httpx) to model server’s /v1/models/...:predict endpoint, following the helper script pattern in environments/local/test/test_inference.sh.
	3.	Write a dummy implementation that returns a fixed list of circles, and corresponding integration tests mocking HTTP.
	4.	Refactor upload flow: after saving image and creating DB record, call ModelClient.detect_circles(), then insert CircularObject rows into DB.
	5.	Write end-to-end tests: upload image (using sample JPEG/COCO JSON), then GET objects returns the dummy circles.
	6.	Add logging around model calls (request time, status code).

Chunk 8: Docker + K8s Manifests
	1.	Write a Dockerfile in services/backend/ that:
	•	Uses python:3.12-slim, installs dependencies, copies code, and sets CMD ["uvicorn","main:app","--host","0.0.0.0","--port","80"].
	2.	Add a docker-compose.yml (optional) to run the app + SQLite mount for local dev.
	3.	Create K8s YAMLs under environments/local/ (or dev/):
	•	deployment.yaml with the container image, env vars for storage path and DB URL.
	•	service.yaml exposing port 80.
	•	(Optional) ingress.yaml or instructions for port-forward.
	4.	Write a smoke test script that builds the Docker image, deploys to Kind (kubectl apply -f), port-forwards, and runs the existing test_inference.sh (modified for FastAPI) to assert endpoints work in-cluster.