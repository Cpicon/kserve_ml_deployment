# KServe ML Deployment Project

This repository provides a **batteries-included template** for serving machine-learning models on Kubernetes with the following stack:

* **Istio** for service mesh & ingress
* **Knative Serving** for serverless autoscaling
* **KServe** for model management & prediction endpoints

The goal is to give you an opinionated, yet extensible starting point that you can run locally on Kind _or_ promote to any managed Kubernetes service.

## Architecture Overview

The following diagram illustrates the complete request flow when a user uploads an image for circular object detection:

```mermaid
flowchart TB
    User[User/Client] 
    
    subgraph K8s["Kubernetes Cluster"]
        Gateway[Istio Gateway :80/:443]
        
        subgraph NS1["Namespace: aiq-backend"]
            Service[Backend Service]
            
            subgraph BackendPod["Backend Pod"]
                Envoy1[Envoy Proxy]
                FastAPI[FastAPI :8000]
            end
            
            Storage[(PVC Storage)]
            DB[(SQLite DB)]
        end
        
        subgraph NS2["Namespace: kserve-test"]
            ModelService[Model Service]
            
            subgraph ModelDeployment["KServe Deployment"]
                Knative[Knative Autoscaler]
                
                subgraph ModelPod["Model Pod"]
                    Envoy2[Envoy Proxy]
                    Model[Model Server :8080]
                end
            end
        end
        
        Gateway --> Service
        Service --> Envoy1
        Envoy1 --> FastAPI
        FastAPI --> Storage
        FastAPI --> DB
        FastAPI --> ModelService
        ModelService --> Knative
        Knative --> ModelPod
        Envoy2 --> Model
    end
    
    User --> Gateway
    Gateway --> User
    
    style Gateway fill:#4285F4,color:#fff
    style FastAPI fill:#34A853,color:#fff
    style Model fill:#FBBC04,color:#000
    style Knative fill:#EA4335,color:#fff
```

### Request Flow Steps

1. **User → Istio Gateway**: Client sends POST request to `/images/` endpoint
2. **Gateway → Backend Service**: VirtualService routes request to backend service
3. **Service → Pod**: Kubernetes service load balances to backend pod
4. **Envoy → FastAPI**: Sidecar proxy forwards request with mTLS
5. **FastAPI Processing**:
   - Stores image in PVC-backed storage
   - Saves metadata to SQLite database
   - Prepares inference request
6. **FastAPI → Model Service**: Sends base64 image for inference
7. **Knative Autoscaling**: Scales model pod from 0 to 1 if needed
8. **Model Inference**: Detects circles and returns bounding boxes
9. **Response Flow**: Results flow back through the same path
10. **User Response**: Client receives JSON with detected objects

### Detailed Request Flow

```mermaid
sequenceDiagram
    box External
        participant U as User
    end
    
    box Kubernetes Cluster
        participant IG as Istio Gateway
        participant VS as VirtualService
        participant EP as Envoy Proxy
        participant API as FastAPI Backend
        participant S as Storage PVC
        participant DB as Database
        participant KS as KServe
        participant M as Model Server
    end
    
    U->>IG: POST /images/ with image file
    IG->>VS: Route based on path
    VS->>EP: Forward to backend service
    EP->>API: mTLS secured request
    API->>S: Store image file
    API->>DB: Save image metadata
    API->>KS: Request inference with base64 image
    Note over KS: Knative scales from 0 to 1
    KS->>M: Forward to model pod
    M->>M: Detect circles
    M->>KS: Return detections
    KS->>API: JSON response
    API->>DB: Save detected objects
    API->>EP: Response with detections
    EP->>VS: Return via mesh
    VS->>IG: Route response
    IG->>U: JSON result
```

### Workflow Description

1. **User Upload**: Client sends a POST request to `/api/v1/images/` with an image file
2. **Istio Ingress**: Request enters through Istio Gateway and is routed by VirtualService rules
3. **Service Mesh**: Envoy sidecar proxy handles mTLS, observability, and load balancing
4. **FastAPI Backend**: 
   - Receives and validates the image
   - Stores image in persistent storage (PVC-backed filesystem)
   - Registers metadata in database (image ID, path, timestamp)
5. **Model Inference Request**: Backend prepares inference request with base64-encoded image
6. **KServe/Knative Autoscaling**: 
   - Knative autoscaler receives request
   - Scales model pod from 0 to 1 (cold start) or routes to existing pod
   - Creates revision-specific pods on demand
7. **Model Processing**: 
   - Model server receives image
   - PyTorch/TensorFlow model detects circular objects
   - Returns bounding boxes, centroids, and confidence scores
8. **Response Processing**:
   - Backend receives detection results
   - Saves detected objects to database
   - Returns comprehensive response to user

### Key Components

- **Istio Gateway**: Entry point for all external traffic, handles TLS termination
- **VirtualService**: Defines routing rules for different endpoints
- **Envoy Proxy**: Sidecar container providing service mesh capabilities
- **Knative Serving**: Provides serverless scaling, including scale-to-zero
- **KServe**: Manages model deployment, versioning, and inference endpoints
- **Persistent Storage**: Ensures data survives pod restarts (critical for SQLite)

---
## Project Structure

```
.
├── environments/           # Environment-specific Kubernetes manifests
│   ├── dev/                # Development environment configs
│   ├── stage/              # Staging environment configs
│   ├── prod/               # Production configs (Helm values, manifests)
│   └── local/              # Local Kind cluster setup
│       ├── backend/        # Backend service K8s manifests
│       │   ├── *.yaml      # Deployments, services, storage, Istio routing
│       │   ├── deploy.sh   # Automated deployment script
│       │   └── test-istio.sh # Test script with Istio integration
│       └── test/           # Demo model and test payloads
├── model/                  # Pretrained model artifacts
├── services/               # Microservices
│   └── backend/            # FastAPI backend service
│       ├── aiq_circular_detection/  # Main application package
│       ├── config/         # Configuration management
│       ├── tests/          # Unit and integration tests
│       └── Dockerfile      # Optimized multi-stage build
└── scripts/                # Helper scripts
```

## Getting Started

### Prerequisites

- Docker Desktop with Kubernetes enabled
- `kind` - Kubernetes in Docker
- `kubectl` - Kubernetes CLI
- `helm` - Package manager for Kubernetes
- `kustomize` - Tool for customizing Kubernetes YAML configurations
- `jq` - JSON processor
- `pack` - Pack is a CLI tool maintained by the CNB project to support the use of buildpacks (https://buildpacks.io/docs/for-platform-operators/how-to/integrate-ci/pack/)

### Quick Start

1. **Create Kind cluster with Istio, Knative, and KServe**:
   ```bash
   # Create cluster
   bash environments/local/setup_kind.sh
   
   # Install the full stack
   bash environments/local/install_kserve_knative.sh
   ```

2. **Deploy the Backend Service**:
   ```bash
   cd environments/local/backend
   ./deploy.sh
   ```

3. **Test the deployment**:
   ```bash
   # Test through Istio ingress
   ./test-istio.sh
   
   # Access API documentation
   # Visit http://localhost:8080/docs while test-istio.sh is running
   ```

4. **Deploy a sample ML model** (optional):
   ```bash
   kubectl create namespace kserve-test
   kubectl apply -n kserve-test -f environments/local/test/sklearn-iris.yaml
   ```

### Local Development Testing (No Kubernetes/Docker)

For rapid development and testing without Kubernetes or Docker, you can run all services locally on your machine.

#### Prerequisites for Local Testing

- Python 3.9+ installed
- `uv` - Fast Python package manager ([installation guide](https://github.com/astral-sh/uv))
- `jq` - JSON processor
- Available ports: 8000 (backend) and 9090 (model server)

#### One-Command Local Testing

Run all services and tests with a single command:

```bash
./run_all_local.sh
```

This script will:
1. Start the AI model server on port 9090
2. Start the backend API service on port 8000
3. Run integration tests automatically
4. Keep services running for manual testing

The script handles:
- Prerequisite checking
- Port availability verification
- Service health monitoring
- Automatic cleanup on exit
- Detailed logging to `logs/` directory

#### Manual Local Testing

If you prefer to run services individually:

1. **Start the Model Server**:
   ```bash
   cd environments/local/aiq_detector
   ./run_local.sh
   # Model server will run on http://localhost:9090
   # Swagger UI: http://localhost:9090/docs
   ```

2. **Start the Backend Service** (in a new terminal):
   ```bash
   cd services/backend
   ./start-dev-real.sh
   # Backend API will run on http://localhost:8000
   # API Docs: http://localhost:8000/docs
   ```

3. **Run Integration Tests** (in a new terminal):
   ```bash
   cd services/backend
   ./test_full_integration.sh
   ```

#### Troubleshooting Local Development

- **Port conflicts**: Ensure ports 8000 and 9090 are free
- **Model download**: First run downloads the AI model (~300MB)
- **Logs**: Check `logs/model_server.log` and `logs/backend_server.log`
- **Cleanup**: Use `Ctrl+C` to stop all services gracefully

### API Endpoints

Once deployed, the backend service provides:

- `POST /images/` - Upload image for circle detection
- `GET /images/{image_id}/objects` - List detected objects for an image
- `GET /images/{image_id}/objects/{object_id}` - Get object details
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)

### Configuration

The backend service can operate in two modes:

1. **Dummy Mode** (default): Returns mock detection results for testing
2. **Real Mode**: Connects to actual KServe model endpoint

Configure via environment variables:
```yaml
MODE: "dummy"  # or "real"
MODEL_SERVER_URL: "http://model-service.namespace.svc.cluster.local"
```

## Production Deployment

For production environments:

1. **Replace SQLite** with PostgreSQL or MySQL for multi-replica support
2. **Use cloud storage** (S3, GCS, Azure Storage) instead of local filesystem
3. **Configure proper ingress** with TLS certificates and domain names
4. **Set resource limits** and autoscaling policies
5. **Enable monitoring** with Prometheus and distributed tracing

See `environments/local/backend/DEPLOYMENT.md` for detailed deployment instructions.

## Development

### Running Tests
```bash
cd services/backend
pytest
```

### Building Docker Image
```bash
cd services/backend
docker build -t aiq-circular-detection:latest .
```

### Local Development
```bash
cd services/backend
./start-dev.sh
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure the deployment works
5. Submit a pull request

For detailed documentation on the local environment setup, see `environments/local/README.md`.