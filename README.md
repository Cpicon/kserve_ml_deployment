# KServe ML Deployment Project

This repository provides a **batteries-included template** for serving machine-learning models on Kubernetes with the following stack:

* **Istio** for service mesh & ingress
* **Knative Serving** for serverless autoscaling
* **KServe** for model management & prediction endpoints

The goal is to give you an opinionated, yet extensible starting point that you can run locally on Kind _or_ promote to any managed Kubernetes service.

## Architecture Overview

The following diagram illustrates the complete request flow when a user uploads an image for circular object detection:

```mermaid
graph TB
    subgraph "External"
        User[User/Client]
    end
    
    subgraph "Kubernetes Cluster"
        subgraph "Istio Ingress Layer"
            IG[Istio Gateway<br/>:80/:443]
            VS[VirtualService<br/>Routes: /api/v1/*, /health, /docs]
        end
        
        subgraph "Backend Service Pod"
            subgraph "Istio Service Mesh"
                EP1[Envoy Proxy<br/>Sidecar]
            end
            subgraph "Application Container"
                FA[FastAPI Backend<br/>:8000]
                API[POST /images/<br/>Endpoint]
            end
        end
        
        subgraph "Storage & Database"
            PV[PersistentVolume<br/>hostPath/EBS/GCE]
            DB[(SQLite/PostgreSQL<br/>Metadata & Objects)]
            FS[File Storage<br/>Images]
        end
        
        subgraph "KServe Model Service"
            subgraph "Knative Serving"
                KPA[Knative Autoscaler<br/>Scale 0→N]
                REV[Revision<br/>sklearn-iris-predictor]
            end
            
            subgraph "Model Pod (On-Demand)"
                subgraph "Istio Mesh"
                    EP2[Envoy Proxy]
                end
                subgraph "Model Container"
                    MS[Model Server<br/>:8080]
                    ML[PyTorch/TF Model<br/>Circle Detection]
                end
            end
        end
    end
    
    %% Request Flow
    User -->|1. POST /images/<br/>multipart/form-data| IG
    IG -->|2. Route via<br/>HTTP rules| VS
    VS -->|3. Forward to<br/>backend service| EP1
    EP1 -->|4. mTLS| FA
    FA --> API
    
    %% Backend Processing
    API -->|5. Store image| FS
    API -->|6. Save metadata| DB
    API -->|7. Request inference<br/>base64 image| EP1
    
    %% Model Inference Flow
    EP1 -->|8. Service discovery<br/>via Istio| KPA
    KPA -->|9. Scale from 0<br/>if needed| REV
    REV -->|10. Create pod| EP2
    EP2 -->|11. Forward| MS
    MS --> ML
    ML -->|12. Detect circles<br/>bbox + scores| MS
    
    %% Response Flow
    MS -->|13. JSON response| EP2
    EP2 -->|14. Return via mesh| EP1
    EP1 --> API
    API -->|15. Save detections| DB
    API -->|16. Response| FA
    FA --> EP1
    EP1 --> VS
    VS --> IG
    IG -->|17. JSON result| User
    
    %% Styling
    classDef istio fill:#4285F4,stroke:#1a73e8,color:white
    classDef backend fill:#34A853,stroke:#188038,color:white
    classDef storage fill:#EA4335,stroke:#C5221F,color:white
    classDef kserve fill:#FBBC04,stroke:#F29900,color:black
    classDef user fill:#673AB7,stroke:#512DA8,color:white
    
    class IG,VS,EP1,EP2 istio
    class FA,API backend
    class PV,DB,FS storage
    class KPA,REV,MS,ML kserve
    class User user
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

## Getting Started

### Prerequisites

- Docker Desktop with Kubernetes enabled
- `kind` - Kubernetes in Docker
- `kubectl` - Kubernetes CLI
- `helm` - Package manager for Kubernetes
- `jq` - JSON processor

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