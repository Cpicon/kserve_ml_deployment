# Backend Deployment Guide for Kind Cluster

## Table of Contents

1. [Overview](#overview)
2. [Persistence Architecture](#persistence-architecture)
3. [Kustomization.yaml Purpose](#kustomizationyaml-purpose)
4. [Deployment Steps](#deployment-steps)
   - [Automated Deployment](#automated-deployment-recommended)
   - [Manual Deployment](#manual-deployment-steps)
5. [Testing the Deployment](#testing-the-deployment)
6. [Important Notes](#important-notes)
7. [Troubleshooting](#troubleshooting)
8. [Deployment Architecture Summary](#deployment-architecture-summary)

## Overview

This deployment is configured for a local Kind cluster with:
- SQLite database for metadata storage
- Local file storage for uploaded images
- Single replica (due to SQLite limitations)
- Persistent storage that survives pod restarts

## Persistence Architecture

### Why PVC/PV are Required

1. **Pod Ephemeral Storage**: By default, data in a pod's filesystem is lost when the pod restarts
2. **SQLite Database**: The SQLite database file (`/data/db.sqlite3`) must persist across pod restarts
3. **Uploaded Images**: Image files stored at `/data/images` need to survive pod lifecycle events

### Storage Configuration

- **PersistentVolume (PV)**: Uses `hostPath` storage on the Kind node at `/tmp/aiq-storage`
- **PersistentVolumeClaim (PVC)**: Requests 5Gi of storage with ReadWriteOnce access
- **Volume Mounts**: The PVC is mounted with subPaths for separation:
  - `/data` → for SQLite database and images
  - `/logs` → for application logs

## Kustomization.yaml Purpose

The `kustomization.yaml` file is the configuration file for [Kustomize](https://kustomize.io/), a Kubernetes native configuration management tool. It serves several important purposes:

### What it does:

1. **Resource Organization**: Lists all Kubernetes manifests to be deployed together
   ```yaml
   resources:
     - configmap.yaml
     - storage.yaml
     - deployment.yaml
     - istio-routing.yaml
   ```

2. **Namespace Management**: Automatically adds namespace to all resources
   ```yaml
   namespace: aiq-backend
   ```

3. **Common Labels**: Applies consistent labels across all resources for easier management
   ```yaml
   commonLabels:
     app.kubernetes.io/name: aiq-circular-detection
     app.kubernetes.io/part-of: aiq-system
   ```

### Benefits:

- **Single Command Deployment**: Deploy all resources with `kubectl apply -k .`
- **Environment Flexibility**: Easy to create overlays for different environments (dev, staging, prod)
- **DRY Principle**: Avoid repeating namespace and labels in every manifest
- **Native Kubernetes Support**: No additional tools needed, built into kubectl

### Usage:

```bash
# Apply all resources defined in kustomization.yaml
kubectl apply -k .

# Preview what will be deployed
kubectl kustomize .

# Delete all resources
kubectl delete -k .
```

## Deployment Steps

### Automated Deployment (Recommended)

Use the provided deployment script for a fully automated setup:

```bash
# Deploy everything with one command
cd environments/local/backend
./deploy.sh
```

This script will:
1. Create the `aiq-backend` namespace
2. Enable Istio sidecar injection
3. Create storage directory on Kind node
4. Build and load Docker image
5. Deploy all resources using Kustomize
6. Wait for deployment to be ready

### Manual Deployment Steps

If you prefer manual deployment:

```bash
# 1. Build and Load Image
cd services/backend
docker build -t aiq-circular-detection:latest .
kind load docker-image aiq-circular-detection:latest --name kserve-deployment

# 2. Create namespace and enable Istio
kubectl create namespace aiq-backend
kubectl label namespace aiq-backend istio-injection=enabled

# 3. Create storage on Kind node
docker exec kserve-deployment-control-plane mkdir -p /tmp/aiq-storage
docker exec kserve-deployment-control-plane chmod 777 /tmp/aiq-storage

# 4. Deploy using Kustomize
cd environments/local/backend
kubectl apply -k .

# 5. Wait for deployment
kubectl -n aiq-backend rollout status deployment/aiq-circular-detection
```

## Testing the Deployment

### Automated Testing with Istio

Use the test script to verify the deployment through Istio ingress:

```bash
./test-istio.sh
```

This script will:
- Set up port forwarding through Istio ingress gateway
- Test the health endpoint
- Verify API documentation is accessible
- Show all available endpoints
- Keep port forwarding active for manual testing

### Manual Testing

#### Direct Service Access (bypassing Istio)
```bash
# Port forward directly to the service
kubectl -n aiq-backend port-forward svc/aiq-circular-detection 8080:80

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/docs
curl http://localhost:8080/api/v1/images
```

#### Through Istio Ingress
```bash
# Port forward to Istio ingress gateway
kubectl -n istio-system port-forward svc/istio-ingressgateway 8080:80

# Test endpoints through Istio
curl http://localhost:8080/health
curl http://localhost:8080/docs
curl http://localhost:8080/api/v1/images
```

## Cleanup

To remove all deployed resources:

```bash
./cleanup.sh
```

This will prompt for confirmation and optionally clean up persistent data.

## Important Notes

### SQLite Limitations

- **Single Replica Only**: SQLite doesn't support concurrent writes, so the deployment is limited to 1 replica
- **Not for Production**: For production, use PostgreSQL or MySQL with proper multi-replica support

### Istio Integration

The deployment automatically enables Istio sidecar injection for the `aiq-backend` namespace. This provides:

- **Traffic Management**: Advanced routing, retries, and timeouts
- **Observability**: Automatic metrics, logs, and traces
- **Security**: mTLS between services

The pod will have 2/2 containers running (application + Istio sidecar). Traffic flows through:
1. Istio Gateway → receives external traffic
2. VirtualService → routes to the correct service
3. Envoy sidecar → handles service mesh features
4. Application container → processes the request

### Model Integration

Currently configured in `dummy` mode. To use real model inference:

1. Deploy your model using KServe
2. Update the deployment environment variables:
   - Set `MODE=real`
   - Update `MODEL_SERVER_URL` with your model's endpoint

## Troubleshooting

### Pod Stuck in Pending

If the pod is pending with "unbound PersistentVolumeClaims":

```bash
# Create the hostPath directory on Kind node
docker exec kserve-deployment-control-plane mkdir -p /tmp/aiq-storage
docker exec kserve-deployment-control-plane chmod 777 /tmp/aiq-storage
```

### Data Persistence

Data is stored on the Kind node at `/tmp/aiq-storage`. To backup:

```bash
# Backup data from Kind node
docker exec kserve-deployment-control-plane tar -czf /tmp/backup.tar.gz /tmp/aiq-storage
docker cp kserve-deployment-control-plane:/tmp/backup.tar.gz ./backup.tar.gz
```

### Accessing Logs

```bash
# View application logs
kubectl -n aiq-backend logs -f -l app=aiq-circular-detection

# View logs from a specific container (app or istio-proxy)
kubectl -n aiq-backend logs -f -l app=aiq-circular-detection -c api
kubectl -n aiq-backend logs -f -l app=aiq-circular-detection -c istio-proxy

# Access log files from persistent volume
kubectl -n aiq-backend exec -it deployment/aiq-circular-detection -c api -- ls -la /logs
```

## Deployment Architecture Summary

### Resources Created

1. **Namespace**: `aiq-backend` (with Istio injection enabled)
2. **ConfigMaps**: 
   - `aiq-config` - Main configuration
   - `aiq-config-dev` - Development example
3. **Storage**:
   - PersistentVolume: 5Gi hostPath storage
   - PersistentVolumeClaim: Bound to the PV
4. **Application**:
   - Deployment: Single replica with app + Istio sidecar
   - Service: ClusterIP service on port 80
5. **Istio Routing**:
   - Gateway: Accepts traffic on port 80
   - VirtualService: Routes `/health`, `/docs`, and `/api/v1/*`

### Scripts Provided

- **`deploy.sh`**: Automated deployment with all prerequisites
- **`test-istio.sh`**: Test the deployment through Istio ingress
- **`cleanup.sh`**: Remove all resources and optionally clean data

### Configuration Management

Environment variables are injected from ConfigMap, making it easy to:
- Change log levels without rebuilding
- Switch between environments
- Update application settings on the fly

Simply edit the ConfigMap and restart the pod:
```bash
kubectl -n aiq-backend edit configmap aiq-config
kubectl -n aiq-backend rollout restart deployment/aiq-circular-detection
``` 