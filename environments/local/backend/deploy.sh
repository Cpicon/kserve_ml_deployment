#!/bin/bash

# Deployment script for AIQ Circular Detection backend service
set -e

# Configuration
NAMESPACE="aiq-backend"
CLUSTER_NAME="${CLUSTER_NAME:-kserve-deployment}"
STORAGE_PATH="/tmp/aiq-storage"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== AIQ Backend Deployment Script =====${NC}"
echo "Cluster: $CLUSTER_NAME"
echo "Namespace: $NAMESPACE"
echo ""

# 1. Check if cluster exists
echo -e "${YELLOW}Checking Kind cluster...${NC}"
if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo -e "${RED}Error: Kind cluster '$CLUSTER_NAME' not found.${NC}"
    echo "Available clusters:"
    kind get clusters
    exit 1
fi

# 2. Create namespace if it doesn't exist
echo -e "${YELLOW}Creating namespace...${NC}"
if kubectl get namespace $NAMESPACE &>/dev/null; then
    echo "Namespace '$NAMESPACE' already exists"
else
    kubectl create namespace $NAMESPACE
    echo "Created namespace '$NAMESPACE'"
fi

# 3. Label namespace for Istio injection
echo -e "${YELLOW}Enabling Istio sidecar injection...${NC}"
kubectl label namespace $NAMESPACE istio-injection=enabled --overwrite

# 4. Create storage directory on Kind node
echo -e "${YELLOW}Preparing storage on Kind node...${NC}"
docker exec ${CLUSTER_NAME}-control-plane mkdir -p ${STORAGE_PATH}
docker exec ${CLUSTER_NAME}-control-plane chmod 777 ${STORAGE_PATH}
echo "Created storage directory: ${STORAGE_PATH}"

# 5. Build and load Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
cd ../../../services/backend
docker build -t aiq-circular-detection:latest .

echo -e "${YELLOW}Loading image into Kind cluster...${NC}"
kind load docker-image aiq-circular-detection:latest --name ${CLUSTER_NAME}

# 6. Deploy the application
echo -e "${YELLOW}Deploying application...${NC}"
cd ../../environments/local/backend
kubectl apply -k .

# 7. Wait for deployment to be ready
echo -e "${YELLOW}Waiting for deployment to be ready...${NC}"
kubectl -n $NAMESPACE rollout status deployment/aiq-circular-detection --timeout=300s

# 8. Show deployment status
echo ""
echo -e "${GREEN}===== Deployment Status =====${NC}"
kubectl -n $NAMESPACE get pods,svc,pvc -l app=aiq-circular-detection

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo ""
echo "To test the service, run:"
echo "  ./test-istio.sh"
echo ""
echo "To view logs:"
echo "  kubectl -n $NAMESPACE logs -f -l app=aiq-circular-detection"
echo ""
echo "To access directly (bypassing Istio):"
echo "  kubectl -n $NAMESPACE port-forward svc/aiq-circular-detection 8080:80" 