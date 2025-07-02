#!/bin/bash

# Cleanup script for AIQ Backend deployment
set -e

# Configuration
NAMESPACE="aiq-backend"
CLUSTER_NAME="${CLUSTER_NAME:-kserve-deployment}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${RED}===== AIQ Backend Cleanup Script =====${NC}"
echo "This will remove all resources in namespace: $NAMESPACE"
echo ""

# Confirm deletion
read -p "Are you sure you want to delete all resources? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

# 1. Delete Kubernetes resources
echo -e "${YELLOW}Deleting Kubernetes resources...${NC}"
kubectl delete -k . --ignore-not-found=true 2>/dev/null || true

# 2. Delete namespace (this will delete all resources in the namespace)
echo -e "${YELLOW}Deleting namespace '$NAMESPACE'...${NC}"
kubectl delete namespace $NAMESPACE --ignore-not-found=true

# 3. Clean up storage on Kind node (optional)
read -p "Do you want to delete persistent data from Kind node? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cleaning up storage on Kind node...${NC}"
    docker exec ${CLUSTER_NAME}-control-plane rm -rf /tmp/aiq-storage
    echo "Storage cleaned up"
fi

echo ""
echo -e "${GREEN}✅ Cleanup completed!${NC}" 