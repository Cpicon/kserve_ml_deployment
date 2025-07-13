#!/bin/bash
# Quick update script for backend deployment changes

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== Updating Backend Deployment =====${NC}"

# Apply the updated configuration
echo -e "${YELLOW}Applying updated configuration...${NC}"
kubectl apply -k .

# Restart the deployment to pick up changes
echo -e "${YELLOW}Restarting backend deployment...${NC}"
kubectl rollout restart deployment/aiq-circular-detection -n aiq-backend

# Wait for rollout to complete
echo -e "${YELLOW}Waiting for rollout to complete...${NC}"
kubectl rollout status deployment/aiq-circular-detection -n aiq-backend --timeout=300s

# Show the updated pod status
echo -e "${GREEN}✅ Backend deployment updated!${NC}"
kubectl get pods -n aiq-backend -l app=aiq-circular-detection

echo -e "\n${BLUE}To check the configuration:${NC}"
echo -e "kubectl exec -n aiq-backend -it \$(kubectl get pod -n aiq-backend -l app=aiq-circular-detection -o jsonpath='{.items[0].metadata.name}') -- env | grep -E '(MODE|MODEL_)'" 