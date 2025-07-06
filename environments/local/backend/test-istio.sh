#!/bin/bash

# Test script for AIQ Backend service through Istio
set -e

# Configuration
NAMESPACE="aiq-backend"
CLUSTER_NAME="${CLUSTER_NAME:-kserve-deployment}"
ISTIO_NAMESPACE="istio-system"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== AIQ Backend Istio Test Script =====${NC}"
echo ""

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    jobs -p | xargs -r kill 2>/dev/null || true
    exit
}
trap cleanup EXIT INT TERM

# 1. Check if deployment exists
echo -e "${YELLOW}Checking deployment status...${NC}"
if ! kubectl -n $NAMESPACE get deployment aiq-circular-detection &>/dev/null; then
    echo -e "${RED}Error: Deployment not found in namespace '$NAMESPACE'${NC}"
    echo "Please run ./deploy.sh first"
    exit 1
fi

# 2. Wait for pods to be ready
echo -e "${YELLOW}Waiting for pods to be ready...${NC}"
kubectl -n $NAMESPACE wait --for=condition=ready pod -l app=aiq-circular-detection --timeout=60s

# 3. Get Istio ingress gateway service type
ISTIO_SVC_TYPE=$(kubectl -n $ISTIO_NAMESPACE get svc istio-ingressgateway -o jsonpath='{.spec.type}')
echo -e "${BLUE}Istio ingress gateway type: $ISTIO_SVC_TYPE${NC}"

# 4. Set up port forwarding based on service type
if [[ "$ISTIO_SVC_TYPE" == "LoadBalancer" ]]; then
    # For LoadBalancer, check if external IP is available
    EXTERNAL_IP=$(kubectl -n $ISTIO_NAMESPACE get svc istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -z "$EXTERNAL_IP" || "$EXTERNAL_IP" == "<pending>" ]]; then
        echo -e "${YELLOW}LoadBalancer has no external IP, using port-forward...${NC}"
        kubectl -n $ISTIO_NAMESPACE port-forward svc/istio-ingressgateway 8080:80 &
        INGRESS_URL="http://localhost:8080"
        sleep 3
    else
        INGRESS_URL="http://$EXTERNAL_IP"
    fi
elif [[ "$ISTIO_SVC_TYPE" == "NodePort" ]]; then
    # For NodePort in Kind, forward from the node
    NODE_PORT=$(kubectl -n $ISTIO_NAMESPACE get svc istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')
    echo -e "${YELLOW}Setting up port forwarding through Kind node (NodePort: $NODE_PORT)...${NC}"
    kubectl -n $ISTIO_NAMESPACE port-forward svc/istio-ingressgateway 8080:80 &
    INGRESS_URL="http://localhost:8080"
    sleep 3
else
    # ClusterIP - direct port forward
    echo -e "${YELLOW}Setting up port forwarding...${NC}"
    kubectl -n $ISTIO_NAMESPACE port-forward svc/istio-ingressgateway 8080:80 &
    INGRESS_URL="http://localhost:8080"
    sleep 3
fi

echo -e "${GREEN}Istio ingress available at: $INGRESS_URL${NC}"
echo ""

# 5. Test health endpoint
echo -e "${YELLOW}Testing health endpoint through Istio...${NC}"
echo "GET $INGRESS_URL/health"
if curl -s -f -w "\nHTTP Status: %{http_code}\n" $INGRESS_URL/health | jq . 2>/dev/null || curl -s -f $INGRESS_URL/health; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
else
    echo -e "${RED}❌ Health check failed!${NC}"
    echo "Checking pod logs..."
    kubectl -n $NAMESPACE logs -l app=aiq-circular-detection --tail=20
    exit 1
fi

echo ""
echo -e "${YELLOW}Testing API docs...${NC}"
echo "GET $INGRESS_URL/docs"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" $INGRESS_URL/docs)
if [[ "$HTTP_CODE" == "200" ]]; then
    echo -e "${GREEN}✅ API docs available at: $INGRESS_URL/docs${NC}"
else
    echo -e "${RED}❌ API docs returned HTTP $HTTP_CODE${NC}"
fi

echo ""
echo -e "${YELLOW}Testing config endpoint...${NC}"
echo "GET $INGRESS_URL/config"
curl -s -w "\nHTTP Status: %{http_code}\n" $INGRESS_URL/config | jq . 2>/dev/null || true

echo ""
echo -e "${GREEN}===== Test Summary =====${NC}"
echo "Service is accessible through Istio at: $INGRESS_URL"
echo ""
echo "Available endpoints:"
echo "  - Health: $INGRESS_URL/health"
echo "  - API Docs: $INGRESS_URL/docs"
echo "  - API: $INGRESS_URL/api/v1/*"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop port forwarding${NC}"

# Keep the script running
wait 