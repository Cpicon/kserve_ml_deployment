#!/bin/bash

# Script to set up ingress routing for Kind clusters
# This solves the LoadBalancer "pending" issue in local Kind environments

set -e

CLUSTER_NAME="${1:-kserve-deployment}"

echo "Setting up ingress routing for Kind cluster: $CLUSTER_NAME"

# Check if the cluster exists
if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "Error: Kind cluster '$CLUSTER_NAME' not found."
    echo "Available clusters:"
    kind get clusters
    exit 1
fi

# Check if socat is installed in the Kind node
echo "Checking if socat is installed..."
if ! docker exec ${CLUSTER_NAME}-control-plane which socat &>/dev/null; then
    echo "Installing socat..."
    docker exec ${CLUSTER_NAME}-control-plane sh -c "apt-get update && apt-get install -y socat" >/dev/null 2>&1
fi

# Kill existing socat processes if any
echo "Cleaning up existing port forwards..."
docker exec ${CLUSTER_NAME}-control-plane sh -c "pkill socat || true" 2>/dev/null

# Patch Istio ingress gateway to use NodePort if it's LoadBalancer
CURRENT_TYPE=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.spec.type}' 2>/dev/null || echo "")
if [[ "$CURRENT_TYPE" == "LoadBalancer" ]]; then
    echo "Patching Istio ingress gateway to use NodePort..."
    kubectl patch svc istio-ingressgateway -n istio-system -p '{
        "spec": {
            "type": "NodePort",
            "ports": [
                {"name": "status-port", "nodePort": 30021, "port": 15021, "protocol": "TCP", "targetPort": 15021},
                {"name": "http2", "nodePort": 30080, "port": 80, "protocol": "TCP", "targetPort": 8080},
                {"name": "https", "nodePort": 30443, "port": 443, "protocol": "TCP", "targetPort": 8443}
            ]
        }
    }'
fi

# Set up port forwarding
echo "Setting up port forwarding..."
docker exec -d ${CLUSTER_NAME}-control-plane sh -c "socat TCP-LISTEN:80,fork,reuseaddr TCP:localhost:30080"
docker exec -d ${CLUSTER_NAME}-control-plane sh -c "socat TCP-LISTEN:443,fork,reuseaddr TCP:localhost:30443"

echo "✅ Ingress routing configured successfully!"
echo ""
echo "You can now access services through:"
echo "  - HTTP:  http://localhost:80"
echo "  - HTTPS: https://localhost:443"
echo ""
echo "Note: The EXTERNAL-IP will show as <none> for NodePort services, which is normal." 