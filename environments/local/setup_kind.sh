#!/bin/bash

# Script to install Kind and create a Kubernetes cluster.

set -e

# Check if Kind is installed
if ! command -v kind &> /dev/null
then
    echo "Error: kind is not installed. Please install kind to continue." >&2
    echo "Installation instructions: https://kind.sigs.k8s.io/docs/user/quick-start/#installation" >&2
    exit 1
fi

# Create a Kubernetes cluster
echo "Creating Kind cluster 'kserve-deployment'..."
cat <<EOF | kind create cluster --name kserve-deployment --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
EOF

# Set the kubectl context
echo "Setting kubectl context to 'kind-kserve-deployment'..."
kubectl cluster-info --context kind-kserve-deployment

# Verify cluster status
echo "Verifying cluster status..."
if kubectl wait --for=condition=Ready node --all --timeout=60s; then
  echo "Cluster is up and running."
else
  echo "Error: Cluster nodes are not ready." >&2
  exit 1
fi

# Verify node labels
echo "Verifying node labels..."
if kubectl get nodes -l ingress-ready=true | grep -q "kserve-deployment-control-plane"; then
    echo "Node 'kserve-deployment-control-plane' has the 'ingress-ready=true' label."
else
    echo "Error: Node 'kserve-deployment-control-plane' does not have the 'ingress-ready=true' label." >&2
    kubectl get nodes --show-labels >&2
    exit 1
fi

echo "Cluster setup successful."
