#!/bin/bash

set -e

echo "🚀 Starting AIQ Detector deployment (Docker build)..."

# Check if we're in the right directory
if [ ! -f "model.py" ]; then
    echo "❌ Error: model.py not found. Please run this script from the aiq_detector directory."
    exit 1
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t aiq-detector:latest .

# Test the image locally
echo "🧪 Testing image locally..."
echo "Starting local container..."
docker run -d --name aiq-detector-test -p 8080:8080 aiq-detector:latest

# Wait for the server to start
echo "Waiting for server to start..."
sleep 20

# Check if server is running
if curl -sf http://localhost:8080/v1/models/aiq-detector -o /dev/null; then
    echo "✅ Server is running!"
    echo "📖 Swagger UI available at: http://localhost:8080/docs"
else
    echo "❌ Server failed to start. Checking logs..."
    docker logs aiq-detector-test
    docker stop aiq-detector-test || true
    docker rm aiq-detector-test || true
    exit 1
fi

# Stop test container
docker stop aiq-detector-test
docker rm aiq-detector-test

# Load image into kind cluster
echo "📦 Loading image into kind cluster..."
kind load docker-image aiq-detector:latest --name kserve-deployment

# Create namespace if it doesn't exist
echo "📁 Creating namespace..."
kubectl create namespace aiq-model-serving --dry-run=client -o yaml | kubectl apply -f -

# Apply the InferenceService
echo "🎯 Deploying InferenceService..."
kubectl apply -f aiq-detector.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=Ready inferenceservice/aiq-detector -n aiq-model-serving --timeout=300s

# Get the service status
echo "✅ Deployment complete!"
echo ""
echo "📊 InferenceService status:"
kubectl get inferenceservice aiq-detector -n aiq-model-serving

echo ""
echo "🔍 To check pod status:"
echo "kubectl get pods -n aiq-model-serving -l serving.kserve.io/inferenceservice=aiq-detector"

echo ""
echo "📡 Service endpoint:"
kubectl get inferenceservice aiq-detector -n aiq-model-serving -o jsonpath='{.status.url}' | sed 's/http:/http:\/\//' | sed 's/$/\n/' 