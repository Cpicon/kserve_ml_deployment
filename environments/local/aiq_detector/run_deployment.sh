#!/bin/bash

set -e

echo "🚀 AIQ Detector - Full Deployment Process"
echo "========================================="

# Navigate to the correct directory
cd "$(dirname "$0")"

echo "📁 Current directory: $(pwd)"
echo ""

# Step 1: Build and test the image
echo "Step 1: Building and testing the image..."
echo "-----------------------------------------"
./deploy.sh

if [ $? -ne 0 ]; then
    echo "❌ Deployment failed!"
    exit 1
fi

echo ""
echo "Step 2: Testing the deployed model..."
echo "-------------------------------------"
sleep 5  # Give the pod time to fully start

./test_inference.sh

echo ""
echo "🎉 Deployment and testing complete!"
echo ""
echo "📊 Final status check:"
kubectl get inferenceservice aiq-detector -n aiq-model-serving
echo ""
kubectl get pods -n aiq-model-serving -l serving.kserve.io/inferenceservice=aiq-detector 