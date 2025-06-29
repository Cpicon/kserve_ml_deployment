#!/bin/bash

# Helper script to test KServe InferenceServices
# Usage: ./test_inference.sh <service-name> <namespace> <payload-file>

set -e

# Parse arguments, or use default values
SERVICE_NAME="${1:-sklearn-iris}"
NAMESPACE="${2:-kserve}"
PAYLOAD_FILE="${3:-iris-input.json}"

# Check arguments
if [[ -z "$SERVICE_NAME" || -z "$NAMESPACE" || -z "$PAYLOAD_FILE" ]]; then
    echo "Usage: $0 <service-name> <namespace> <payload-file>"
    echo "Example: $0 sklearn-iris kserve-test iris-input.json"
    exit 1
fi

# Check if payload file exists
if [[ ! -f "$PAYLOAD_FILE" ]]; then
    echo "Error: Payload file '$PAYLOAD_FILE' not found."
    exit 1
fi

# Get the InferenceService URL
echo "Checking InferenceService status..."
URL=$(kubectl get inferenceservice ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.status.url}' 2>/dev/null)

if [[ -z "$URL" ]]; then
    echo "Error: InferenceService '${SERVICE_NAME}' not found in namespace '${NAMESPACE}'"
    exit 1
fi

# Extract hostname from URL
HOSTNAME=$(echo $URL | sed 's|http://||' | sed 's|https://||')

echo "InferenceService URL: $URL"
echo ""

# Test prediction endpoint
echo "Testing prediction endpoint..."
echo "Request payload:"
cat $PAYLOAD_FILE | jq . 2>/dev/null || cat $PAYLOAD_FILE
echo ""

echo "Response:"
curl -s -H "Host: ${HOSTNAME}" \
     -H "Content-Type: application/json" \
     http://localhost:80/v1/models/${SERVICE_NAME}:predict \
     -d @${PAYLOAD_FILE} | jq . 2>/dev/null || echo "Error: Failed to get prediction"

echo ""

# Test model metadata endpoint
echo "Model metadata:"
curl -s -H "Host: ${HOSTNAME}" \
     http://localhost:80/v1/models/${SERVICE_NAME} | jq . 2>/dev/null || echo "Error: Failed to get model metadata" 