#!/bin/bash

set -e

echo "🧪 Testing AIQ Detector inference..."

# Set model name
MODEL_NAME=aiq-detector
LOCAL_PORT=8080

# Check if port-forward is running, start if not
if ! ./check_port_forward.sh; then
    echo "⚠️  Port-forward is not running, starting it..."
    ./start_port_forward.sh
    if [ $? -ne 0 ]; then
        echo "❌ Failed to start port-forward"
        exit 1
    fi
else
    echo "✅ Port-forward is already running"
fi

# Use provided image path or generate test image
if [ -n "$1" ]; then
    INPUT_PATH="$1"
    echo "Using provided image: $INPUT_PATH"
else
    # Default to the dataset image
    INPUT_PATH="../../../dataset/0ba4fa31-913c-45be-9e59-bc14fe4f324e_jpg.rf.89dddeb3544e94d2c5f1aa763b85823d.jpg"
    
    if [ ! -f "$INPUT_PATH" ]; then
        echo "Dataset image not found. Generating test image..."
        python generate_test_image.py
        INPUT_PATH="test_data/test_coin.jpg"
    fi
fi

# Prepare test input JSON
echo ""
echo "📝 Preparing test input..."
python generate_test_image.py "$INPUT_PATH"

# Test model readiness
echo ""
echo "🏥 Checking model readiness..."
RESPONSE=$(curl -s http://localhost:${LOCAL_PORT}/v1/models/${MODEL_NAME} 2>&1)
if [ $? -eq 0 ]; then
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
else
    echo "❌ Failed to connect to model endpoint"
    exit 1
fi

# Send prediction request
echo ""
echo "🚀 Sending inference request..."
PREDICTION=$(curl -s -X POST \
     -H "Content-Type: application/json" \
     -d @test_data/input.json \
     http://localhost:${LOCAL_PORT}/v1/models/${MODEL_NAME}:predict 2>&1)
if [ $? -eq 0 ]; then
    echo "$PREDICTION" | jq '.' 2>/dev/null || echo "$PREDICTION"
else
    echo "❌ Failed to get prediction"
    exit 1
fi

echo ""
echo "✅ Test complete!"
echo ""
echo "📖 Note: Port-forward is still running. To stop it, run: ./stop_port_forward.sh" 