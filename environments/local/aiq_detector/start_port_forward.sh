#!/bin/bash

set -e

# Configuration
MODEL_NAME=aiq-detector
NAMESPACE=aiq-model-serving
LOCAL_PORT=8080
SERVICE_PORT=80
SERVICE_NAME="${MODEL_NAME}-predictor-00001-private"
PID_FILE="/tmp/aiq-detector-port-forward.pid"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔌 Starting port-forward for AIQ Detector...${NC}"

# Check if port-forward is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Port-forward is already running (PID: $PID)${NC}"
        echo "   Forwarding localhost:${LOCAL_PORT} -> ${SERVICE_NAME}:${SERVICE_PORT}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Stale PID file found, cleaning up...${NC}"
        rm -f "$PID_FILE"
    fi
fi

# Check if the port is already in use
if lsof -Pi :${LOCAL_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}❌ Port ${LOCAL_PORT} is already in use${NC}"
    echo "   Please stop the process using this port or choose a different port"
    exit 1
fi

# Check if the service exists
if ! kubectl get svc ${SERVICE_NAME} -n ${NAMESPACE} >/dev/null 2>&1; then
    echo -e "${RED}❌ Service ${SERVICE_NAME} not found in namespace ${NAMESPACE}${NC}"
    exit 1
fi

# Start port-forwarding
echo "📡 Starting port-forward..."
echo "   Service: ${SERVICE_NAME}"
echo "   Namespace: ${NAMESPACE}"
echo "   Local port: ${LOCAL_PORT}"

kubectl port-forward -n ${NAMESPACE} svc/${SERVICE_NAME} ${LOCAL_PORT}:${SERVICE_PORT} > /tmp/aiq-detector-port-forward.log 2>&1 &
PF_PID=$!

# Save PID
echo $PF_PID > "$PID_FILE"

# Wait for port-forward to be ready
echo -n "⏳ Waiting for port-forward to start"
for i in {1..10}; do
    sleep 1
    echo -n "."
    if curl -s http://localhost:${LOCAL_PORT}/v1/models/${MODEL_NAME} >/dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}✅ Port-forward started successfully (PID: $PF_PID)${NC}"
        echo ""
        echo "📌 Service endpoints:"
        echo "   - Model info: http://localhost:${LOCAL_PORT}/v1/models/${MODEL_NAME}"
        echo "   - Predictions: http://localhost:${LOCAL_PORT}/v1/models/${MODEL_NAME}:predict"
        echo "   - Swagger UI: http://localhost:${LOCAL_PORT}/docs"
        echo ""
        echo "💡 To stop port-forwarding, run: ./stop_port_forward.sh"
        exit 0
    fi
done

echo ""
echo -e "${RED}❌ Port-forward failed to start${NC}"
echo "Check logs at: /tmp/aiq-detector-port-forward.log"
cat /tmp/aiq-detector-port-forward.log
kill $PF_PID 2>/dev/null || true
rm -f "$PID_FILE"
exit 1 