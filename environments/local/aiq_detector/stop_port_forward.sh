#!/bin/bash

# Configuration
PID_FILE="/tmp/aiq-detector-port-forward.pid"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Stopping port-forward for AIQ Detector...${NC}"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo -e "${GREEN}✅ Port-forward stopped (PID: $PID)${NC}"
        rm -f "$PID_FILE"
        rm -f "/tmp/aiq-detector-port-forward.log"
    else
        echo -e "${YELLOW}⚠️  Port-forward process not found (stale PID file)${NC}"
        rm -f "$PID_FILE"
    fi
else
    echo -e "${YELLOW}ℹ️  No port-forward is running${NC}"
    
    # Try to find any kubectl port-forward processes for our service
    PIDS=$(pgrep -f "kubectl port-forward.*aiq-detector-predictor" || true)
    if [ -n "$PIDS" ]; then
        echo -e "${YELLOW}Found orphaned port-forward processes, cleaning up...${NC}"
        for pid in $PIDS; do
            kill $pid 2>/dev/null || true
            echo "   Killed process $pid"
        done
    fi
fi 