#!/bin/bash

# Configuration
MODEL_NAME=aiq-detector
LOCAL_PORT=8080
PID_FILE="/tmp/aiq-detector-port-forward.pid"

# Check if port-forward is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        # Verify it's actually working
        if curl -s http://localhost:${LOCAL_PORT}/v1/models/${MODEL_NAME} >/dev/null 2>&1; then
            exit 0  # Port-forward is running and working
        fi
    fi
fi

exit 1  # Port-forward is not running 