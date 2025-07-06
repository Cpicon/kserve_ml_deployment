#!/bin/bash
# run_all_local.sh - Run all services locally for development and testing
# This script starts the model server, backend service, and runs integration tests

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
MODEL_PORT=9090
BACKEND_PORT=8000
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR="${PROJECT_ROOT}/logs"
MODEL_LOG="${LOG_DIR}/model_server.log"
BACKEND_LOG="${LOG_DIR}/backend_server.log"

# PIDs for cleanup
MODEL_PID=""
BACKEND_PID=""

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    # Kill model server
    if [ ! -z "$MODEL_PID" ]; then
        echo -e "${BLUE}Stopping model server (PID: $MODEL_PID)...${NC}"
        kill $MODEL_PID 2>/dev/null || true
        # Also kill any child processes
        pkill -P $MODEL_PID 2>/dev/null || true
    fi
    
    # Kill backend server
    if [ ! -z "$BACKEND_PID" ]; then
        echo -e "${BLUE}Stopping backend server (PID: $BACKEND_PID)...${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        # Also kill any child processes
        pkill -P $BACKEND_PID 2>/dev/null || true
    fi
    
    # Additional cleanup for any lingering processes
    echo -e "${BLUE}Cleaning up any remaining processes...${NC}"
    lsof -ti:$MODEL_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# Function to check if a port is open
check_port() {
    local port=$1
    nc -z localhost $port 2>/dev/null
}

# Function to wait for a service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local url=$3
    local max_attempts=60
    local attempt=0
    
    echo -e "${BLUE}Waiting for $service_name to be ready on port $port...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if check_port $port; then
            # Port is open, now check if service is responding
            if curl -sf "$url" -o /dev/null 2>/dev/null; then
                echo -e "${GREEN}✅ $service_name is ready!${NC}"
                return 0
            fi
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "\n${RED}❌ $service_name failed to start after $max_attempts attempts${NC}"
    return 1
}

# Print banner
echo -e "${BOLD}${CYAN}================================================${NC}"
echo -e "${BOLD}${CYAN}     AIQ Circular Detection - Local Testing     ${NC}"
echo -e "${BOLD}${CYAN}================================================${NC}"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv is not installed${NC}"
    echo -e "${YELLOW}Please install uv: https://github.com/astral-sh/uv${NC}"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${RED}❌ jq is not installed${NC}"
    echo -e "${YELLOW}Please install jq: brew install jq (macOS) or apt-get install jq (Linux)${NC}"
    exit 1
fi

# Check if ports are available
if check_port $MODEL_PORT; then
    echo -e "${RED}❌ Port $MODEL_PORT is already in use${NC}"
    echo -e "${YELLOW}Please stop the service using port $MODEL_PORT${NC}"
    exit 1
fi

if check_port $BACKEND_PORT; then
    echo -e "${RED}❌ Port $BACKEND_PORT is already in use${NC}"
    echo -e "${YELLOW}Please stop the service using port $BACKEND_PORT${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites met${NC}\n"

# Step 1: Start Model Server
echo -e "${BOLD}${BLUE}Step 1: Starting Model Server${NC}"
echo -e "${BLUE}================================${NC}"

cd "${PROJECT_ROOT}/environments/local/aiq_detector" || exit 1

# Start model server in background
echo -e "${BLUE}Starting model server on port $MODEL_PORT...${NC}"
echo -e "${YELLOW}Log file: $MODEL_LOG${NC}"

# Run in a subshell to capture the PID properly
(
    ./run_local.sh > "$MODEL_LOG" 2>&1
) &
MODEL_PID=$!

echo -e "${BLUE}Model server started with PID: $MODEL_PID${NC}"

# Wait for model server to be ready
if ! wait_for_service "Model Server" $MODEL_PORT "http://localhost:$MODEL_PORT/v1/models/aiq-detector"; then
    echo -e "${RED}Failed to start model server. Check logs at: $MODEL_LOG${NC}"
    echo -e "${YELLOW}Last 20 lines of model server log:${NC}"
    tail -20 "$MODEL_LOG"
    exit 1
fi

echo -e "${GREEN}✅ Model server is running!${NC}"
echo -e "${CYAN}Swagger UI available at: http://localhost:$MODEL_PORT/docs${NC}\n"

# Step 2: Start Backend Service
echo -e "${BOLD}${BLUE}Step 2: Starting Backend Service${NC}"
echo -e "${BLUE}===================================${NC}"

cd "${PROJECT_ROOT}/services/backend" || exit 1

# Ensure environment variables are exported for the backend service
export ENVIRONMENT=dev
export LOG_LEVEL=DEBUG
export MODE=real
export MODEL_SERVER_URL=http://localhost:$MODEL_PORT
export MODEL_NAME=aiq-detector
export MODEL_SERVICE_TIMEOUT=30.0
export METADATA_STORAGE=database

# Start backend service in background
echo -e "${BLUE}Starting backend service on port $BACKEND_PORT...${NC}"
echo -e "${YELLOW}Log file: $BACKEND_LOG${NC}"
echo -e "${YELLOW}Environment: MODE=$MODE, MODEL_SERVER_URL=$MODEL_SERVER_URL${NC}"

# Run in a subshell to capture the PID properly
(
    # Source the script to ensure environment variables are passed
    bash -c "export MODE=$MODE && export MODEL_SERVER_URL=$MODEL_SERVER_URL && export MODEL_NAME=$MODEL_NAME && export METADATA_STORAGE=$METADATA_STORAGE && export LOG_LEVEL=$LOG_LEVEL && export MODEL_SERVICE_TIMEOUT=$MODEL_SERVICE_TIMEOUT && ./start-dev-real.sh" > "$BACKEND_LOG" 2>&1
) &
BACKEND_PID=$!

echo -e "${BLUE}Backend service started with PID: $BACKEND_PID${NC}"

# Wait for backend service to be ready
if ! wait_for_service "Backend Service" $BACKEND_PORT "http://localhost:$BACKEND_PORT/health"; then
    echo -e "${RED}Failed to start backend service. Check logs at: $BACKEND_LOG${NC}"
    echo -e "${YELLOW}Last 20 lines of backend service log:${NC}"
    tail -20 "$BACKEND_LOG"
    exit 1
fi

echo -e "${GREEN}✅ Backend service is running!${NC}"
echo -e "${CYAN}API Documentation available at: http://localhost:$BACKEND_PORT/docs${NC}\n"

# Step 3: Run Integration Tests
echo -e "${BOLD}${CYAN}Step 3: Running Integration Tests${NC}"
echo -e "${CYAN}====================================${NC}"
cd "${PROJECT_ROOT}/services/backend" || exit 1
bash test_full_integration.sh
TEST_RESULT=$?
cd "${PROJECT_ROOT}" || exit 1

if [ $TEST_RESULT -ne 0 ]; then
    echo -e "${RED}❌ Tests failed!${NC}"
    echo -e "${YELLOW}Check the logs for more information:${NC}"
    echo -e "  - Model server log: ${MODEL_LOG}"
    echo -e "  - Backend service log: ${BACKEND_LOG}"
    echo -e "${YELLOW}Stopping services due to test failure...${NC}"
    cleanup
    exit 1
fi

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${BOLD}${GREEN}✅ All tests passed!${NC}"
    
    # Print service information
    echo -e "\n${BOLD}${CYAN}Services Running:${NC}"
    echo -e "${CYAN}================================================${NC}"
    echo -e "${GREEN}Model Server:${NC}"
    echo -e "  - URL: http://localhost:$MODEL_PORT"
    echo -e "  - Swagger UI: http://localhost:$MODEL_PORT/docs"
    echo -e "  - PID: $MODEL_PID"
    echo -e "  - Log: $MODEL_LOG"
    echo ""
    echo -e "${GREEN}Backend Service:${NC}"
    echo -e "  - URL: http://localhost:$BACKEND_PORT"
    echo -e "  - API Docs: http://localhost:$BACKEND_PORT/docs"
    echo -e "  - PID: $BACKEND_PID"
    echo -e "  - Log: $BACKEND_LOG"
    echo -e "${CYAN}================================================${NC}"
    
    # Interactive mode
    echo -e "\n${YELLOW}Services are running. Press Ctrl+C to stop all services.${NC}"
    echo -e "${BLUE}You can now:${NC}"
    echo -e "  - Test the API at http://localhost:$BACKEND_PORT/docs"
    echo -e "  - View model server at http://localhost:$MODEL_PORT/docs"
    echo -e "  - Check logs in the $LOG_DIR directory"
    
    # Wait for user to press Ctrl+C
    wait
else
    echo -e "${BOLD}${RED}❌ Tests failed!${NC}"
    echo -e "${YELLOW}Check the logs for more information:${NC}"
    echo -e "  - Model server log: $MODEL_LOG"
    echo -e "  - Backend service log: $BACKEND_LOG"
    echo -e "${RED}Stopping services due to test failure...${NC}"
    exit 1
fi 