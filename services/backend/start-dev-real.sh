#!/bin/bash
# Development startup script for real mode (with model server)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting AIQ Circular Detection Service (Development - Real Mode)${NC}"

# Ensure we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    uv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "${GREEN}Using existing virtual environment.${NC}"
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Install/update dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
uv sync --active --dev

# Set development environment variables for real mode
export ENVIRONMENT=dev
export LOG_LEVEL=DEBUG
export MODE=real
export MODEL_SERVER_URL=http://localhost:9090
export MODEL_NAME=aiq-detector
export MODEL_SERVICE_TIMEOUT=30.0
export METADATA_STORAGE=database

# Check if model server is running
echo -e "${BLUE}Checking model server at ${MODEL_SERVER_URL}...${NC}"
if curl -sf "${MODEL_SERVER_URL}/v1/models/${MODEL_NAME}" -o /dev/null; then
    echo -e "${GREEN}✅ Model server is running!${NC}"
else
    echo -e "${RED}❌ Model server is not running at ${MODEL_SERVER_URL}${NC}"
    echo -e "${RED}Please start the model server first:${NC}"
    echo -e "${BLUE}cd environments/local/aiq_detector && ./run_local.sh${NC}"
    exit 1
fi

# Start the FastAPI dev server
echo -e "${GREEN}Starting FastAPI development server in REAL mode...${NC}"
echo -e "${BLUE}Model Server URL: ${MODEL_SERVER_URL}${NC}"
echo -e "${BLUE}Model Name: ${MODEL_NAME}${NC}"
echo -e "${BLUE}Access the API at: http://localhost:8000${NC}"
echo -e "${BLUE}API Documentation: http://localhost:8000/docs${NC}"
echo -e "${BLUE}Health check: http://localhost:8000/health${NC}"
echo -e "${BLUE}Configuration: http://localhost:8000/config${NC}"

uv run --active fastapi dev aiq_circular_detection 