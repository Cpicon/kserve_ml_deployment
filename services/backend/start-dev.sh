#!/bin/bash
# Development startup script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments
QUIET_MODE=false
for arg in "$@"; do
    case $arg in
        --quiet)
            QUIET_MODE=true
            shift
            ;;
        *)
            # Keep other arguments
            ;;
    esac
done

# Determine mode from environment variable, default to dummy
MODE=${MODE:-dummy}

if [ "$MODE" = "real" ]; then
    echo -e "${BLUE}Starting AIQ Circular Detection Service (Development - Real Mode)${NC}"
else
    echo -e "${BLUE}Starting AIQ Circular Detection Service (Development)${NC}"
fi

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
if [ "$QUIET_MODE" = true ]; then
    uv sync --active --dev > /dev/null 2>&1
else
    uv sync --active --dev
fi

# Set development environment variables
export ENVIRONMENT=dev
export LOG_LEVEL=DEBUG

# Set mode-specific variables
if [ "$MODE" = "real" ]; then
    export MODE=real
    export MODEL_SERVER_URL=${MODEL_SERVER_URL:-http://localhost:9090}
    export MODEL_NAME=${MODEL_NAME:-aiq-detector}
    export MODEL_SERVICE_TIMEOUT=${MODEL_SERVICE_TIMEOUT:-30.0}
    export METADATA_STORAGE=${METADATA_STORAGE:-database}
    
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
else
    export MODE=dummy
    export MODEL_NAME="circular-detector"
    export METADATA_STORAGE=database
fi

# Start the FastAPI dev server
if [ "$MODE" = "real" ]; then
    echo -e "${GREEN}Starting FastAPI development server in REAL mode...${NC}"
    echo -e "${BLUE}Model Server URL: ${MODEL_SERVER_URL}${NC}"
    echo -e "${BLUE}Model Name: ${MODEL_NAME}${NC}"
else
    echo -e "${GREEN}Starting FastAPI development server in DUMMY mode...${NC}"
    echo -e "${BLUE}Model Name: ${MODEL_NAME}${NC}"
fi

echo -e "${BLUE}Access the API at: http://localhost:8000${NC}"
echo -e "${BLUE}API Documentation: http://localhost:8000/docs${NC}"
echo -e "${BLUE}Health check: http://localhost:8000/health${NC}"
echo -e "${BLUE}Configuration: http://localhost:8000/config${NC}"

if [ "$QUIET_MODE" = true ]; then
    echo -e "${BLUE}🔇 Server logs suppressed - check logs for full output${NC}"
    uv run --active fastapi dev aiq_circular_detection > /dev/null 2>&1
else
    uv run --active fastapi dev aiq_circular_detection
fi