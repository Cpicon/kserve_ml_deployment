#!/bin/bash
# Development startup script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting AIQ Circular Detection Service (Development)${NC}"

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
    echo -e "${GREEN}Acvivating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Install/update dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
uv sync --active --dev

# Set development environment variables if needed
export ENVIRONMENT=dev
export LOG_LEVEL=DEBUG

# Start the FastAPI dev server
echo -e "${GREEN}Starting FastAPI development server...${NC}"
echo -e "${BLUE}Access the API at: http://localhost:8000${NC}"
echo -e "${BLUE}API Documentation: http://localhost:8000/docs${NC}"
echo -e "${BLUE}Health check: http://localhost:8000/health${NC}"
echo -e "${BLUE}Configuration: http://localhost:8000/config${NC}"

uv run --active fastapi dev aiq_circular_detection