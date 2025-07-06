#!/bin/bash
# Script to run model evaluation with proper environment setup

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}🔍 Model Evaluation Tool${NC}"
echo -e "${BLUE}========================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if servers are running
check_servers() {
    echo -e "\n${BLUE}Checking required services...${NC}"
    
    # Check FastAPI backend
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✅ FastAPI backend is running${NC}"
    else
        echo -e "${RED}❌ FastAPI backend is not running${NC}"
        echo -e "${YELLOW}   Please start it with: cd ../../services/backend && ./start-dev-real.sh${NC}"
        return 1
    fi
    
    # Check model server
    if curl -sf http://localhost:9090/v1/models/aiq-detector > /dev/null; then
        echo -e "${GREEN}✅ Model server is running${NC}"
    else
        echo -e "${RED}❌ Model server is not running${NC}"
        echo -e "${YELLOW}   Please start it with: cd ../../environments/local/aiq_detector && ./run_local.sh${NC}"
        return 1
    fi
    
    return 0
}

# Function to setup environment
setup_environment() {
    echo -e "\n${BLUE}Setting up Python environment...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv .venv
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to create virtual environment${NC}"
            exit 1
        fi
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    echo -e "${YELLOW}Upgrading pip...${NC}"
    pip install --upgrade pip --quiet
    
    # Install dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt --quiet
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install dependencies${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Environment setup complete${NC}"
}

# Function to check dataset
check_dataset() {
    echo -e "\n${BLUE}Checking dataset...${NC}"
    
    if [ ! -d "dataset" ]; then
        echo -e "${RED}❌ Dataset directory not found${NC}"
        echo -e "${YELLOW}   Please create 'dataset' directory and add your evaluation data:${NC}"
        echo -e "${YELLOW}   mkdir -p dataset${NC}"
        echo -e "${YELLOW}   cp /path/to/_annotations.coco.json dataset/${NC}"
        echo -e "${YELLOW}   cp /path/to/images/*.jpg dataset/${NC}"
        return 1
    fi
    
    if [ ! -f "dataset/_annotations.coco.json" ]; then
        echo -e "${RED}❌ Annotations file not found: dataset/_annotations.coco.json${NC}"
        return 1
    fi
    
    # Count images
    image_count=$(find dataset -name "*.jpg" -type f | wc -l)
    if [ $image_count -eq 0 ]; then
        echo -e "${RED}❌ No .jpg images found in services/evaluation/dataset directory${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Dataset found: $image_count images${NC}"
    return 0
}

# Function to run evaluation
run_evaluation() {
    echo -e "\n${BLUE}Starting model evaluation...${NC}"
    echo -e "${BLUE}===========================${NC}\n"
    
    # Create output directory
    mkdir -p output
    
    # Run the evaluation script
    python evaluate_model.py
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✅ Evaluation completed successfully!${NC}"
        echo -e "${BLUE}   Output images saved in: output/${NC}"
    else
        echo -e "\n${RED}❌ Evaluation failed${NC}"
        exit 1
    fi
}

# Main execution
main() {
    # Handle setup command
    if [ "$1" == "setup" ]; then
        setup_environment
        exit 0
    fi
    
    # Regular evaluation run
    echo -e "${BLUE}Preparing to run model evaluation...${NC}"
    
    # Check Python
    if ! command_exists python3; then
        echo -e "${RED}❌ Python 3 is not installed${NC}"
        exit 1
    fi
    
    # Setup environment if needed
    if [ ! -d ".venv" ] || [ "$1" == "--fresh" ]; then
        setup_environment
    else
        source .venv/bin/activate
    fi
    
    # Check dataset
    if ! check_dataset; then
        exit 1
    fi
    
    # Check servers
    if ! check_servers; then
        echo -e "\n${YELLOW}⚠️  Please start the required services and try again${NC}"
        exit 1
    fi
    
    # Run evaluation
    run_evaluation
}

# Execute main function
main "$@" 