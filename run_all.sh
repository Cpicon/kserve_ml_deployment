#!/bin/bash
# run_all.sh - Run all services locally or in Kind for development and testing
# This script starts the model server, backend service, runs integration tests,
# and optionally performs model evaluation if a dataset is available
# Usage: ./run_all.sh [--mode local|kind]

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
CLUSTER_NAME="kserve-deployment"
INFRASTRUCTURE_LOG="${LOG_DIR}/infrastructure_setup.log"

# PIDs for cleanup
MODEL_PID=""
BACKEND_PID=""
PORT_FORWARD_PIDS=()

# Default mode
RUN_MODE="local"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            RUN_MODE="$2"
            if [[ "$RUN_MODE" != "local" && "$RUN_MODE" != "kind" ]]; then
                echo -e "${RED}Error: Invalid mode '$RUN_MODE'. Use 'local' or 'kind'${NC}"
                echo "Usage: $0 [--mode local|kind]"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--mode local|kind]"
            exit 1
            ;;
    esac
done

# Function to setup infrastructure for Kind mode
setup_kind_infrastructure() {
    echo -e "${BOLD}${BLUE}Setting up Kind infrastructure...${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo -e "${YELLOW}This is a one-time setup that will take several minutes...${NC}"
    echo -e "${YELLOW}Log file: $INFRASTRUCTURE_LOG${NC}"
    
    cd "${PROJECT_ROOT}/environments/local" || exit 1
    
    # Step 1: Create Kind cluster
    echo -e "\n${BLUE}1. Creating Kind cluster...${NC}"
    if ./setup_kind.sh 2>&1 | tee "$INFRASTRUCTURE_LOG"; then
        echo -e "${GREEN}✅ Kind cluster created${NC}"
    else
        echo -e "${RED}❌ Failed to create Kind cluster${NC}"
        return 1
    fi
    
    # Step 2: Install KServe and dependencies
    echo -e "\n${BLUE}2. Installing KServe and dependencies...${NC}"
    echo -e "${YELLOW}This may take 5-10 minutes...${NC}"
    if ./install_kserve_knative.sh 2>&1 | tee -a "$INFRASTRUCTURE_LOG"; then
        echo -e "${GREEN}✅ KServe installed${NC}"
    else
        echo -e "${RED}❌ Failed to install KServe${NC}"
        return 1
    fi
    
    # Step 3: Setup ingress routing
    echo -e "\n${BLUE}3. Setting up ingress routing...${NC}"
    if ./setup_ingress_routing.sh ${CLUSTER_NAME} 2>&1 | tee -a "$INFRASTRUCTURE_LOG"; then
        echo -e "${GREEN}✅ Ingress routing configured${NC}"
    else
        echo -e "${RED}❌ Failed to setup ingress routing${NC}"
        return 1
    fi
    
    # Step 4: Deploy model server
    echo -e "\n${BLUE}4. Deploying model server to Kubernetes...${NC}"
    cd "${PROJECT_ROOT}/environments/local/aiq_detector" || exit 1
    if ./deploy-docker.sh 2>&1 | tee -a "$INFRASTRUCTURE_LOG"; then
        echo -e "${GREEN}✅ Model server deployed${NC}"
    else
        echo -e "${RED}❌ Failed to deploy model server${NC}"
        return 1
    fi
    
    # Step 5: Deploy backend service
    echo -e "\n${BLUE}5. Deploying backend service to Kubernetes...${NC}"
    cd "${PROJECT_ROOT}/environments/local/backend" || exit 1
    if ./deploy.sh 2>&1 | tee -a "$INFRASTRUCTURE_LOG"; then
        echo -e "${GREEN}✅ Backend service deployed${NC}"
    else
        echo -e "${RED}❌ Failed to deploy backend service${NC}"
        return 1
    fi
    
    cd "${PROJECT_ROOT}" || exit 1
    echo -e "\n${GREEN}✅ Infrastructure setup complete!${NC}"
    return 0
}

# Function to check if Kind infrastructure is ready
check_kind_infrastructure() {
    # Check if cluster exists
    if ! kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        return 1
    fi
    
    # Check if KServe is installed
    if ! kubectl get namespace kserve &>/dev/null; then
        return 1
    fi
    
    # Check if model is deployed
    if ! kubectl get inferenceservice aiq-detector -n aiq-model-serving &>/dev/null; then
        return 1
    fi
    
    # Check if backend is deployed
    if ! kubectl get deployment aiq-circular-detection -n aiq-backend &>/dev/null; then
        return 1
    fi
    
    return 0
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    if [[ "$RUN_MODE" == "local" ]]; then
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
    else
        # Kind mode cleanup - kill port forwards
        echo -e "${BLUE}Cleaning up port forwards...${NC}"
        for pid in "${PORT_FORWARD_PIDS[@]}"; do
            kill $pid 2>/dev/null || true
        done
    fi
    
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
echo -e "${BOLD}${CYAN}     AIQ Circular Detection - Testing           ${NC}"
echo -e "${BOLD}${CYAN}     Mode: ${RUN_MODE}                          ${NC}"
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

# Kind-specific prerequisites
if [[ "$RUN_MODE" == "kind" ]]; then
    if ! command -v kind &> /dev/null; then
        echo -e "${RED}❌ kind is not installed${NC}"
        echo -e "${YELLOW}Please install kind: https://kind.sigs.k8s.io/docs/user/quick-start/#installation${NC}"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed${NC}"
        exit 1
    fi
    
    # Check if infrastructure is ready
    if ! check_kind_infrastructure; then
        echo -e "${YELLOW}⚠️  Kind infrastructure not found or incomplete${NC}"
        echo -e "${BLUE}Would you like to set it up now? This will:${NC}"
        echo -e "  - Create a Kind cluster"
        echo -e "  - Install KServe and dependencies"
        echo -e "  - Deploy the model and backend services"
        echo -e "${YELLOW}This is a one-time setup that takes 5-10 minutes.${NC}"
        
        read -p "Continue with setup? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if setup_kind_infrastructure; then
                echo -e "${GREEN}✅ Infrastructure is now ready!${NC}"
            else
                echo -e "${RED}❌ Infrastructure setup failed. Check log: $INFRASTRUCTURE_LOG${NC}"
                exit 1
            fi
        else
            echo -e "${YELLOW}Setup cancelled. You can run './run_all_k8s.sh' manually to set up infrastructure.${NC}"
            exit 0
        fi
    else
        echo -e "${GREEN}✅ Kind infrastructure is ready${NC}"
    fi
else
    # Check if ports are available for local mode
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
fi

echo -e "${GREEN}✅ All prerequisites met${NC}\n"

if [[ "$RUN_MODE" == "local" ]]; then
    # === LOCAL MODE ===
    
    # Step 1: Start Model Server
    echo -e "${BOLD}${BLUE}Step 1: Starting Model Server (Local)${NC}"
    echo -e "${BLUE}=====================================${NC}"

    cd "${PROJECT_ROOT}/environments/local/aiq_detector" || exit 1

    # Start model server in background
    echo -e "${BLUE}Starting model server on port $MODEL_PORT...${NC}"
    echo -e "${YELLOW}Setup logs will be shown below and saved to: $MODEL_LOG${NC}"
    echo -e "${YELLOW}Server logs are suppressed for cleaner output${NC}"
    echo -e "${CYAN}=================== MODEL SERVER SETUP ===================${NC}"

    # Run in a subshell to capture the PID properly
    (
        ./run_local.sh --quiet 2>&1 | tee "$MODEL_LOG"
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
    echo -e "${BOLD}${BLUE}Step 2: Starting Backend Service (Local)${NC}"
    echo -e "${BLUE}========================================${NC}"

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
    echo -e "${YELLOW}Setup logs will be shown below and saved to: $BACKEND_LOG${NC}"
    echo -e "${YELLOW}Server logs are suppressed for cleaner output${NC}"
    echo -e "${YELLOW}Environment: MODE=$MODE, MODEL_SERVER_URL=$MODEL_SERVER_URL${NC}"
    echo -e "${CYAN}================== BACKEND SERVICE SETUP ==================${NC}"

    # Run in a subshell to capture the PID properly
    (
        # Source the script to ensure environment variables are passed
        bash -c "export MODE=$MODE && export MODEL_SERVER_URL=$MODEL_SERVER_URL && export MODEL_NAME=$MODEL_NAME && export METADATA_STORAGE=$METADATA_STORAGE && export LOG_LEVEL=$LOG_LEVEL && export MODEL_SERVICE_TIMEOUT=$MODEL_SERVICE_TIMEOUT && ./start-dev.sh --quiet" 2>&1 | tee "$BACKEND_LOG"
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

else
    # === KIND MODE ===
    
    # Step 1: Setup Port Forwards for Model Server
    echo -e "${BOLD}${BLUE}Step 1: Setting up Model Server Access (K8s)${NC}"
    echo -e "${BLUE}=============================================${NC}"
    
    # Get the predictor pod
    PREDICTOR_POD=$(kubectl get pods -n aiq-model-serving -l serving.kserve.io/inferenceservice=aiq-detector -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -z "$PREDICTOR_POD" ]; then
        echo -e "${RED}❌ Could not find model predictor pod${NC}"
        echo -e "${YELLOW}Make sure the model is deployed: ./run_all_k8s.sh${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Found predictor pod: $PREDICTOR_POD${NC}"
    echo -e "${BLUE}Setting up port-forward for model server...${NC}"
    
    kubectl -n aiq-model-serving port-forward pod/$PREDICTOR_POD $MODEL_PORT:8080 > /dev/null 2>&1 &
    MODEL_PORT_FORWARD_PID=$!
    PORT_FORWARD_PIDS+=($MODEL_PORT_FORWARD_PID)
    
    # Wait for model server to be ready
    if ! wait_for_service "Model Server" $MODEL_PORT "http://localhost:$MODEL_PORT/v1/models/aiq-detector"; then
        echo -e "${RED}Failed to connect to model server${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Model server is accessible!${NC}"
    echo -e "${CYAN}Swagger UI available at: http://localhost:$MODEL_PORT/docs${NC}\n"
    
    # Step 2: Setup Port Forward for Backend Service
    echo -e "${BOLD}${BLUE}Step 2: Setting up Backend Service Access (K8s)${NC}"
    echo -e "${BLUE}================================================${NC}"
    
    echo -e "${BLUE}Setting up port-forward for backend service...${NC}"
    kubectl -n aiq-backend port-forward svc/aiq-circular-detection $BACKEND_PORT:80 > /dev/null 2>&1 &
    BACKEND_PORT_FORWARD_PID=$!
    PORT_FORWARD_PIDS+=($BACKEND_PORT_FORWARD_PID)
    
    # Wait for backend service to be ready
    if ! wait_for_service "Backend Service" $BACKEND_PORT "http://localhost:$BACKEND_PORT/health"; then
        echo -e "${RED}Failed to connect to backend service${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Backend service is accessible!${NC}"
    echo -e "${CYAN}API Documentation available at: http://localhost:$BACKEND_PORT/docs${NC}\n"
fi

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
    if [[ "$RUN_MODE" == "local" ]]; then
        echo -e "  - Model server setup log: ${MODEL_LOG}"
        echo -e "  - Backend service setup log: ${BACKEND_LOG}"
    else
        echo -e "  - Model logs: kubectl logs -n aiq-model-serving -l serving.kserve.io/inferenceservice=aiq-detector"
        echo -e "  - Backend logs: kubectl logs -n aiq-backend -l app=aiq-circular-detection"
    fi
    echo -e "${YELLOW}Stopping services due to test failure...${NC}"
    cleanup
    exit 1
fi

# Step 4: Run Model Evaluation (Optional)
echo -e "\n${BOLD}${CYAN}Step 4: Running Model Evaluation${NC}"
echo -e "${CYAN}====================================${NC}"

EVALUATION_DIR="${PROJECT_ROOT}/services/evaluation"
EVALUATION_LOG="${LOG_DIR}/evaluation.log"

# Check if evaluation dataset exists
if [ -d "${EVALUATION_DIR}/dataset" ] && [ -f "${EVALUATION_DIR}/dataset/_annotations.coco.json" ]; then
    echo -e "${BLUE}Evaluation dataset found. Running model evaluation...${NC}"
    echo -e "${YELLOW}This may take a few minutes depending on dataset size...${NC}"
    
    cd "${EVALUATION_DIR}" || exit 1
    
    # Run evaluation and capture output
    if ./run_evaluation.sh > "${EVALUATION_LOG}" 2>&1; then
        echo -e "${GREEN}✅ Model evaluation completed successfully!${NC}"
        
        # Extract and display key metrics from the log
        echo -e "\n${BOLD}${CYAN}Model Performance Summary:${NC}"
        echo -e "${CYAN}================================================${NC}"
        
        # Extract F1 Score results
        if grep -A 10 "F1 SCORE RESULTS" "${EVALUATION_LOG}" > /dev/null; then
            echo -e "${GREEN}F1 Score Metrics (Multiple IoU Thresholds):${NC}"
            # Extract the header, separator line, and data rows from the new table format
            grep -A 10 "F1 SCORE RESULTS" "${EVALUATION_LOG}" | grep -E "^IoU|^---|^[0-9]\." | sed 's/^/  /'
        fi
        
        # Extract Jaccard Index results
        if grep -A 10 "JACCARD INDEX RESULTS" "${EVALUATION_LOG}" > /dev/null; then
            echo -e "\n${GREEN}Jaccard Index Metrics:${NC}"
            grep -E "Simple Average|Weighted Average|Minimum|Maximum|Total ground truth" "${EVALUATION_LOG}" | tail -5 | sed 's/^/  /'
        fi
        
        echo -e "${CYAN}================================================${NC}"
        echo -e "${YELLOW}Full evaluation log: ${EVALUATION_LOG}${NC}"
        echo -e "${YELLOW}Annotated images saved in: ${EVALUATION_DIR}/output/${NC}"
        
        EVAL_RESULT=0
    else
        echo -e "${RED}❌ Model evaluation failed!${NC}"
        echo -e "${YELLOW}Check the evaluation log: ${EVALUATION_LOG}${NC}"
        echo -e "${YELLOW}Last 20 lines of evaluation log:${NC}"
        tail -20 "${EVALUATION_LOG}"
        EVAL_RESULT=1
    fi
    
    cd "${PROJECT_ROOT}" || exit 1
else
    echo -e "${YELLOW}⚠️  Evaluation dataset not found. Skipping model evaluation.${NC}"
    echo -e "${BLUE}To enable evaluation, add your dataset to: ${EVALUATION_DIR}/dataset/${NC}"
    echo -e "${BLUE}Required files:${NC}"
    echo -e "  - ${EVALUATION_DIR}/dataset/_annotations.coco.json"
    echo -e "  - ${EVALUATION_DIR}/dataset/*.jpg (image files)"
    EVAL_RESULT=0  # Don't fail if evaluation dataset is not present
fi

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${BOLD}${GREEN}✅ All tests passed!${NC}"
    
    # Show evaluation status if it was run
    if [ -n "${EVAL_RESULT+x}" ]; then
        if [ $EVAL_RESULT -eq 0 ] && [ -f "${EVALUATION_LOG}" ]; then
            echo -e "${BOLD}${GREEN}✅ Model evaluation completed!${NC}"
        fi
    fi
    
    # Print service information
    echo -e "\n${BOLD}${CYAN}Services Running (${RUN_MODE} mode):${NC}"
    echo -e "${CYAN}================================================${NC}"
    
    if [[ "$RUN_MODE" == "local" ]]; then
        echo -e "${GREEN}Model Server:${NC}"
        echo -e "  - URL: http://localhost:$MODEL_PORT"
        echo -e "  - Swagger UI: http://localhost:$MODEL_PORT/docs"
        echo -e "  - PID: $MODEL_PID"
        echo -e "  - Setup Log: $MODEL_LOG"
        echo ""
        echo -e "${GREEN}Backend Service:${NC}"
        echo -e "  - URL: http://localhost:$BACKEND_PORT"
        echo -e "  - API Docs: http://localhost:$BACKEND_PORT/docs"
        echo -e "  - PID: $BACKEND_PID"
        echo -e "  - Setup Log: $BACKEND_LOG"
    else
        echo -e "${GREEN}Model Server (K8s):${NC}"
        echo -e "  - URL: http://localhost:$MODEL_PORT (via port-forward)"
        echo -e "  - Swagger UI: http://localhost:$MODEL_PORT/docs"
        echo -e "  - Pod: $PREDICTOR_POD"
        echo -e "  - Logs: kubectl logs -n aiq-model-serving pod/$PREDICTOR_POD"
        echo ""
        echo -e "${GREEN}Backend Service (K8s):${NC}"
        echo -e "  - URL: http://localhost:$BACKEND_PORT (via port-forward)"
        echo -e "  - API Docs: http://localhost:$BACKEND_PORT/docs"
        echo -e "  - Logs: kubectl logs -n aiq-backend -l app=aiq-circular-detection"
    fi
    
    # Add evaluation info if available
    if [ -n "${EVAL_RESULT+x}" ] && [ -f "${EVALUATION_LOG}" ]; then
        echo ""
        echo -e "${GREEN}Model Evaluation:${NC}"
        echo -e "  - Log: $EVALUATION_LOG"
        echo -e "  - Output: ${EVALUATION_DIR}/output/"
    fi
    
    echo -e "${CYAN}================================================${NC}"
    
    # Interactive mode
    echo -e "\n${YELLOW}Services are running. Press Ctrl+C to stop all services.${NC}"
    echo -e "${BLUE}You can now:${NC}"
    echo -e "  - Test the API at http://localhost:$BACKEND_PORT/docs"
    echo -e "  - View model server at http://localhost:$MODEL_PORT/docs"
    if [[ "$RUN_MODE" == "local" ]]; then
        echo -e "  - Check setup logs in the $LOG_DIR directory"
    else
        echo -e "  - View K8s resources: kubectl get all -n aiq-model-serving"
        echo -e "  - View K8s resources: kubectl get all -n aiq-backend"
    fi
    if [ -n "${EVAL_RESULT+x}" ] && [ -f "${EVALUATION_LOG}" ]; then
        echo -e "  - Review evaluation results in $EVALUATION_LOG"
        echo -e "  - View annotated images in ${EVALUATION_DIR}/output/"
    else
        echo -e "  - Run evaluation: cd services/evaluation && ./run_evaluation.sh"
    fi
    
    # Wait for user to press Ctrl+C
    wait
else
    echo -e "${BOLD}${RED}❌ Tests failed!${NC}"
    echo -e "${YELLOW}Check the logs for more information:${NC}"
    if [[ "$RUN_MODE" == "local" ]]; then
        echo -e "  - Model server setup log: $MODEL_LOG"
        echo -e "  - Backend service setup log: $BACKEND_LOG"
    else
        echo -e "  - Model logs: kubectl logs -n aiq-model-serving -l serving.kserve.io/inferenceservice=aiq-detector"
        echo -e "  - Backend logs: kubectl logs -n aiq-backend -l app=aiq-circular-detection"
    fi
    echo -e "${RED}Stopping services due to test failure...${NC}"
    exit 1
fi 