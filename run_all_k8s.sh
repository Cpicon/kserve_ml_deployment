#!/bin/bash
# run_all_k8s.sh - Deploy and run all services in Kubernetes using Kind
# This script sets up a Kind cluster, installs KServe/Knative, deploys services,
# runs integration tests, and performs evaluation
# Usage: ./run_all_k8s.sh [--clean] [--keep-running]
# Options:
#   --clean         Clean up existing infrastructure before starting
#   --keep-running  Keep the infrastructure running after tests complete

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR="${PROJECT_ROOT}/logs"
CLUSTER_NAME="kserve-deployment"
CLUSTER_LOG="${LOG_DIR}/cluster_setup.log"
DEPLOY_LOG="${LOG_DIR}/k8s_deployment.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    # Delete Kind cluster
    echo -e "${BLUE}Deleting Kind cluster...${NC}"
    kind delete cluster --name ${CLUSTER_NAME} 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Parse arguments
CLEAN_ONLY=false
KEEP_RUNNING=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_ONLY=true
            shift
            ;;
        --keep-running)
            KEEP_RUNNING=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--clean] [--keep-running]"
            exit 1
            ;;
    esac
done

# If --clean flag is passed, just cleanup and exit
if [ "$CLEAN_ONLY" = true ]; then
    cleanup
    exit 0
fi

# Set up trap to cleanup on exit (only if not keeping infrastructure running)
if [ "$KEEP_RUNNING" = false ]; then
    trap cleanup EXIT INT TERM
fi

# Print banner
echo -e "${BOLD}${CYAN}================================================${NC}"
echo -e "${BOLD}${CYAN}   AIQ Circular Detection - K8s Deployment      ${NC}"
echo -e "${BOLD}${CYAN}================================================${NC}"
echo ""

# Step 1: Setup Kind Cluster
echo -e "${BOLD}${BLUE}Step 1: Setting up Kind cluster${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "${YELLOW}Log file: $CLUSTER_LOG${NC}"

cd "${PROJECT_ROOT}/environments/local" || exit 1

# Check if cluster already exists
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    echo -e "${YELLOW}Cluster '${CLUSTER_NAME}' already exists. Deleting...${NC}"
    kind delete cluster --name ${CLUSTER_NAME}
fi

# Run setup_kind.sh
echo -e "${BLUE}Creating Kind cluster...${NC}"
if ./setup_kind.sh 2>&1 | tee "$CLUSTER_LOG"; then
    echo -e "${GREEN}✅ Kind cluster created successfully${NC}"
else
    echo -e "${RED}❌ Failed to create Kind cluster${NC}"
    exit 1
fi

# Step 2: Install KServe and Dependencies
echo -e "\n${BOLD}${BLUE}Step 2: Installing KServe and dependencies${NC}"
echo -e "${BLUE}==========================================${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"

# Run install_kserve_knative.sh with output shown
if ./install_kserve_knative.sh 2>&1 | tee -a "$CLUSTER_LOG"; then
    echo -e "${GREEN}✅ KServe and dependencies installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install KServe and dependencies${NC}"
    exit 1
fi

# Step 3: Setup Ingress Routing
echo -e "\n${BOLD}${BLUE}Step 3: Setting up ingress routing${NC}"
echo -e "${BLUE}===================================${NC}"

# Run setup_ingress_routing.sh with output shown
if ./setup_ingress_routing.sh ${CLUSTER_NAME} 2>&1 | tee -a "$CLUSTER_LOG"; then
    echo -e "${GREEN}✅ Ingress routing configured successfully${NC}"
else
    echo -e "${RED}❌ Failed to setup ingress routing${NC}"
    exit 1
fi

# Step 4: Deploy Model Server
echo -e "\n${BOLD}${BLUE}Step 4: Deploying Model Server${NC}"
echo -e "${BLUE}==============================${NC}"
echo -e "${YELLOW}Log file: $DEPLOY_LOG${NC}"

cd "${PROJECT_ROOT}/environments/local/aiq_detector" || exit 1

echo -e "${BLUE}Building and deploying model server...${NC}"
if ./deploy-docker.sh 2>&1 | tee "$DEPLOY_LOG"; then
    echo -e "${GREEN}✅ Model server deployed successfully${NC}"
    
    # Get the InferenceService URL
    MODEL_URL=$(kubectl get inferenceservice aiq-detector -n aiq-model-serving -o jsonpath='{.status.url}' 2>/dev/null)
    echo -e "${CYAN}Model endpoint: ${MODEL_URL}${NC}"
else
    echo -e "${RED}❌ Failed to deploy model server${NC}"
    exit 1
fi

# Step 5: Deploy Backend Service
echo -e "\n${BOLD}${BLUE}Step 5: Deploying Backend Service${NC}"
echo -e "${BLUE}==================================${NC}"

cd "${PROJECT_ROOT}/environments/local/backend" || exit 1

echo -e "${BLUE}Building and deploying backend service...${NC}"
if ./deploy.sh 2>&1 | tee -a "$DEPLOY_LOG"; then
    echo -e "${GREEN}✅ Backend service deployed successfully${NC}"
else
    echo -e "${RED}❌ Failed to deploy backend service${NC}"
    exit 1
fi

# Step 6: Run Integration Tests
echo -e "\n${BOLD}${CYAN}Step 6: Running Integration Tests${NC}"
echo -e "${CYAN}===================================${NC}"

# Get model service endpoint
MODEL_HOST="aiq-detector.aiq-model-serving.example.com"
MODEL_PORT="80"

# Setup port-forward for backend service
echo -e "${BLUE}Setting up port-forward for backend service...${NC}"
kubectl -n aiq-backend port-forward svc/aiq-circular-detection 8080:80 > /dev/null 2>&1 &
PORT_FORWARD_PID=$!
sleep 5

# Run tests
cd "${PROJECT_ROOT}/environments/local/backend" || exit 1
export BACKEND_URL="http://localhost:8080"
export MODEL_URL="http://${MODEL_HOST}:${MODEL_PORT}"

echo -e "${BLUE}Running integration tests...${NC}"
if ./test-istio.sh; then
    TEST_RESULT=0
    echo -e "${GREEN}✅ Integration tests passed!${NC}"
else
    TEST_RESULT=1
    echo -e "${RED}❌ Integration tests failed!${NC}"
fi

# Kill port-forward
kill $PORT_FORWARD_PID 2>/dev/null || true

# Step 7: Run Model Evaluation (Optional)
echo -e "\n${BOLD}${CYAN}Step 7: Running Model Evaluation${NC}"
echo -e "${CYAN}===================================${NC}"

EVALUATION_DIR="${PROJECT_ROOT}/services/evaluation"
EVALUATION_LOG="${LOG_DIR}/evaluation_k8s.log"

# Check if evaluation dataset exists
if [ -d "${EVALUATION_DIR}/dataset" ] && [ -f "${EVALUATION_DIR}/dataset/_annotations.coco.json" ]; then
    echo -e "${BLUE}Evaluation dataset found. Running model evaluation...${NC}"
    echo -e "${YELLOW}Setting up port-forward for model evaluation...${NC}"
    
    # Get the predictor pod
    PREDICTOR_POD=$(kubectl get pods -n aiq-model-serving -l serving.kserve.io/inferenceservice=aiq-detector -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -n "$PREDICTOR_POD" ]; then
        # Setup port-forward for model
        kubectl -n aiq-model-serving port-forward pod/$PREDICTOR_POD 9090:8080 > /dev/null 2>&1 &
        MODEL_PORT_FORWARD_PID=$!
        sleep 5
        
        cd "${EVALUATION_DIR}" || exit 1
        
        # Override MODEL_SERVER_URL for evaluation
        export MODEL_SERVER_URL="http://localhost:9090"
        
        # Run evaluation
        if ./run_evaluation.sh > "${EVALUATION_LOG}" 2>&1; then
            echo -e "${GREEN}✅ Model evaluation completed successfully!${NC}"
            
            # Extract and display key metrics
            echo -e "\n${BOLD}${CYAN}Model Performance Summary:${NC}"
            echo -e "${CYAN}================================================${NC}"
            
            # Extract F1 Score results
            if grep -A 10 "F1 SCORE RESULTS" "${EVALUATION_LOG}" > /dev/null; then
                echo -e "${GREEN}F1 Score Metrics (Multiple IoU Thresholds):${NC}"
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
            EVAL_RESULT=1
        fi
        
        # Kill model port-forward
        kill $MODEL_PORT_FORWARD_PID 2>/dev/null || true
    else
        echo -e "${RED}❌ Could not find predictor pod for evaluation${NC}"
        EVAL_RESULT=1
    fi
    
    cd "${PROJECT_ROOT}" || exit 1
else
    echo -e "${YELLOW}⚠️  Evaluation dataset not found. Skipping model evaluation.${NC}"
    echo -e "${BLUE}To enable evaluation, add your dataset to: ${EVALUATION_DIR}/dataset/${NC}"
    EVAL_RESULT=0
fi

# Final Summary
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
    echo -e "\n${BOLD}${CYAN}Services Running in Kubernetes:${NC}"
    echo -e "${CYAN}================================================${NC}"
    echo -e "${GREEN}Model Server:${NC}"
    kubectl get inferenceservice aiq-detector -n aiq-model-serving
    echo ""
    echo -e "${GREEN}Backend Service:${NC}"
    kubectl get pods,svc -n aiq-backend -l app=aiq-circular-detection
    echo -e "${CYAN}================================================${NC}"
    
    echo -e "\n${BLUE}To access services:${NC}"
    echo -e "  - Model: kubectl port-forward -n aiq-model-serving svc/aiq-detector-predictor 9090:80"
    echo -e "  - Backend: kubectl port-forward -n aiq-backend svc/aiq-circular-detection 8080:80"
    echo -e "  - View logs: kubectl logs -f -n aiq-backend -l app=aiq-circular-detection"
    
    if [ "$KEEP_RUNNING" = true ]; then
        echo -e "\n${GREEN}Infrastructure will remain running.${NC}"
        echo -e "${YELLOW}To clean up later, run: $0 --clean${NC}"
        echo -e "\n${BLUE}You can also use: ./run_all.sh --mode kind${NC}"
        echo -e "${BLUE}This will reuse the existing infrastructure for testing.${NC}"
    else
        echo -e "\n${YELLOW}Note: The cluster will be automatically cleaned up when this script exits.${NC}"
        echo -e "${YELLOW}To keep the cluster running, use: $0 --keep-running${NC}"
        
        # Give user time to see the message
        sleep 5
    fi
else
    echo -e "${BOLD}${RED}❌ Tests failed!${NC}"
    echo -e "${YELLOW}Check the logs for more information:${NC}"
    echo -e "  - Cluster setup log: $CLUSTER_LOG"
    echo -e "  - Deployment log: $DEPLOY_LOG"
fi 