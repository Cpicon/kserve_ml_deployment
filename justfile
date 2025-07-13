# KServe ML Deployment - Justfile
# Modern command runner with better flag support

# Show available recipes organized by category
default:
    #!/bin/bash
    echo "KServe ML Deployment - Available Commands:"
    echo "========================================="
    echo ""
    echo "🚀 Development Commands:"
    echo "  just dev              # Start all services locally (no Kubernetes)"
    echo "  just dev --k8s        # Start Kubernetes development environment"
    echo "  just dev -k           # Start Kubernetes development (short flag)"
    echo "  just dev --k8s --clean # Clean first, then setup Kubernetes"
    echo "  just test             # Run integration tests (auto-manages services)"
    echo "  just test --manual    # Run tests (assumes services running)"
    echo "  just test --k8s       # Run tests on Kubernetes (auto-setup if needed)"
    echo "  just test -k          # Run tests on Kubernetes (short flag)"
    echo ""
    echo "🔧 Service Management:"
    echo "  just model-server               # Start model server (interactive)"
    echo "  just model-server --background  # Start model server (background)"
    echo "  just model-server -b            # Start model server (background, short)"
    echo "  just backend                    # Start backend (dummy mode)"
    echo "  just backend --real             # Start backend (real mode)"
    echo "  just backend --real --background # Real mode + background"
    echo "  just backend -r -b              # Real mode + background (short)"
    echo "  just clean                      # Clean up all local services"
    echo "  just clean --k8s                # Clean up Kubernetes resources"
    echo "  just clean -k                   # Clean up Kubernetes (short flag)"
    echo "  just clean --k8s --force        # Force delete Kind cluster"
    echo ""
    echo "🧪 Evaluation and Testing:"
    echo "  just eval             # Run model evaluation (requires dataset)"
    echo "  just lint             # Run code linting"
    echo "  just pytest           # Run unit tests only"
    echo ""
    echo "🔧 Utility Commands:"
    echo "  just install-deps     # Install development dependencies"
    echo "  just logs             # Show service logs (local mode)"
    echo "  just k8s-logs         # Show Kubernetes service logs"
    echo "  just status           # Show service status"
    echo ""
    echo "ℹ️ Information Commands:"
    echo "  just endpoints        # Show service endpoints"
    echo "  just health           # Check service health"
    echo "  just check-deps       # Check if required tools are installed"
    echo ""
    echo "💡 Use 'just --list' for a simple alphabetical list"

# Development Commands

# Start development environment (supports --k8s/-k for Kubernetes mode)
dev *flags:
    #!/bin/bash
    if [[ "{{flags}}" == *"--k8s"* ]] || [[ "{{flags}}" == *"-k"* ]]; then
        echo "🚀 Starting Kubernetes development environment..."
        echo "☸️ Setting up Kubernetes infrastructure..."
        if [[ "{{flags}}" == *"--clean"* ]]; then
            ./k8s-deploy.sh --clean
            echo "✅ Cleaned up first, now setting up..."
        fi
        ./k8s-deploy.sh --keep-running
    else
        echo "🚀 Starting local development environment..."
        ./dev-test.sh --mode local
    fi

# Run integration tests (supports --manual/-m, --k8s/-k flags)
test *flags:
    #!/bin/bash
    if [[ "{{flags}}" == *"--k8s"* ]] || [[ "{{flags}}" == *"-k"* ]]; then
        echo "🧪 Running integration tests on Kubernetes..."
        echo "☸️ Checking/setting up Kubernetes infrastructure..."
        # Check if cluster exists, if not set it up
        if ! kind get clusters 2>/dev/null | grep -q "^kserve-deployment$"; then
            echo "🔧 Kubernetes cluster not found, setting up infrastructure..."
            ./k8s-deploy.sh --keep-running
        else
            echo "✅ Kubernetes cluster found, running tests..."
            ./dev-test.sh --mode kind
        fi
    elif [[ "{{flags}}" == *"--manual"* ]] || [[ "{{flags}}" == *"-m"* ]]; then
        echo "🧪 Running integration tests (manual mode - assumes services already running)..."
        cd services/backend && ./test_full_integration.sh
    else
        echo "🧪 Starting services and running integration tests..."
        echo "🔧 Starting test services..."
        echo "Starting model server in background..."
        cd environments/local/aiq_detector && ./run_local.sh --quiet > /tmp/model_server.log 2>&1 & echo $! > /tmp/model_server.pid
        echo "Waiting for model server to be ready..."
        for i in {1..30}; do
            if curl -sf http://localhost:9090/v1/models/aiq-detector >/dev/null 2>&1; then
                echo "✅ Model server ready"
                break
            fi
            if [ $i -eq 30 ]; then
                echo "❌ Model server failed to start"
                if [ -f /tmp/model_server.pid ]; then 
                    kill $(cat /tmp/model_server.pid) 2>/dev/null || true
                    rm -f /tmp/model_server.pid
                fi
                lsof -ti:9090 | xargs kill -9 2>/dev/null || true
                exit 1
            fi
            echo -n "."
            sleep 2
        done
        echo "Starting backend service in background..."
        cd services/backend && MODE=real MODEL_SERVER_URL=http://localhost:9090 ./start-dev.sh --quiet > /tmp/backend_server.log 2>&1 & echo $! > /tmp/backend_server.pid
        echo "Waiting for backend service to be ready..."
        for i in {1..15}; do
            if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
                echo "✅ Backend service ready"
                break
            fi
            if [ $i -eq 15 ]; then
                echo "❌ Backend service failed to start"
                if [ -f /tmp/backend_server.pid ]; then 
                    kill $(cat /tmp/backend_server.pid) 2>/dev/null || true
                    rm -f /tmp/backend_server.pid
                fi
                if [ -f /tmp/model_server.pid ]; then 
                    kill $(cat /tmp/model_server.pid) 2>/dev/null || true
                    rm -f /tmp/model_server.pid
                fi
                lsof -ti:8000 | xargs kill -9 2>/dev/null || true
                lsof -ti:9090 | xargs kill -9 2>/dev/null || true
                exit 1
            fi
            echo -n "."
            sleep 2
        done
        echo "✅ All test services are ready!"
        echo "Running integration tests..."
        cd services/backend && ./test_full_integration.sh
        TEST_RESULT=$?
        echo "Stopping test services..."
        if [ -f /tmp/backend_server.pid ]; then 
            kill $(cat /tmp/backend_server.pid) 2>/dev/null || true
            rm -f /tmp/backend_server.pid
        fi
        if [ -f /tmp/model_server.pid ]; then 
            kill $(cat /tmp/model_server.pid) 2>/dev/null || true
            pkill -P $(cat /tmp/model_server.pid) 2>/dev/null || true
            rm -f /tmp/model_server.pid
        fi
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        lsof -ti:9090 | xargs kill -9 2>/dev/null || true
        rm -f /tmp/model_server.log /tmp/backend_server.log
        echo "✅ Test services stopped"
        exit $TEST_RESULT
    fi

# Start backend service (supports --real flag for real mode, --background or -b for background)
backend *flags:
    #!/bin/bash
    REAL_MODE=""
    BACKGROUND_MODE=""
    
    if [[ "{{flags}}" == *"--real"* ]] || [[ "{{flags}}" == *"-r"* ]]; then
        REAL_MODE="real"
        echo "🔧 Starting backend service in real mode..."
    else
        echo "🔧 Starting backend service in dummy mode..."
    fi
    
    if [[ "{{flags}}" == *"--background"* ]] || [[ "{{flags}}" == *"-b"* ]]; then
        BACKGROUND_MODE="true"
        if [ "$REAL_MODE" = "real" ]; then
            echo "💡 Make sure model server is running: just model-server -b"
            cd services/backend && MODE=real ./start-dev.sh --quiet > /tmp/backend_server.log 2>&1 & echo $! > /tmp/backend_server.pid
        else
            cd services/backend && ./start-dev.sh --quiet > /tmp/backend_server.log 2>&1 & echo $! > /tmp/backend_server.pid
        fi
        echo "Waiting for backend service to be ready..."
        for i in {1..15}; do
            if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
                echo "✅ Backend service ready at http://localhost:8000"
                echo "🔧 PID: $(cat /tmp/backend_server.pid)"
                echo "📋 Logs: tail -f /tmp/backend_server.log"
                echo "🛑 Stop: just clean"
                break
            fi
            if [ $i -eq 15 ]; then
                echo "❌ Backend service failed to start"
                if [ -f /tmp/backend_server.pid ]; then 
                    kill $(cat /tmp/backend_server.pid) 2>/dev/null || true
                    rm -f /tmp/backend_server.pid
                fi
                exit 1
            fi
            echo -n "."
            sleep 2
        done
    else
        echo "This will block the terminal. Press Ctrl+C to stop."
        echo "Backend will be available at http://localhost:8000"
        echo "API Docs: http://localhost:8000/docs"
        echo "💡 Use '--background' or '-b' for background mode"
        if [ "$REAL_MODE" = "real" ]; then
            echo "💡 Make sure model server is running: just model-server -b"
            cd services/backend && MODE=real ./start-dev.sh
        else
            cd services/backend && ./start-dev.sh
        fi
    fi

# Start model server (supports --background or -b flag)
model-server *flags:
    #!/bin/bash
    if [[ "{{flags}}" == *"--background"* ]] || [[ "{{flags}}" == *"-b"* ]]; then
        echo "🤖 Starting model server in background..."
        cd environments/local/aiq_detector && ./run_local.sh --quiet > /tmp/model_server.log 2>&1 & echo $! > /tmp/model_server.pid
        echo "Waiting for model server to be ready..."
        for i in {1..30}; do
            if curl -sf http://localhost:9090/v1/models/aiq-detector >/dev/null 2>&1; then
                echo "✅ Model server ready at http://localhost:9090"
                echo "🔧 PID: $(cat /tmp/model_server.pid)"
                echo "📋 Logs: tail -f /tmp/model_server.log"
                echo "🛑 Stop: just clean"
                break
            fi
            if [ $i -eq 30 ]; then
                echo "❌ Model server failed to start"
                if [ -f /tmp/model_server.pid ]; then 
                    kill $(cat /tmp/model_server.pid) 2>/dev/null || true
                    rm -f /tmp/model_server.pid
                fi
                exit 1
            fi
            echo -n "."
            sleep 2
        done
    else
        echo "🤖 Starting model server (interactive mode)..."
        echo "This will block the terminal. Press Ctrl+C to stop."
        echo "Model will be available at http://localhost:9090"
        echo "💡 Use 'just model-server --background' or 'just model-server -b' for background mode"
        cd environments/local/aiq_detector && ./run_local.sh
    fi

# Service Management

# Clean up services and resources (supports --k8s/-k for Kubernetes cleanup)
clean *flags:
    #!/bin/bash
    if [[ "{{flags}}" == *"--k8s"* ]] || [[ "{{flags}}" == *"-k"* ]]; then
        echo "🧹 Cleaning up Kubernetes resources..."
        if [[ "{{flags}}" == *"--force"* ]]; then
            echo "💀 Force cleanup mode..."
            kind delete cluster --name kserve-deployment 2>/dev/null || true
        else
            ./k8s-deploy.sh --clean
        fi
    else
        echo "🧹 Cleaning up local environment..."
        echo "Stopping local services..."
        if [ -f /tmp/backend_server.pid ]; then 
            kill $(cat /tmp/backend_server.pid) 2>/dev/null || true
            rm -f /tmp/backend_server.pid
        fi
        if [ -f /tmp/model_server.pid ]; then 
            kill $(cat /tmp/model_server.pid) 2>/dev/null || true
            pkill -P $(cat /tmp/model_server.pid) 2>/dev/null || true
            rm -f /tmp/model_server.pid
        fi
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        lsof -ti:9090 | xargs kill -9 2>/dev/null || true
        rm -f /tmp/model_server.log /tmp/backend_server.log
        echo "Cleaning up logs..."
        rm -f logs/*.log 2>/dev/null || true
        echo "✅ Cleanup complete"
    fi

# Legacy Kubernetes Commands (deprecated - use dev/test/clean with --k8s flag)

# Setup Kind cluster and deploy all services (deprecated: use 'just dev --k8s')
k8s-setup *flags:
    #!/bin/bash
    echo "⚠️  DEPRECATED: Use 'just dev --k8s' instead"
    echo "☸️ Setting up Kubernetes infrastructure..."
    if [[ "{{flags}}" == *"--clean"* ]]; then
        ./k8s-deploy.sh --clean
        echo "✅ Cleaned up first, now setting up..."
    fi
    ./k8s-deploy.sh --keep-running

# Run tests on existing Kubernetes infrastructure (deprecated: use 'just test --k8s')
k8s-test *flags:
    #!/bin/bash
    echo "⚠️  DEPRECATED: Use 'just test --k8s' instead"
    echo "☸️ Testing on Kubernetes..."
    if [[ "{{flags}}" == *"--setup"* ]]; then
        echo "🔧 Setting up infrastructure first..."
        ./k8s-deploy.sh --keep-running
    fi
    ./dev-test.sh --mode kind

# Clean up Kubernetes cluster and resources (deprecated: use 'just clean --k8s')
k8s-clean *flags:
    #!/bin/bash
    echo "⚠️  DEPRECATED: Use 'just clean --k8s' instead"
    echo "🧹 Cleaning up Kubernetes resources..."
    if [[ "{{flags}}" == *"--force"* ]]; then
        echo "💀 Force cleanup mode..."
        kind delete cluster --name kserve-deployment 2>/dev/null || true
    else
        ./k8s-deploy.sh --clean
    fi

# Evaluation and Testing

# Run model evaluation (requires dataset in services/evaluation/dataset/)
eval:
    #!/bin/bash
    echo "📊 Running model evaluation..."
    cd services/evaluation && ./run_evaluation.sh

# Run code linting
lint:
    #!/bin/bash
    echo "🔍 Running linter..."
    cd services/backend && ruff check

# Run unit tests only
pytest:
    #!/bin/bash
    echo "🧪 Running unit tests..."
    cd services/backend && pytest

# Utility Commands

# Install development dependencies
install-deps:
    #!/bin/bash
    echo "📦 Installing dependencies..."
    cd services/backend && uv sync --dev
    cd environments/local/aiq_detector && uv venv --python 3.12 && uv pip install -r requirements.txt

# Show service logs (local mode)
logs:
    #!/bin/bash
    echo "📋 Showing service logs..."
    echo "=== Model Server Logs ==="
    tail -20 logs/model_server.log 2>/dev/null || echo "No model server logs found"
    echo ""
    echo "=== Backend Server Logs ==="
    tail -20 logs/backend_server.log 2>/dev/null || echo "No backend server logs found"

# Show Kubernetes service logs
k8s-logs:
    #!/bin/bash
    echo "📋 Showing Kubernetes logs..."
    echo "=== Model Server Logs ==="
    kubectl logs -n aiq-model-serving -l serving.kserve.io/inferenceservice=aiq-detector --tail=20 || echo "Model server not found"
    echo ""
    echo "=== Backend Service Logs ==="
    kubectl logs -n aiq-backend -l app=aiq-circular-detection --tail=20 || echo "Backend service not found"

# Show service status
status:
    #!/bin/bash
    echo "📊 Service Status:"
    echo "=================="
    echo "Local services:"
    lsof -ti:9090 >/dev/null 2>&1 && echo "✅ Model server running (port 9090)" || echo "❌ Model server not running"
    lsof -ti:8000 >/dev/null 2>&1 && echo "✅ Backend service running (port 8000)" || echo "❌ Backend service not running"
    echo ""
    echo "Kubernetes services:"
    kubectl get inferenceservice aiq-detector -n aiq-model-serving 2>/dev/null && echo "✅ Model server deployed" || echo "❌ Model server not deployed"
    kubectl get deployment aiq-circular-detection -n aiq-backend 2>/dev/null && echo "✅ Backend service deployed" || echo "❌ Backend service not deployed"

# Information Commands

# Show service endpoints
endpoints:
    #!/bin/bash
    echo "🌐 Service Endpoints:"
    echo "===================="
    echo "Local Development:"
    echo "  Backend API:      http://localhost:8000"
    echo "  API Docs:         http://localhost:8000/docs"
    echo "  Model Server:     http://localhost:9090"
    echo "  Model Docs:       http://localhost:9090/docs"
    echo ""
    echo "Kubernetes (via port-forward):"
    echo "  Backend:          kubectl port-forward -n aiq-backend svc/aiq-circular-detection 8080:80"
    echo "  Model Server:     kubectl port-forward -n aiq-model-serving svc/aiq-detector-predictor 9090:80"

# Check service health
health:
    #!/bin/bash
    echo "🏥 Health Check:"
    echo "================"
    echo "Local services:"
    curl -sf http://localhost:8000/health 2>/dev/null && echo "✅ Backend healthy" || echo "❌ Backend not responding"
    curl -sf http://localhost:9090/v1/models/aiq-detector 2>/dev/null >/dev/null && echo "✅ Model server healthy" || echo "❌ Model server not responding"

# Check if required tools are installed
check-deps:
    #!/bin/bash
    echo "🔧 Checking dependencies..."
    command -v python3 >/dev/null 2>&1 && echo "✅ Python 3" || echo "❌ Python 3 not found"
    command -v uv >/dev/null 2>&1 && echo "✅ uv" || echo "❌ uv not found"
    command -v kind >/dev/null 2>&1 && echo "✅ kind" || echo "❌ kind not found"
    command -v kubectl >/dev/null 2>&1 && echo "✅ kubectl" || echo "❌ kubectl not found"
    command -v jq >/dev/null 2>&1 && echo "✅ jq" || echo "❌ jq not found"
    command -v just >/dev/null 2>&1 && echo "✅ just" || echo "❌ just not found"