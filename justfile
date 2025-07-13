# KServe ML Deployment - Justfile
# Modern command runner with better flag support

# Show available recipes
default:
    @just --list

# Start all services locally (no Kubernetes)
dev:
    #!/bin/bash
    echo "🚀 Starting local development environment..."
    ./dev-test.sh --mode local

# Run integration tests (supports --manual flag to skip auto service management)
test *flags:
    #!/bin/bash
    if [[ "{{flags}}" == *"--manual"* ]] || [[ "{{flags}}" == *"-m"* ]]; then
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

# Clean up local processes and logs
clean:
    #!/bin/bash
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

# Setup Kind cluster and deploy all services (one-time setup)
k8s-setup *flags:
    #!/bin/bash
    echo "☸️ Setting up Kubernetes infrastructure..."
    if [[ "{{flags}}" == *"--clean"* ]]; then
        ./k8s-deploy.sh --clean
        echo "✅ Cleaned up first, now setting up..."
    fi
    ./k8s-deploy.sh --keep-running

# Run tests on existing Kubernetes infrastructure
k8s-test *flags:
    #!/bin/bash
    echo "☸️ Testing on Kubernetes..."
    if [[ "{{flags}}" == *"--setup"* ]]; then
        echo "🔧 Setting up infrastructure first..."
        ./k8s-deploy.sh --keep-running
    fi
    ./dev-test.sh --mode kind

# Clean up Kubernetes cluster and resources
k8s-clean *flags:
    #!/bin/bash
    echo "🧹 Cleaning up Kubernetes resources..."
    if [[ "{{flags}}" == *"--force"* ]]; then
        echo "💀 Force cleanup mode..."
        kind delete cluster --name kserve-deployment 2>/dev/null || true
    else
        ./k8s-deploy.sh --clean
    fi