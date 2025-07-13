# KServe ML Deployment - Makefile
# Simplified commands for development and deployment workflows

.PHONY: help dev test k8s-setup k8s-test k8s-clean backend model-server eval clean logs lint install-deps

# Default target
help: ## Show this help message
	@echo "KServe ML Deployment - Available Commands:"
	@echo "========================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development Commands

dev: ## Start all services locally (no Kubernetes)
	@echo "🚀 Starting local development environment..."
	./dev-test.sh --mode local

test: ## Run integration tests with automatic service startup
	@echo "🧪 Starting services and running integration tests..."
	@$(MAKE) _start-test-services
	@echo "Running integration tests..."
	@cd services/backend && ./test_full_integration.sh; \
	TEST_RESULT=$$?; \
	echo "Stopping test services..."; \
	if [ -f /tmp/backend_server.pid ]; then kill `cat /tmp/backend_server.pid` 2>/dev/null || true; rm -f /tmp/backend_server.pid; fi; \
	if [ -f /tmp/model_server.pid ]; then kill `cat /tmp/model_server.pid` 2>/dev/null || true; pkill -P `cat /tmp/model_server.pid` 2>/dev/null || true; rm -f /tmp/model_server.pid; fi; \
	lsof -ti:8000 | xargs kill -9 2>/dev/null || true; \
	lsof -ti:9090 | xargs kill -9 2>/dev/null || true; \
	rm -f /tmp/model_server.log /tmp/backend_server.log; \
	echo "✅ Test services stopped"; \
	exit $$TEST_RESULT

test-manual: ## Run integration tests (assumes services are already running)
	@echo "🧪 Running integration tests (manual mode)..."
	cd services/backend && ./test_full_integration.sh

backend: ## Start backend service only (dummy mode)
	@echo "🔧 Starting backend service in dummy mode..."
	cd services/backend && ./start-dev.sh

backend-real: ## Start backend service with real model server
	@echo "🔧 Starting backend service in real mode..."
	cd services/backend && MODE=real ./start-dev.sh

model-server: ## Start model server only (interactive)
	@echo "🤖 Starting model server..."
	@echo "This will block the terminal. Press Ctrl+C to stop."
	@echo "Model will be available at http://localhost:9090"
	@cd environments/local/aiq_detector && ./run_local.sh

model-server-bg: ## Start model server in background
	@echo "🤖 Starting model server in background..."
	@cd environments/local/aiq_detector && ./run_local.sh --quiet > /tmp/model_server.log 2>&1 & echo $$! > /tmp/model_server.pid
	@echo "Waiting for model server to be ready..."
	@for i in {1..30}; do \
		if curl -sf http://localhost:9090/v1/models/aiq-detector >/dev/null 2>&1; then \
			echo "✅ Model server ready at http://localhost:9090"; \
			echo "🔧 PID: `cat /tmp/model_server.pid`"; \
			echo "📋 Logs: tail -f /tmp/model_server.log"; \
			echo "🛑 Stop: make clean"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "❌ Model server failed to start"; \
			if [ -f /tmp/model_server.pid ]; then kill `cat /tmp/model_server.pid` 2>/dev/null || true; rm -f /tmp/model_server.pid; fi; \
			exit 1; \
		fi; \
		echo -n "."; \
		sleep 2; \
	done

##@ Kubernetes Commands

k8s-setup: ## Setup Kind cluster and deploy all services (one-time setup)
	@echo "☸️ Setting up Kubernetes infrastructure..."
	./k8s-deploy.sh --keep-running

k8s-test: ## Run tests on existing Kubernetes infrastructure  
	@echo "☸️ Testing on Kubernetes..."
	./dev-test.sh --mode kind

k8s-clean: ## Clean up Kubernetes cluster and resources
	@echo "🧹 Cleaning up Kubernetes resources..."
	./k8s-deploy.sh --clean

##@ Evaluation and Testing

eval: ## Run model evaluation (requires dataset)
	@echo "📊 Running model evaluation..."
	cd services/evaluation && ./run_evaluation.sh

lint: ## Run code linting
	@echo "🔍 Running linter..."
	cd services/backend && ruff check

pytest: ## Run unit tests
	@echo "🧪 Running unit tests..."
	cd services/backend && pytest

##@ Utility Commands

install-deps: ## Install development dependencies
	@echo "📦 Installing dependencies..."
	cd services/backend && uv sync --dev
	cd environments/local/aiq_detector && uv venv --python 3.12 && uv pip install -r requirements.txt

logs: ## Show service logs (local mode)
	@echo "📋 Showing service logs..."
	@echo "=== Model Server Logs ==="
	@tail -20 logs/model_server.log 2>/dev/null || echo "No model server logs found"
	@echo "\n=== Backend Server Logs ==="
	@tail -20 logs/backend_server.log 2>/dev/null || echo "No backend server logs found"

k8s-logs: ## Show Kubernetes service logs
	@echo "📋 Showing Kubernetes logs..."
	@echo "=== Model Server Logs ==="
	kubectl logs -n aiq-model-serving -l serving.kserve.io/inferenceservice=aiq-detector --tail=20 || echo "Model server not found"
	@echo "\n=== Backend Service Logs ==="
	kubectl logs -n aiq-backend -l app=aiq-circular-detection --tail=20 || echo "Backend service not found"

status: ## Show service status
	@echo "📊 Service Status:"
	@echo "=================="
	@echo "Local services:"
	@lsof -ti:9090 >/dev/null 2>&1 && echo "✅ Model server running (port 9090)" || echo "❌ Model server not running"
	@lsof -ti:8000 >/dev/null 2>&1 && echo "✅ Backend service running (port 8000)" || echo "❌ Backend service not running"
	@echo "\nKubernetes services:"
	@kubectl get inferenceservice aiq-detector -n aiq-model-serving 2>/dev/null && echo "✅ Model server deployed" || echo "❌ Model server not deployed"
	@kubectl get deployment aiq-circular-detection -n aiq-backend 2>/dev/null && echo "✅ Backend service deployed" || echo "❌ Backend service not deployed"

clean: ## Clean up local processes and logs
	@echo "🧹 Cleaning up local environment..."
	@echo "Stopping local services..."
	@if [ -f /tmp/backend_server.pid ]; then kill `cat /tmp/backend_server.pid` 2>/dev/null || true; rm -f /tmp/backend_server.pid; fi
	@if [ -f /tmp/model_server.pid ]; then kill `cat /tmp/model_server.pid` 2>/dev/null || true; pkill -P `cat /tmp/model_server.pid` 2>/dev/null || true; rm -f /tmp/model_server.pid; fi
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:9090 | xargs kill -9 2>/dev/null || true
	@rm -f /tmp/model_server.log /tmp/backend_server.log
	@echo "Cleaning up logs..."
	@rm -f logs/*.log
	@echo "✅ Cleanup complete"

##@ Information Commands

endpoints: ## Show service endpoints
	@echo "🌐 Service Endpoints:"
	@echo "===================="
	@echo "Local Development:"
	@echo "  Backend API:      http://localhost:8000"
	@echo "  API Docs:         http://localhost:8000/docs"
	@echo "  Model Server:     http://localhost:9090"
	@echo "  Model Docs:       http://localhost:9090/docs"
	@echo ""
	@echo "Kubernetes (via port-forward):"
	@echo "  Backend:          kubectl port-forward -n aiq-backend svc/aiq-circular-detection 8080:80"
	@echo "  Model Server:     kubectl port-forward -n aiq-model-serving svc/aiq-detector-predictor 9090:80"

health: ## Check service health
	@echo "🏥 Health Check:"
	@echo "================"
	@echo "Local services:"
	@curl -sf http://localhost:8000/health 2>/dev/null && echo "✅ Backend healthy" || echo "❌ Backend not responding"
	@curl -sf http://localhost:9090/v1/models/aiq-detector 2>/dev/null >/dev/null && echo "✅ Model server healthy" || echo "❌ Model server not responding"

check-deps: ## Check if required tools are installed
	@echo "🔧 Checking dependencies..."
	@command -v python3 >/dev/null 2>&1 && echo "✅ Python 3" || echo "❌ Python 3 not found"
	@command -v uv >/dev/null 2>&1 && echo "✅ uv" || echo "❌ uv not found"
	@command -v kind >/dev/null 2>&1 && echo "✅ kind" || echo "❌ kind not found"
	@command -v kubectl >/dev/null 2>&1 && echo "✅ kubectl" || echo "❌ kubectl not found"
	@command -v jq >/dev/null 2>&1 && echo "✅ jq" || echo "❌ jq not found"

##@ Internal Service Management (Used by other targets)

_start-test-services: ## Start services needed for testing (internal)
	@echo "🔧 Starting test services..."
	@echo "Starting model server in background..."
	@cd environments/local/aiq_detector && ./run_local.sh --quiet > /tmp/model_server.log 2>&1 & echo $$! > /tmp/model_server.pid
	@echo "Waiting for model server to be ready..."
	@for i in {1..30}; do \
		if curl -sf http://localhost:9090/v1/models/aiq-detector >/dev/null 2>&1; then \
			echo "✅ Model server ready"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "❌ Model server failed to start"; \
			if [ -f /tmp/model_server.pid ]; then kill `cat /tmp/model_server.pid` 2>/dev/null || true; rm -f /tmp/model_server.pid; fi; \
			lsof -ti:9090 | xargs kill -9 2>/dev/null || true; \
			exit 1; \
		fi; \
		echo -n "."; \
		sleep 2; \
	done
	@echo "Starting backend service in background..."
	@cd services/backend && MODE=real MODEL_SERVER_URL=http://localhost:9090 ./start-dev.sh --quiet > /tmp/backend_server.log 2>&1 & echo $$! > /tmp/backend_server.pid
	@echo "Waiting for backend service to be ready..."
	@for i in {1..15}; do \
		if curl -sf http://localhost:8000/health >/dev/null 2>&1; then \
			echo "✅ Backend service ready"; \
			break; \
		fi; \
		if [ $$i -eq 15 ]; then \
			echo "❌ Backend service failed to start"; \
			if [ -f /tmp/backend_server.pid ]; then kill `cat /tmp/backend_server.pid` 2>/dev/null || true; rm -f /tmp/backend_server.pid; fi; \
			if [ -f /tmp/model_server.pid ]; then kill `cat /tmp/model_server.pid` 2>/dev/null || true; rm -f /tmp/model_server.pid; fi; \
			lsof -ti:8000 | xargs kill -9 2>/dev/null || true; \
			lsof -ti:9090 | xargs kill -9 2>/dev/null || true; \
			exit 1; \
		fi; \
		echo -n "."; \
		sleep 2; \
	done
	@echo "✅ All test services are ready!"

_stop-test-services: ## Stop test services (internal)
	@echo "🛑 Stopping test services..."
	@if [ -f /tmp/backend_server.pid ]; then \
		echo "Stopping backend service..."; \
		kill `cat /tmp/backend_server.pid` 2>/dev/null || true; \
		rm -f /tmp/backend_server.pid; \
	fi
	@if [ -f /tmp/model_server.pid ]; then \
		echo "Stopping model server..."; \
		kill `cat /tmp/model_server.pid` 2>/dev/null || true; \
		pkill -P `cat /tmp/model_server.pid` 2>/dev/null || true; \
		rm -f /tmp/model_server.pid; \
	fi
	@echo "Force cleanup any remaining processes..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:9090 | xargs kill -9 2>/dev/null || true
	@rm -f /tmp/model_server.log /tmp/backend_server.log
	@echo "✅ Test services stopped"