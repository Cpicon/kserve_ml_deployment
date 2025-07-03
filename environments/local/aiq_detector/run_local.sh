#!/bin/bash
# Script to run the model server locally with Python

# Parse arguments
QUIET_MODE=false
for arg in "$@"; do
    case $arg in
        --quiet)
            QUIET_MODE=true
            shift
            ;;
        *)
            # Keep other arguments for the server
            ;;
    esac
done

echo "🚀 Starting AIQ Detector model server locally..."

# Check if we're in the right directory
if [ ! -f "model.py" ]; then
    echo "❌ Error: model.py not found. Please run this script from the aiq_detector directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    # create a virtual environment in the current directory with python 3.12 using uv
    uv venv --python 3.12
fi

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📦 Installing dependencies..."
# use uv to install the dependencies
if [ "$QUIET_MODE" = true ]; then
    uv pip install -r requirements.txt > /dev/null 2>&1
else
    uv pip install -r requirements.txt
fi

echo "🚀 Starting model server on port 9090..."
echo "📖 Swagger UI will be available at: http://localhost:9090/docs"
if [ "$QUIET_MODE" = true ]; then
    echo "🔇 Server logs suppressed - check logs for full output"
fi
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the model server
if [ "$QUIET_MODE" = true ]; then
    python model.py --model_name=aiq-detector --http_port=9090 > /dev/null 2>&1
else
    python model.py --model_name=aiq-detector --http_port=9090
fi