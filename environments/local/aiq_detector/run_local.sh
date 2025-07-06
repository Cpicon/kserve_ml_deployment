#!/bin/bash
# Script to run the model server locally with Python

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
uv pip install -r requirements.txt

# run the model server
uv run model.py --model_name=aiq-detector --http_port=9090

echo "🚀 Starting model server on port 9090..."
echo "📖 Swagger UI will be available at: http://localhost:9090/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the model server
python model.py --model_name=aiq-detector --http_port=9090