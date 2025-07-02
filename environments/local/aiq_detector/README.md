# AIQ Detector - KServe Deployment

This directory contains the KServe custom predictor deployment for object detection that returns Prediction objects with bounding boxes, centroids, and radii.

## Overview

The AIQ Detector is a custom KServe model that:
- Detects objects (default: coins) in images using zero-shot object detection
- Returns structured Prediction objects containing:
  - Bounding box coordinates
  - Centroid position
  - Radius estimation
  - Detection label
  - Confidence score

## Prerequisites

1. Kind cluster named `kserve-deployment` with KServe installed
2. Docker and pack CLI installed
3. kubectl configured to work with the kind cluster
4. Python 3.11 installed

## Files

- `model.py` - Custom KServe predictor implementation
- `requirements.txt` - Python dependencies
- `aiq-detector.yaml` - KServe InferenceService specification
- `deploy.sh` - Deployment script
- `test_inference.sh` - Test script (automatically starts port-forward if needed)
- `generate_test_image.py` - Generates a test image with a circle
- `start_port_forward.sh` - Starts port-forwarding to the model service
- `stop_port_forward.sh` - Stops port-forwarding
- `check_port_forward.sh` - Checks if port-forwarding is running

## Deployment

1. Make the scripts executable:
```bash
chmod +x deploy.sh test_inference.sh run_deployment.sh
chmod +x start_port_forward.sh stop_port_forward.sh check_port_forward.sh
```

2. Deploy the model:
```bash
./deploy.sh
```

Or use the full deployment script:
```bash
./run_deployment.sh
```

This will:
- Build the Docker image using buildpacks
- Test the image locally to ensure it starts correctly
- Load it into the kind cluster (kserve-deployment)
- Create the namespace `aiq-model-serving`
- Deploy the InferenceService
- Wait for it to be ready

The deployment includes:
- Swagger UI enabled at `/docs`
- Custom response headers with prediction latency
- CORS middleware for cross-origin requests

## Testing

### Quick Test
Run the test script:
```bash
./test_inference.sh
```

This will:
- Check if port-forwarding is running and start it if needed
- Generate a test image (if needed)
- Prepare the test input JSON
- Send an inference request to the deployed model
- Display the results

### Port-Forwarding Management
The port-forwarding is now managed separately for better control:

```bash
# Start port-forwarding (if not already running)
./start_port_forward.sh

# Check if port-forwarding is running
./check_port_forward.sh && echo "Running" || echo "Not running"

# Stop port-forwarding
./stop_port_forward.sh
```

The test script automatically handles port-forwarding, but you can manage it manually if needed.

## API Usage

The model accepts requests in TensorFlow V1 HTTP API format:

```json
{
  "instances": [
    {
      "image": {
        "b64": "<base64-encoded-image>"
      }
    }
  ],
  "parameters": {
    "label": "coin"  // optional, defaults to "coin"
  }
}
```

Response format:
```json
{
  "predictions": [
    {
      "bounding_box": {
        "xmin": 120,
        "ymin": 120,
        "xmax": 280,
        "ymax": 280
      },
      "centroid": {
        "x": 200,
        "y": 200
      },
      "radius": {
        "radius": 80
      },
      "label": "coin",
      "score": 0.95
    }
  ]
}
```

## Using Custom Images

To test with your own images:
```bash
python prepare_test_input.py /path/to/your/image.jpg
./test_inference.sh
```

## Monitoring

Check deployment status:
```bash
kubectl get inferenceservice aiq-detector -n aiq-model-serving
```

Check pod logs:
```bash
kubectl logs -n aiq-model-serving -l serving.kserve.io/inferenceservice=aiq-detector
```

Access Swagger UI:
```bash
# Port-forward to access locally
kubectl port-forward -n aiq-model-serving svc/aiq-detector-predictor-default 8080:80

# Then visit http://localhost:8080/docs in your browser
```

## Cleanup

To remove the deployment:
```bash
kubectl delete inferenceservice aiq-detector -n aiq-model-serving
```

To remove the namespace:
```bash
kubectl delete namespace aiq-model-serving
``` 