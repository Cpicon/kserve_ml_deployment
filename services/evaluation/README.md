# Model Evaluation Module

This module provides comprehensive evaluation of the circular object detection model using industry-standard computer vision metrics.

## Overview

The evaluation module (`evaluate_model.py`) tests the performance of the deployed circular object detection model by:
1. Loading a dataset of images with ground truth annotations
2. Sending images to the model server via the FastAPI backend
3. Comparing predictions against ground truth using multiple metrics
4. Generating annotated output images showing both predictions and ground truth

## Metrics Used

### 1. **Jaccard Index (Intersection over Union - IoU)**
The Jaccard Index measures the overlap between predicted and ground truth regions:
- **Formula**: `J = |A ∩ B| / |A ∪ B|`
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: 
  - 1.0 = Perfect overlap
  - 0.7+ = Good detection
  - 0.5+ = Acceptable detection
  - <0.5 = Poor detection

The module computes:
- **Simple Average**: Mean Jaccard Index across all images
- **Weighted Average**: Weighted by number of ground truth objects per image
- **Min/Max Values**: To understand the range of performance

### 2. **F1 Score**
The F1 score balances precision and recall:
- **Formula**: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- **Components**:
  - **Precision**: `TP / (TP + FP)` - How many detected objects are correct?
  - **Recall**: `TP / (TP + FN)` - How many actual objects were detected?
- **IoU Threshold**: 0.5 (configurable) - minimum overlap to consider a match

### 3. **Hungarian Assignment Algorithm**
Used to optimally match predictions to ground truth objects:
- Solves the assignment problem to minimize total "cost" (1 - IoU)
- Ensures each prediction matches at most one ground truth
- Handles cases with different numbers of predictions vs ground truth

## Project Structure

```
services/evaluation/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── evaluate_model.py       # Main evaluation script
├── run_evaluation.sh       # Execution script with environment setup
├── dataset/               # Evaluation dataset (needs to be added)
│   ├── _annotations.coco.json  # COCO format annotations
│   └── *.jpg              # Image files
└── output/                # Generated after running evaluation
    └── *_detected.jpg     # Annotated images with predictions
```

## Setup

### 1. Prepare the Dataset

Place your evaluation dataset in the `dataset/` directory:
```bash
cd services/evaluation
mkdir -p dataset
# Copy your COCO annotations and images
cp /path/to/_annotations.coco.json dataset/
cp /path/to/images/*.jpg dataset/
```

### 2. Install Dependencies

Use the provided script to set up the environment:
```bash
./run_evaluation.sh setup
```

Or manually:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Evaluation

### Prerequisites
Ensure both servers are running:
1. Model server: `cd environments/local/aiq_detector && ./run_local.sh`
2. Backend API: `cd services/backend && ./start-dev-real.sh`

### Execute Evaluation

Run the complete evaluation:
```bash
./run_evaluation.sh
```

Or manually:
```bash
python evaluate_model.py
```

### Configuration

Edit these variables in `evaluate_model.py`:
- `N_IMAGES_TO_PROCESS`: Number of images to evaluate (default: 190)
- `IOU_THRESHOLD`: Minimum IoU for matching (default: 0.5)

## Output

### Console Output
The script provides detailed progress information:
- Per-image results (TP, FP, FN counts)
- IoU values for each matched object
- Jaccard Index computation details

### Final Metrics Summary
```
============================================================
📊 JACCARD INDEX RESULTS
============================================================
🎯 Simple Average Jaccard Index: 0.8951
⚖️  Weighted Average Jaccard Index: 0.8865
📏 Minimum Jaccard Index: 0.0000
📏 Maximum Jaccard Index: 0.9896
🏷️  Total ground truth objects: 520

============================================================
📊 F1 SCORE RESULTS
============================================================
🎯 IoU Threshold: 0.5
✅ Total True Positives (TP): 503
❌ Total False Positives (FP): 40
⭕ Total False Negatives (FN): 17
📊 Precision: 0.9263
📊 Recall: 0.9673
🎯 F1 Score: 0.9464
```

### Annotated Images
Generated in the `output/` directory:
- **Red boxes**: Model predictions
- **Green boxes**: Ground truth annotations
- Includes confidence scores for predictions

## Understanding the Results

### Good Performance Indicators
- F1 Score > 0.9
- Jaccard Index > 0.85
- High recall (few missed objects)
- High precision (few false detections)

### Common Issues
- **Low Recall**: Model missing objects (increase sensitivity)
- **Low Precision**: Too many false positives (increase confidence threshold)
- **Low Jaccard Index**: Poor localization (bounding boxes don't align well)

## Extending the Evaluation

### Adding New Metrics
1. Implement metric calculation in `evaluate_model.py`
2. Add to the metrics summary section
3. Update this README with metric explanation

### Changing IoU Thresholds
To generate precision-recall curves:
```python
for threshold in [0.3, 0.5, 0.7, 0.9]:
    IOU_THRESHOLD = threshold
    # Run evaluation
```

### Batch Evaluation
To evaluate multiple model versions:
```bash
for model in model_v1 model_v2 model_v3; do
    MODEL_NAME=$model ./run_evaluation.sh
done
```

## Troubleshooting

### Common Issues
1. **"FastAPI server is not running"**: Start the backend server first
2. **"Dataset folder not found"**: Ensure dataset is in `services/evaluation/dataset/`
3. **"No predictions returned"**: Check model server is running and accessible
4. **Out of memory**: Reduce `N_IMAGES_TO_PROCESS` or process in batches

### Debug Mode
Enable verbose logging:
```bash
LOG_LEVEL=DEBUG python evaluate_model.py
```

## Contributing

When adding new evaluation capabilities:
1. Update the metrics calculation
2. Add unit tests for new metrics
3. Update this README
4. Ensure backward compatibility

## References

- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
- [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) 