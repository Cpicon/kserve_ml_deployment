from transformers import pipeline
import os
import numpy as np
from PIL import Image, ImageDraw
import json
from scipy.optimize import linear_sum_assignment
from pydantic import BaseModel
import httpx
import asyncio
from typing import Optional

# Set the path to the COCO JSON file, use current directory
# and ensure it is relative to the script location
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
COCO_JSON_PATH = os.path.join(SCRIPT_DIR, "dataset", "_annotations.coco.json")
# COCO_JSON_PATH = "dataset/_annotations.coco.json"
N_IMAGES_TO_PROCESS = 190
# Iterate over multiple IoU thresholds to compute precision-recall curves
IOU_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]  # Multiple thresholds for evaluation


class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class Centroid(BaseModel):
    x: int
    y: int


class Radius(BaseModel):
    radius: int


class Prediction(BaseModel):
    bounding_box: BoundingBox
    centroid: Centroid
    radius: Radius
    label: str
    score: float


class Assignment(BaseModel):
    prediction_bounding_box: BoundingBox
    ground_truth_bounding_box: BoundingBox
    iou: float


class HungarianAssignment(BaseModel):
    tp_count: int
    fp_count: int
    fn_count: int
    assignments: list[Assignment]


class Metrics(BaseModel):
    precision: float
    recall: float
    f1_score: float


class JaccardIndex(BaseModel):
    simple_average: float
    weighted_average: float
    min_jaccard_index: float
    max_jaccard_index: float


def load_coco_json(path) -> dict[str, list[BoundingBox]]:
    with open(path, 'r') as f:
        coco_data = json.load(f)

    image_to_annotations = {}
    for image in coco_data['images']:
        annotations = [
            annotation for annotation in coco_data['annotations'] if
            annotation['image_id'] == image['id'] and annotation['category_id'] == 1
        ]
        bounding_boxes = []
        for annotation in annotations:
            xmin, ymin, width, height = annotation['bbox']
            xmax = xmin + width
            ymax = ymin + height
            bounding_boxes.append(BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
        image_to_annotations[image['file_name']] = bounding_boxes
    return image_to_annotations


def compute_jaccard_index_union(predictions: list[Prediction], image_annotations: list[BoundingBox], image_width: int,
                                image_height: int) -> float:
    """
    Compute Jaccard Index between union of predicted bounding boxes and union of ground truth bounding boxes.

    Args:
        predictions: List of Prediction objects
        image_annotations: List of BoundingBox objects
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        Jaccard Index (float between 0 and 1)
    """

    # Create binary masks for the image
    pred_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    gt_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Fill predicted bounding boxes mask
    print(f"    📦 Processing {len(predictions)} predictions for Jaccard Index...")
    for i, bounding_box in enumerate(predictions):
        box = bounding_box.bounding_box
        xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax

        # Ensure coordinates are within image bounds
        xmin = max(0, min(int(xmin), image_width - 1))
        ymin = max(0, min(int(ymin), image_height - 1))
        xmax = max(0, min(int(xmax), image_width))
        ymax = max(0, min(int(ymax), image_height))

        # Fill the mask region
        pred_mask[ymin:ymax, xmin:xmax] = 1
        print(f"      🎯 Prediction {i + 1}: [{xmin}, {ymin}, {xmax}, {ymax}]")

    # Fill ground truth bounding boxes mask
    gt_count = 0
    for bounding_box in image_annotations:
        # Ensure coordinates are within image bounds
        xmin = max(0, min(int(bounding_box.xmin), image_width - 1))
        ymin = max(0, min(int(bounding_box.ymin), image_height - 1))
        xmax = max(0, min(int(bounding_box.xmax), image_width))
        ymax = max(0, min(int(bounding_box.ymax), image_height))

        # Fill the mask region
        gt_mask[ymin:ymax, xmin:xmax] = 1
        gt_count += 1
        print(f"      ✅ Ground Truth {gt_count}: [{xmin}, {ymin}, {xmax}, {ymax}]")

    # Compute intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    # Compute Jaccard Index
    if union == 0:
        jaccard_index = 1.0 if intersection == 0 else 0.0
    else:
        jaccard_index = intersection / union

    print(f"    📊 Intersection pixels: {intersection}")
    print(f"    📊 Union pixels: {union}")
    print(f"    🎯 Jaccard Index: {jaccard_index:.4f}")

    return jaccard_index


def get_predictions(image, detector, labels) -> list[Prediction]:
    predictions = detector(
        image,
        candidate_labels=labels,
    )
    return [
        Prediction(
            bounding_box=BoundingBox(
                xmin=prediction["box"]["xmin"],
                ymin=prediction["box"]["ymin"],
                xmax=prediction["box"]["xmax"],
                ymax=prediction["box"]["ymax"]),
            centroid=get_centroid_and_radius(
                BoundingBox(xmin=prediction["box"]["xmin"],
                            ymin=prediction["box"]["ymin"],
                            xmax=prediction["box"]["xmax"],
                            ymax=prediction["box"]["ymax"]
                            )
            )[0],
            radius=get_centroid_and_radius(
                BoundingBox(xmin=prediction["box"]["xmin"],
                            ymin=prediction["box"]["ymin"],
                            xmax=prediction["box"]["xmax"],
                            ymax=prediction["box"]["ymax"]
                            )
            )[1],
            label=labels[0],
            score=prediction["score"]
        ) for prediction in predictions
    ]


def get_centroid_and_radius(bounding_box: BoundingBox) -> tuple[Centroid, Radius]:
    xmin, ymin, xmax, ymax = bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    radius = (xmax - xmin) / 2
    return Centroid(x=int(x_center), y=int(y_center)), Radius(radius=int(radius))


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: BoundingBox
        box2: BoundingBox

    Returns:
        IoU value (float between 0 and 1)
    """
    # Handle different box formats
    x1_min, y1_min, x1_max, y1_max = box1.xmin, box1.ymin, box1.xmax, box1.ymax

    x2_min, y2_min, x2_max, y2_max = box2.xmin, box2.ymin, box2.xmax, box2.ymax

    # Compute intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def hungarian_assignment(predictions: list[Prediction], ground_truths: list[BoundingBox],
                         iou_threshold: float = 0.5) -> HungarianAssignment:
    """
    Use Hungarian algorithm to assign predictions to ground truths based on IoU.

    Args:
        predictions: List of prediction objects with 'box' attribute
        ground_truths: List of ground truth annotations with 'bbox' attribute
        iou_threshold: Minimum IoU for a valid match

    Returns:
        tp_count: Number of true positives
        fp_count: Number of false positives
        fn_count: Number of false negatives
        assignments: List of (pred_idx, gt_idx, iou) for matched pairs
    """
    if len(predictions) == 0 and len(ground_truths) == 0:
        return HungarianAssignment(tp_count=0, fp_count=0, fn_count=0, assignments=[])

    if len(predictions) == 0:
        return HungarianAssignment(tp_count=0, fp_count=0, fn_count=len(ground_truths), assignments=[])

    if len(ground_truths) == 0:
        return HungarianAssignment(tp_count=0, fp_count=len(predictions), fn_count=0, assignments=[])

    # Create cost matrix (using negative IoU as cost since Hungarian minimizes)
    cost_matrix = np.zeros((len(predictions), len(ground_truths)))
    iou_matrix = np.zeros((len(predictions), len(ground_truths)))

    for i, prediction in enumerate(predictions):
        pred_box = prediction.bounding_box
        for j, gt_annotation in enumerate(ground_truths):
            gt_box = gt_annotation
            iou = compute_iou(pred_box, gt_box)
            iou_matrix[i, j] = iou
            # Use negative IoU as cost (Hungarian minimizes cost)
            cost_matrix[i, j] = 1.0 - iou

    # Apply Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Count TP, FP, FN based on IoU threshold
    tp_count = 0
    fp_count = 0
    fn_count = 0
    assignments = []

    # Track which predictions and ground truths are matched
    matched_predictions = set()
    matched_ground_truths = set()

    # Check assignments from Hungarian algorithm
    # The Hungarian algorithm finds the optimal assignment that minimizes total cost (maximizes total IoU)
    # We only count assignments as true positives if they meet the IoU threshold
    # This ensures high-quality matches while allowing low-IoU pairs to remain unmatched
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        if iou >= iou_threshold:
            # Valid match - this is a true positive
            tp_count += 1
            matched_predictions.add(pred_idx)
            matched_ground_truths.add(gt_idx)
            assignments.append(Assignment(prediction_bounding_box=predictions[pred_idx].bounding_box,
                                          ground_truth_bounding_box=ground_truths[gt_idx], iou=float(iou)))
        # If IoU is below threshold, both prediction and GT remain unmatched
        # This correctly handles cases where Hungarian assigns low-quality matches

    # Count false positives (unmatched predictions)
    fp_count = len(predictions) - len(matched_predictions)

    # Count false negatives (unmatched ground truths with category_id == 1)
    fn_count = len(ground_truths) - len(matched_ground_truths)

    return HungarianAssignment(tp_count=tp_count, fp_count=fp_count, fn_count=fn_count, assignments=assignments)


def compute_metrics(tp_count: int, fp_count: int, fn_count: int) -> Metrics:
    """
    Compute precision, recall, and F1 score from TP, FP, FN counts.

    Args:
        tp_count: Total true positives
        fp_count: Total false positives
        fn_count: Total false negatives

    Returns:
        Metrics object with precision, recall, and F1 score
    """
    # Compute precision
    if tp_count + fp_count == 0:
        precision = 0.0
    else:
        precision = tp_count / (tp_count + fp_count)

    # Compute recall
    if tp_count + fn_count == 0:
        recall = 0.0
    else:
        recall = tp_count / (tp_count + fn_count)

    # Compute F1 score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return Metrics(precision=precision, recall=recall, f1_score=f1_score)


async def upload_image_to_server(image_path: str, client: httpx.AsyncClient) -> Optional[dict]:
    """Upload an image to the FastAPI server and get predictions.
    
    Args:
        image_path: Path to the image file
        client: HTTP client instance
        
    Returns:
        Dictionary with image_id and detections, or None if image already exists
    """
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        try:
            response = await client.post(
                'http://localhost:8000/images/',
                files=files
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                # Image already exists, extract the image ID from error message
                error_detail = e.response.json().get('detail', '')
                if 'Image already exists with ID:' in error_detail:
                    image_id = error_detail.split(': ')[-1]
                    return {'image_id': image_id, 'already_exists': True}
            raise


async def get_existing_predictions(image_id: str, client: httpx.AsyncClient) -> list[dict]:
    """Get predictions for an existing image from the server.
    
    Args:
        image_id: The SHA-256 hash ID of the image
        client: HTTP client instance
        
    Returns:
        List of circular object predictions
    """
    response = await client.get(f'http://localhost:8000/images/{image_id}/objects')
    response.raise_for_status()
    data = response.json()
    
    objects = []
    for obj_summary in data['objects']:
        # Get detailed object information
        detail_response = await client.get(
            f'http://localhost:8000/images/{image_id}/objects/{obj_summary["object_id"]}'
        )
        detail_response.raise_for_status()
        objects.append(detail_response.json())
    
    return objects


def convert_server_predictions_to_predictions(server_objects: list[dict]) -> list[Prediction]:
    """Convert server response objects to Prediction objects.
    
    Args:
        server_objects: List of object details from server
        
    Returns:
        List of Prediction objects
    """
    predictions = []
    for obj in server_objects:
        bbox = obj['bbox']
        centroid = obj['centroid']
        radius = obj['radius']
        
        predictions.append(Prediction(
            bounding_box=BoundingBox(
                xmin=bbox[0],
                ymin=bbox[1],
                xmax=bbox[2],
                ymax=bbox[3]
            ),
            centroid=Centroid(x=centroid[0], y=centroid[1]),
            radius=Radius(radius=radius),
            label="coin",
            score=1.0  # Server doesn't return confidence scores
        ))
    
    return predictions


async def main():
    """Main async function to process images through the FastAPI server."""
    print("Starting object detection process...")
    
    # Check FastAPI server health
    try:
        async with httpx.AsyncClient() as client:
            health_response = await client.get('http://localhost:8000/health')
            health_response.raise_for_status()
            print("✅ FastAPI server is running!")
            
            config_response = await client.get('http://localhost:8000/config')
            config = config_response.json()
            print(f"Server mode: {config['mode']}")
            print(f"Model name: {config['model_name']}")
    except Exception as e:
        print(f"❌ Error: FastAPI server is not running at http://localhost:8000")
        print(f"   {e}")
        print("Please start the servers:")
        print("1. cd environments/local/aiq_detector && ./run_local.sh")
        print("2. cd services/backend && ./start-dev-real.sh")
        exit(1)

    # Create an output directory if it doesn't exist
    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ready: {output_dir}")

    # Check if dataset folder exists
    dataset_dir = os.path.join(SCRIPT_DIR, "dataset")
    if not os.path.exists(dataset_dir):
        print("ERROR: 'dataset' folder not found!")
        print("Please create a 'dataset' folder in the evaluation directory and add .jpg images to process.")
        exit(1)

    # Load images from dataset folder
    print("Loading COCO JSON file...")
    image_to_bounding_boxes = load_coco_json(COCO_JSON_PATH)

    if len(image_to_bounding_boxes.keys()) <= 0:
        print("WARNING: No .jpg images found in 'dataset' folder!")
        print("Please add .jpg images to the 'dataset' folder.")
        exit(1)

    print(f"Found {len(image_to_bounding_boxes.keys())} images to process")

    images_to_process = []
    image_names = []  # Keep track of original filenames
    image_paths = []

    counter = 0
    for image_file in image_to_bounding_boxes:
        image_path = os.path.join(dataset_dir, image_file)
        print(f"  - Loading: {image_file}")
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        images_to_process.append(image)
        image_names.append(image_file)
        image_paths.append(image_path)
        counter += 1
        if counter >= N_IMAGES_TO_PROCESS:
            break

    # Process images
    print(f"\nStarting object detection on {len(images_to_process)} images...")
    jaccard_indices = []  # Store Jaccard indices for all images
    gt_counts = []  # Store ground truth counts for each image

    # Store metrics for each IoU threshold
    threshold_metrics = {}
    for threshold in IOU_THRESHOLDS:
        threshold_metrics[threshold] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0
        }

    print(f"🎯 Using IoU thresholds: {IOU_THRESHOLDS}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, (image, image_name, image_path) in enumerate(zip(images_to_process, image_names, image_paths)):
            print(f"\nProcessing image {idx + 1}/{len(images_to_process)}: {image_name}")

            # Get image dimensions
            image_width, image_height = image.size
            print(f"  Image dimensions: {image_width} x {image_height}")

            # Count ground truth bounding boxes for this image
            gt_count = len(image_to_bounding_boxes[image_name])
            gt_counts.append(gt_count)
            print(f"  Ground truth objects: {gt_count}")

            # Upload image to server and get predictions
            print("  Uploading image to server...")
            try:
                result = await upload_image_to_server(image_path, client)
                
                if result.get('already_exists'):
                    print(f"  ⚠️  Image already exists in database with ID: {result['image_id']}")
                    print("  Fetching existing predictions from database...")
                    server_objects = await get_existing_predictions(result['image_id'], client)
                    predictions = convert_server_predictions_to_predictions(server_objects)
                    print(f"  Retrieved {len(predictions)} existing predictions from database")
                else:
                    print(f"  ✅ Image uploaded successfully with ID: {result['image_id']}")
                    # Extract predictions from the upload response
                    if result.get('detection') and result['detection'].get('detections'):
                        detections = result['detection']['detections']
                        predictions = []
                        for det in detections:
                            predictions.append(Prediction(
                                bounding_box=BoundingBox(
                                    xmin=det['bbox'][0],
                                    ymin=det['bbox'][1],
                                    xmax=det['bbox'][2],
                                    ymax=det['bbox'][3]
                                ),
                                centroid=Centroid(x=det['centroid']['x'], y=det['centroid']['y']),
                                radius=Radius(radius=det['radius']),
                                label="coin",
                                score=1.0
                            ))
                        print(f"  Model server returned {len(predictions)} predictions")
                    else:
                        predictions = []
                        print("  No predictions returned from model server")
                        
            except Exception as e:
                print(f"  ❌ Error processing image: {e}")
                predictions = []

            print(f"  Found {len(predictions)} objects")

            # Compute Hungarian assignment and F1 metrics for this image at each threshold
            print("  Computing Hungarian assignments for different IoU thresholds...")
            
            for iou_threshold in IOU_THRESHOLDS:
                assignment = hungarian_assignment(
                    predictions,
                    image_to_bounding_boxes[image_name],
                    iou_threshold
                )

                print(f"    IoU={iou_threshold}: TP: {assignment.tp_count}, FP: {assignment.fp_count}, FN: {assignment.fn_count}")

                # Add to totals for this threshold
                threshold_metrics[iou_threshold]['total_tp'] += assignment.tp_count
                threshold_metrics[iou_threshold]['total_fp'] += assignment.fp_count
                threshold_metrics[iou_threshold]['total_fn'] += assignment.fn_count

            # Show assignments for the default threshold (0.5)
            default_assignment = hungarian_assignment(
                predictions,
                image_to_bounding_boxes[image_name],
                0.5
            )
            if default_assignment.assignments:
                print(f"    Assignments:")
                for assign in default_assignment.assignments:
                    print(
                        f"      Prediction {assign.prediction_bounding_box} -> GT {assign.ground_truth_bounding_box} (IoU: {assign.iou:.3f})")

            # Compute Jaccard Index
            print("  Computing Jaccard Index...")
            jaccard_index = compute_jaccard_index_union(
                predictions,
                image_to_bounding_boxes[image_name],
                image_width,
                image_height
            )
            jaccard_indices.append(jaccard_index)

            # Create a copy for drawing (to preserve original)
            image_with_boxes = image.copy()
            draw = ImageDraw.Draw(image_with_boxes)

            # Draw bounding boxes
            for i, prediction in enumerate(predictions):
                box = prediction.bounding_box
                label = prediction.label
                score = prediction.score

                print(f"    Object {i + 1}: {label} (confidence: {score:.3f})")

                xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
                # Draw predicted bounding box
                draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=3)
                draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="red")
                
            # Draw ground truth bounding box
            for bounding_box in image_to_bounding_boxes[image_name]:
                xmin, ymin, xmax, ymax = bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax
                draw.rectangle((xmin, ymin, xmax, ymax), outline="green", width=3)
                draw.text((xmin, ymin), f"GT", fill="green")
                
            # Save individual image
            output_filename = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_detected.jpg")
            image_with_boxes.save(output_filename)
            print(f"  Saved annotated image to: {output_filename}")

    # Compute F1 metrics for each threshold
    all_metrics = {}
    for threshold in IOU_THRESHOLDS:
        metrics = compute_metrics(
            threshold_metrics[threshold]['total_tp'],
            threshold_metrics[threshold]['total_fp'],
            threshold_metrics[threshold]['total_fn']
        )
        all_metrics[threshold] = metrics

    # Print Jaccard Index statistics
    print(f"\n{'=' * 60}")
    print("📊 JACCARD INDEX RESULTS")
    print(f"{'=' * 60}")
    if jaccard_indices:
        # Compute simple average
        avg_jaccard = sum(jaccard_indices) / len(jaccard_indices)
        min_jaccard = min(jaccard_indices)
        max_jaccard = max(jaccard_indices)

        # Compute weighted average using ground truth counts as weights
        total_weighted_score = sum(jaccard_idx * gt_count for jaccard_idx, gt_count in zip(jaccard_indices, gt_counts))
        total_weight = sum(gt_counts)

        if total_weight > 0:
            weighted_avg_jaccard = total_weighted_score / total_weight
        else:
            weighted_avg_jaccard = 0.0

        jaccard_index = JaccardIndex(
            simple_average=avg_jaccard,
            weighted_average=weighted_avg_jaccard,
            min_jaccard_index=min_jaccard,
            max_jaccard_index=max_jaccard
        )

        print(f"🎯 Simple Average Jaccard Index: {avg_jaccard:.4f}")
        print(f"⚖️  Weighted Average Jaccard Index: {weighted_avg_jaccard:.4f}")
        print(f"📏 Minimum Jaccard Index: {min_jaccard:.4f}")
        print(f"📏 Maximum Jaccard Index: {max_jaccard:.4f}")
        print(f"🏷️  Total ground truth objects: {total_weight}")

    # Print F1 Score statistics for each threshold
    print(f"\n{'=' * 60}")
    print("📊 F1 SCORE RESULTS (PRECISION-RECALL CURVE)")
    print(f"{'=' * 60}")
    print(f"🎯 IoU Thresholds: {IOU_THRESHOLDS}")
    print(f"\n{'IoU':<10} {'TP':<10} {'FP':<10} {'FN':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 82)
    
    for threshold in IOU_THRESHOLDS:
        tp = threshold_metrics[threshold]['total_tp']
        fp = threshold_metrics[threshold]['total_fp']
        fn = threshold_metrics[threshold]['total_fn']
        metrics = all_metrics[threshold]
        
        print(f"{threshold:<10.2f} {tp:<10} {fp:<10} {fn:<10} "
              f"{metrics.precision:<12.4f} {metrics.recall:<12.4f} {metrics.f1_score:<12.4f}")

    print(f"\nProcess completed successfully!")
    print(f"Processed {len(images_to_process)} images")
    print(f"Output files saved in: {output_dir}")
    print(f"Total annotated images created: {len([f for f in os.listdir(output_dir) if f.endswith('_detected.jpg')])}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
