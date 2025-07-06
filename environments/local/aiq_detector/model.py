import argparse
import base64
import io
import logging
import time
from typing import Dict
from PIL import Image
from transformers import pipeline
from pydantic import BaseModel
from kserve import Model, ModelServer, model_server
from kserve import logging as kserve_logging
from kserve.model_server import app
from fastapi.middleware.cors import CORSMiddleware
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


def get_centroid_and_radius(bbox: BoundingBox) -> tuple[Centroid, Radius]:
    """Calculate the centroid and radius from a bounding box."""
    center_x = (bbox.xmin + bbox.xmax) // 2
    center_y = (bbox.ymin + bbox.ymax) // 2
    
    # Calculate radius as half of the average of width and height
    width = bbox.xmax - bbox.xmin
    height = bbox.ymax - bbox.ymin
    radius = (width + height) // 4
    
    return Centroid(x=center_x, y=center_y), Radius(radius=radius)


class AiqDetectorModel(Model):
    def __init__(self, name: str):
        super().__init__(name, return_response_headers=True)
        self.name = name
        self.ready = False
        self.detector = None
        self.default_label = "coin"
        self.load()
        
    def load(self):
        """Load the object detection model."""
        try:
            # Initialize the object detection pipeline
            self.detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
            self.ready = True
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    async def predict(
        self,
        payload: Dict,
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None
    ) -> Dict:
        """Process the prediction request and return Prediction objects."""
        start = time.time()
        try:
            # Extract image data from the request
            # Support both base64 encoded and raw image data
            if "instances" in payload:
                # TensorFlow V1 HTTP API format
                img_data = payload["instances"][0]["image"]["b64"]
                raw_img_data = base64.b64decode(img_data)
            elif "inputs" in payload:
                # Alternative format
                img_data = payload["inputs"][0]["data"]
                raw_img_data = base64.b64decode(img_data)
            else:
                raise ValueError("Invalid payload format")
            
            # Open the image
            image = Image.open(io.BytesIO(raw_img_data))
            
            # Get the label from the request or use default
            label = self.default_label
            if "parameters" in payload and "label" in payload["parameters"]:
                label = payload["parameters"]["label"]
            
            # Run object detection
            detections = self.detector(
                image,
                candidate_labels=[label],
            )
            
            # Convert detections to Prediction objects
            predictions = []
            for detection in detections:
                bbox = BoundingBox(
                    xmin=detection["box"]["xmin"],
                    ymin=detection["box"]["ymin"],
                    xmax=detection["box"]["xmax"],
                    ymax=detection["box"]["ymax"]
                )
                centroid, radius = get_centroid_and_radius(bbox)
                
                prediction = Prediction(
                    bounding_box=bbox,
                    centroid=centroid,
                    radius=radius,
                    label=label,
                    score=detection["score"]
                )
                predictions.append(prediction.dict())
            end = time.time()
            if response_headers is not None:
                response_headers.update(
                    {"prediction-time-latency": f"{round((end - start) * 1000, 9)}"}
                )
            return {"predictions": predictions}
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise


parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    # Configure logging
    if args.configure_logging:
        kserve_logging.configure_logging(args.log_config_file)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    model = AiqDetectorModel(args.model_name)
    model.load()
    ModelServer().start([model]) 