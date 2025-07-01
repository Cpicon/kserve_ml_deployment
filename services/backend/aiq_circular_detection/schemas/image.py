"""Image-related Pydantic schemas for API responses."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .detection import DetectionResponse


class ImageUploadResponse(BaseModel):
    """Response model for a successful image upload.
    
    This model is returned when an image is successfully uploaded
    and stored in the system. The image_id can be used to retrieve
    or reference the image in subsequent API calls.
    
    The image_id is a SHA-256 hash of the image content, ensuring
    that duplicate images are detected and the same ID is returned.
    
    The response also includes detection results from the model inference.
    """
    
    model_config = ConfigDict(
        # Enable JSON schema generation with examples
        json_schema_extra={
            "examples": [
                {
                    "image_id": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "detection": {
                        "detections": [
                            {
                                "object_id": "550e8400-e29b-41d4-a716-446655440000",
                                "image_id": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                                "bbox": [10.0, 10.0, 50.0, 50.0],
                                "centroid": {"x": 30.0, "y": 30.0},
                                "radius": 20.0
                            }
                        ],
                        "count": 1
                    }
                }
            ]
        }
    )
    
    image_id: str = Field(
        ...,
        description="Unique identifier for the uploaded image."
                    " This is a SHA-256 hash of the image content,"
                    " ensuring that identical images will have the same ID.",
        min_length=64,
        max_length=64,
        pattern="^[a-f0-9]{64}$",
        examples=[
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
        ]
    )
    
    detection: Optional[DetectionResponse] = Field(
        None,
        description="Detection results from model inference. "
                    "May be None if inference fails or is disabled."
    )
    
    @field_validator("image_id", mode="before")
    @classmethod
    def normalize_and_validate_sha256(cls, v: str) -> str:
        """Normalize and validate that the image_id is a valid SHA-256 hash."""
        if not isinstance(v, str):
            raise ValueError("image_id must be a string")
        
        # Convert to lowercase
        v = v.lower()
        
        # Validate length
        if len(v) != 64:
            raise ValueError("image_id must be exactly 64 characters long")
        
        # Validate characters
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("image_id must contain only hexadecimal characters (0-9, a-f)")
        
        return v 