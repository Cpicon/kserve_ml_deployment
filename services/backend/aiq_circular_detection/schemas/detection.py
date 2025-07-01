"""Pydantic schemas for circular object detection."""

from typing import Dict, List
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CircularObjectCreate(BaseModel):
    """Schema for creating new circular object detection."""
    
    bbox: List[float] = Field(
        ...,
        description="Bounding box coordinates as [x_min, y_min, x_max, y_max]",
        min_length=4,
        max_length=4,
        examples=[[10.0, 20.0, 30.0, 40.0]]
    )
    
    centroid: Dict[str, float] = Field(
        ...,
        description="Center point of the circular object",
        examples=[{"x": 20.0, "y": 30.0}]
    )
    
    radius: float = Field(
        ...,
        gt=0,
        description="Radius of the circular object in pixels",
        examples=[10.0]
    )
    
    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        """Validate bounding box coordinates."""
        if len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        
        x_min, y_min, x_max, y_max = v
        if x_min >= x_max:
            raise ValueError("x_min must be less than x_max")
        if y_min >= y_max:
            raise ValueError("y_min must be less than y_max")
        
        return v
    
    @field_validator("centroid")
    @classmethod
    def validate_centroid(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate centroid coordinates."""
        if "x" not in v or "y" not in v:
            raise ValueError("Centroid must have 'x' and 'y' coordinates")
        
        # Ensure only x and y keys exist
        if set(v.keys()) != {"x", "y"}:
            raise ValueError("Centroid must only contain 'x' and 'y' coordinates")
        
        return v


class CircularObjectResponse(CircularObjectCreate):
    """Schema for circular object detection response."""

    model_config = ConfigDict(from_attributes=True)

    object_id: UUID = Field(
        ...,
        description="Unique identifier for this detection"
    )


class DetectionResponse(BaseModel):
    """Schema for circular object detection results."""

    detections: List[CircularObjectResponse] = Field(
        ...,
        description="List of detected circular objects"
    )
    
    count: int = Field(
        ...,
        ge=0,
        description="Total number of circular objects detected"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "image_id": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                    "detections": [
                        {
                            "object_id": "550e8400-e29b-41d4-a716-446655440000",
                            "bbox": [10.0, 20.0, 30.0, 40.0],
                            "centroid": {"x": 20.0, "y": 30.0},
                            "radius": 10.0
                        }
                    ],
                    "count": 1
                }
            ]
        }
    ) 