"""Object-related Pydantic schemas for API responses."""

from typing import List
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ObjectSummary(BaseModel):
    """Summary schema for circular object detection.
    
    This model provides a simplified view of a detected circular object,
    containing only the object ID and its bounding box coordinates.
    The bounding box is represented as integer pixel coordinates.
    """
    
    model_config = ConfigDict(
        from_attributes=True,
        # Enable JSON schema generation with examples
        json_schema_extra={
            "examples": [
                {
                    "object_id": "550e8400-e29b-41d4-a716-446655440000",
                    "bbox": [10, 20, 30, 40]
                },
                {
                    "object_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                    "bbox": [100, 150, 200, 250]
                },
                {
                    "object_id": "6ba7b812-9dad-11d1-80b4-00c04fd430c8",
                    "bbox": [50, 75, 150, 175]
                }
            ]
        }
    )
    
    object_id: UUID = Field(
        ...,
        description="Unique identifier for the detected circular object",
        examples=[
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        ]
    )
    
    bbox: List[int] = Field(
        ...,
        description="Bounding box coordinates as [x_min, y_min, x_max, y_max] in integer pixels",
        min_length=4,
        max_length=4,
        examples=[
            [10, 20, 30, 40],
            [100, 150, 200, 250]
        ]
    )


class ObjectListSummaryResponse(BaseModel):
    """Response schema for listing circular objects in an image.
    
    This model provides a comprehensive response for object listing,
    including metadata about the query and the list of detected objects.
    """
    
    model_config = ConfigDict(
        # Enable JSON schema generation with examples
        json_schema_extra={
            "examples": [
                {
                    "image_id": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                    "count": 3,
                    "objects": [
                        {
                            "object_id": "550e8400-e29b-41d4-a716-446655440000",
                            "bbox": [10, 20, 30, 40]
                        },
                        {
                            "object_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                            "bbox": [100, 150, 200, 250]
                        },
                        {
                            "object_id": "6ba7b812-9dad-11d1-80b4-00c04fd430c8",
                            "bbox": [50, 75, 150, 175]
                        }
                    ]
                },
                {
                    "image_id": "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
                    "count": 0,
                    "objects": []
                }
            ]
        }
    )
    
    image_id: str = Field(
        ...,
        description="SHA-256 hash ID of the image containing these objects",
        min_length=64,
        max_length=64,
        pattern="^[a-f0-9]{64}$",
        examples=[
            "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
        ]
    )
    
    count: int = Field(
        ...,
        ge=0,
        description="Total number of circular objects detected in this image",
        examples=[0, 1, 3, 10]
    )
    
    objects: List[ObjectSummary] = Field(
        ...,
        description="List of detected circular objects with their bounding boxes",
        examples=[
            [
                {
                    "object_id": "550e8400-e29b-41d4-a716-446655440000",
                    "bbox": [10, 20, 30, 40]
                }
            ]
        ]
    ) 