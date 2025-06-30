"""SQLAlchemy database models."""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

# Create the declarative base
Base = declarative_base()


class Image(Base):
    """Model representing an uploaded image.
    
    Uses SHA-256 hash as the primary key for content-based deduplication.
    """
    __tablename__ = "images"
    
    # SHA-256 hash of the image content (64 chars)
    id = Column(String(64), primary_key=True, nullable=False)
    
    # Storage path where the image file is saved
    path = Column(String, nullable=False, unique=True)
    
    # Relationship to circular objects detected in this image
    circular_objects = relationship(
        "CircularObject",
        back_populates="image",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Image(id={self.id[:8]}..., path={self.path})>"


class CircularObject(Base):
    """Model representing a detected circular object within an image.
    
    Stores the detection results including bounding box, centroid, and radius.
    """
    __tablename__ = "circular_objects"
    
    # Unique identifier for this detection
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )
    
    # Foreign key to the image containing this object
    image_id = Column(
        String(64),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Bounding box as [x_min, y_min, x_max, y_max]
    bbox = Column(JSON, nullable=False)
    
    # Centroid coordinates as {"x": float, "y": float}
    centroid = Column(JSON, nullable=False)
    
    # Radius of the detected circle in pixels
    radius = Column(Float, nullable=False)
    
    # Relationship to the parent image
    image = relationship("Image", back_populates="circular_objects")
    
    def __repr__(self) -> str:
        return (
            f"<CircularObject(id={self.id}, image_id={self.image_id[:8]}..., "
            f"centroid={self.centroid}, radius={self.radius})>"
        )
    
    @property
    def bbox_list(self) -> List[float]:
        """Get bounding box as a list of floats."""
        if isinstance(self.bbox, list):
            return self.bbox
        return []
    
    @property
    def centroid_dict(self) -> Dict[str, float]:
        """Get centroid as a dictionary with x and y coordinates."""
        if isinstance(self.centroid, dict):
            return self.centroid
        return {"x": 0.0, "y": 0.0}


# Pydantic-compatible types for better integration
class ImageDict(dict):
    """Dictionary representation of Image for Pydantic compatibility."""
    id: str
    path: str
    circular_objects: Optional[List[Dict[str, Any]]] = None


class CircularObjectDict(dict):
    """Dictionary representation of CircularObject for Pydantic compatibility."""
    id: str
    image_id: str
    bbox: List[float]
    centroid: Dict[str, float]
    radius: float 