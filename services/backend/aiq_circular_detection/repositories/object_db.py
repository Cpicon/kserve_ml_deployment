"""SQLAlchemy-based repository for CircularObject entities."""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from aiq_circular_detection.models.db import CircularObject, Image

logger = logging.getLogger(__name__)


class CircularObjectDBRepository:
    """Repository for managing CircularObject entities in the database.
    
    This repository handles CRUD operations for circular objects
    detected within images.
    """
    
    def __init__(self, db: Session):
        """Initialize the repository with a database session.
        
        Args:
            db: SQLAlchemy session for database operations
        """
        self.db = db
    
    def create_object(
        self,
        image_id: str,
        bbox: List[float],
        centroid: dict[str, float],
        radius: float
    ) -> CircularObject:
        """Create a new circular object record.
        
        Args:
            image_id: SHA-256 hash ID of the parent image
            bbox: Bounding box as [x_min, y_min, x_max, y_max]
            centroid: Centroid coordinates as {"x": float, "y": float}
            radius: Radius of the detected circle in pixels
            
        Returns:
            CircularObject: The created CircularObject instance
            
        Raises:
            ValueError: If the image_id doesn't exist or parameters are invalid
        """
        # Validate image exists
        image = self.db.query(Image).filter(Image.id == image_id).first()
        if not image:
            raise ValueError(f"Image with ID {image_id} does not exist")
        
        # Validate bbox
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox must be a list of 4 floats [x_min, y_min, x_max, y_max]")
        
        # Validate centroid
        if not isinstance(centroid, dict) or 'x' not in centroid or 'y' not in centroid:
            raise ValueError("centroid must be a dict with 'x' and 'y' keys")
        
        # Validate radius
        if radius <= 0:
            raise ValueError("radius must be positive")
        
        try:
            # Create circular object
            circular_object = CircularObject(
                image_id=image_id,
                bbox=bbox,
                centroid=centroid,
                radius=radius
            )
            
            self.db.add(circular_object)
            self.db.commit()
            self.db.refresh(circular_object)
            
            logger.info(f"Created circular object: {circular_object.id} for image {image_id}")
            return circular_object
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create circular object: {e}")
            raise
    
    def get_object(self, object_id: UUID) -> Optional[CircularObject]:
        """Retrieve a circular object by its ID.
        
        Args:
            object_id: UUID of the circular object
            
        Returns:
            Optional[CircularObject]: The CircularObject instance if found, None otherwise
        """
        obj = self.db.query(CircularObject).filter(CircularObject.id == object_id).first()
        if obj:
            logger.debug(f"Found circular object: {object_id}")
        else:
            logger.debug(f"Circular object not found: {object_id}")
        return obj
    
    def list_objects(
        self,
        image_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[CircularObject]:
        """List circular objects with optional filtering.
        
        Args:
            image_id: Optional image ID to filter by
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List[CircularObject]: List of circular objects matching the criteria
        """
        query = self.db.query(CircularObject)
        
        if image_id:
            query = query.filter(CircularObject.image_id == image_id)
        
        query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        objects = query.all()
        logger.debug(f"Found {len(objects)} circular objects")
        return objects
    
    def delete_object(self, object_id: UUID) -> bool:
        """Delete a circular object.
        
        Args:
            object_id: UUID of the circular object to delete
            
        Returns:
            bool: True if the object was deleted, False if not found
        """
        obj = self.get_object(object_id)
        if not obj:
            logger.warning(f"Cannot delete non-existent circular object: {object_id}")
            return False
        
        try:
            self.db.delete(obj)
            self.db.commit()
            logger.info(f"Deleted circular object: {object_id}")
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete circular object {object_id}: {e}")
            raise
    
    def count(self, image_id: Optional[str] = None) -> int:
        """Get the total number of circular objects.
        
        Args:
            image_id: Optional image ID to count objects for
            
        Returns:
            int: Number of circular objects
        """
        query = self.db.query(CircularObject)
        if image_id:
            query = query.filter(CircularObject.image_id == image_id)
        return query.count()
    
    def update_object(
        self,
        object_id: UUID,
        bbox: Optional[List[float]] = None,
        centroid: Optional[dict[str, float]] = None,
        radius: Optional[float] = None
    ) -> Optional[CircularObject]:
        """Update a circular object's properties.
        
        Args:
            object_id: UUID of the circular object to update
            bbox: New bounding box (optional)
            centroid: New centroid (optional)
            radius: New radius (optional)
            
        Returns:
            Optional[CircularObject]: Updated object if found, None otherwise
        """
        obj = self.get_object(object_id)
        if not obj:
            logger.warning(f"Cannot update non-existent circular object: {object_id}")
            return None
        
        try:
            # Update provided fields
            if bbox is not None:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError("bbox must be a list of 4 floats")
                obj.bbox = bbox
            
            if centroid is not None:
                if not isinstance(centroid, dict) or 'x' not in centroid or 'y' not in centroid:
                    raise ValueError("centroid must be a dict with 'x' and 'y' keys")
                obj.centroid = centroid
            
            if radius is not None:
                if radius <= 0:
                    raise ValueError("radius must be positive")
                obj.radius = radius
            
            self.db.commit()
            self.db.refresh(obj)
            
            logger.info(f"Updated circular object: {object_id}")
            return obj
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update circular object {object_id}: {e}")
            raise 