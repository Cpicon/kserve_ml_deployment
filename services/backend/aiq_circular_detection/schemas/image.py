"""Image-related Pydantic schemas for API responses."""


from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImageUploadResponse(BaseModel):
    """Response model for successful image upload.
    
    This model is returned when an image is successfully uploaded
    and stored in the system. The image_id can be used to retrieve
    or reference the image in subsequent API calls.
    
    The image_id is a SHA-256 hash of the image content, ensuring
    that duplicate images are detected and the same ID is returned.
    """
    
    model_config = ConfigDict(
        # Enable JSON schema generation with examples
        json_schema_extra={
            "examples": [
                {
                    "image_id": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                },
                {
                    "image_id": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
                },
                {
                    "image_id": "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
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