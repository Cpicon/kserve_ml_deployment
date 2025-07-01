"""Tests for Pydantic schemas."""

import hashlib

import pytest
from pydantic import ValidationError

from aiq_circular_detection.schemas.image import ImageUploadResponse


class TestImageUploadResponse:
    """Test cases for ImageUploadResponse schema."""
    
    def test_valid_sha256_hash(self):
        """Test creating response with valid SHA-256 hash."""
        # Generate a valid SHA-256 hash
        content = b"test content"
        valid_hash = hashlib.sha256(content).hexdigest()
        
        response = ImageUploadResponse(image_id=valid_hash)
        assert response.image_id == valid_hash
        assert len(response.image_id) == 64
    
    def test_uppercase_hash_converted_to_lowercase(self):
        """Test that uppercase hashes are converted to lowercase."""
        uppercase_hash = "A" * 64
        response = ImageUploadResponse(image_id=uppercase_hash)
        assert response.image_id == "a" * 64
    
    def test_invalid_length_hash(self):
        """Test that hashes with incorrect length are rejected."""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            ImageUploadResponse(image_id="a" * 63)
        errors = exc_info.value.errors()
        assert any("64 characters" in str(error) for error in errors)
        
        # Too long
        with pytest.raises(ValidationError) as exc_info:
            ImageUploadResponse(image_id="a" * 65)
        errors = exc_info.value.errors()
        assert any("64 characters" in str(error) for error in errors)
    
    def test_invalid_characters_hash(self):
        """Test that hashes with non-hex characters are rejected."""
        invalid_hash = "g" * 64  # 'g' is not a valid hex character
        with pytest.raises(ValidationError) as exc_info:
            ImageUploadResponse(image_id=invalid_hash)
        errors = exc_info.value.errors()
        assert any("hexadecimal" in str(error) for error in errors)
    
    def test_missing_image_id(self):
        """Test that missing image_id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageUploadResponse()
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("image_id",)
    
    def test_json_serialization(self):
        """Test JSON serialization of the response."""
        valid_hash = "a" * 64
        response = ImageUploadResponse(image_id=valid_hash)
        
        # Test model_dump
        data = response.model_dump()
        assert data == {"image_id": valid_hash, "detection": None}
        
        # Test JSON string serialization
        json_str = response.model_dump_json()
        assert valid_hash in json_str
    
    def test_json_schema(self):
        """Test that JSON schema is properly generated with examples."""
        schema = ImageUploadResponse.model_json_schema()
        
        # Check basic schema properties
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "image_id" in schema["properties"]
        
        # Check that examples are included
        assert "examples" in schema
        assert len(schema["examples"]) > 0
        assert "image_id" in schema["examples"][0]
        assert "detection" in schema["examples"][0]
        
        # Check field description
        image_id_schema = schema["properties"]["image_id"]
        assert "description" in image_id_schema
        assert "SHA-256" in image_id_schema["description"]
        
        # Check pattern and length constraints
        assert "minLength" in image_id_schema
        assert image_id_schema["minLength"] == 64
        assert "maxLength" in image_id_schema
        assert image_id_schema["maxLength"] == 64
        assert "pattern" in image_id_schema
        assert image_id_schema["pattern"] == "^[a-f0-9]{64}$"
    
    def test_real_sha256_examples(self):
        """Test with real SHA-256 hash examples."""
        # Test with empty string hash
        empty_hash = hashlib.sha256(b"").hexdigest()
        response = ImageUploadResponse(image_id=empty_hash)
        assert response.image_id == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        
        # Test with "test" string hash
        test_hash = hashlib.sha256(b"test").hexdigest()
        response = ImageUploadResponse(image_id=test_hash)
        assert response.image_id == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" 