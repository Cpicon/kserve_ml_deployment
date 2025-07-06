"""Unit tests for the model client."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from aiq_circular_detection.model_client import (
    DummyModelClient,
    RealModelClient,
    create_model_client,
)
from config import Settings

# Configure pytest-asyncio for async tests


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.database_url = "sqlite:///:memory:"
    settings.mode = "real"
    settings.model_server_url = "http://test-server.com"
    settings.model_name = "test-model"
    settings.model_service_timeout = 30
    return settings


@pytest.fixture
def dummy_settings():
    """Create settings for dummy mode."""
    settings = Mock(spec=Settings)
    settings.database_url = "sqlite:///:memory:"
    settings.mode = "dummy"
    return settings


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary image file for testing."""
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake image content")
    return str(image_path)

class TestRealModelClient:
    """Tests for the RealModelClient."""
    
    def test_init_requires_model_server_url(self):
        """Test that RealModelClient requires MODEL_SERVER_URL to be set."""
        settings = Mock(spec=Settings)
        settings.model_server_url = None
        
        with pytest.raises(ValueError, match="MODEL_SERVER_URL must be set"):
            RealModelClient(settings)
    
    @pytest.mark.asyncio
    async def test_detect_circles_success(self, mock_settings, sample_image_path):
        """Test successful circle detection with mocked HTTP response."""
        client = RealModelClient(mock_settings)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {
                    "bbox": [100, 100, 200, 200],
                    "centroid": {"x": 150, "y": 150},
                    "radius": 50
                }
            ]
        }
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            circles = await client.detect_circles(sample_image_path)
            
            # Verify the request was made correctly
            expected_url = "http://test-server.com/v1/models/test-model:predict"
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == expected_url
            
            # Check the payload
            payload = call_args[1]["json"]
            assert "image" in payload
            assert "label" in payload
            assert payload["label"] == "circular_objects"
            
            # Verify response
            assert len(circles) == 1
            assert circles[0]["radius"] == 50
    
    @pytest.mark.asyncio
    async def test_detect_circles_file_not_found(self, mock_settings):
        """Test that FileNotFoundError is raised for missing image."""
        client = RealModelClient(mock_settings)
        
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            await client.detect_circles("/nonexistent/image.jpg")
    
    @pytest.mark.asyncio
    async def test_detect_circles_http_error(self, mock_settings, sample_image_path):
        """Test handling of HTTP errors from model server."""
        client = RealModelClient(mock_settings)
        
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Server error",
            request=Mock(),
            response=mock_response
        )
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            with pytest.raises(httpx.HTTPStatusError):
                await client.detect_circles(sample_image_path)
    
    @pytest.mark.asyncio
    async def test_detect_circles_logs_timing(self, mock_settings, sample_image_path, caplog):
        """Test that timing information is logged."""
        client = RealModelClient(mock_settings)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"predictions": []}
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            with caplog.at_level("DEBUG"):
                await client.detect_circles(sample_image_path)
            
            # Check debug log
            assert any("Starting model inference" in record.message for record in caplog.records)
            
            # Check info log with timing
            assert any("Model inference completed" in record.message for record in caplog.records)
            assert any("detected 0 circles" in record.message for record in caplog.records)


class TestCreateModelClient:
    """Tests for the factory function."""
    
    def test_create_dummy_client(self, dummy_settings):
        """Test that dummy mode creates DummyModelClient."""
        client = create_model_client(dummy_settings)
        assert isinstance(client, DummyModelClient)
    
    def test_create_real_client(self, mock_settings):
        """Test that real mode creates RealModelClient."""
        client = create_model_client(mock_settings)
        assert isinstance(client, RealModelClient) 