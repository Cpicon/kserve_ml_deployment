"""Smoke tests for FastAPI application endpoints.

This module uses pytest's monkeypatch fixture to mock settings values,
avoiding dependencies on specific configuration files or environment variables.
This approach ensures tests run reliably in any environment.
"""
from fastapi.testclient import TestClient

from aiq_circular_detection import main
from aiq_circular_detection.main import app


def test_health_endpoint(monkeypatch):
    """Test health endpoint with mocked settings."""
    # Mock the settings attributes used in the health endpoint
    monkeypatch.setattr(main.settings, "environment", "test")
    monkeypatch.setattr(main.settings, "app_version", "1.0.0")
    
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "test",
        "version": "1.0.0",
    }


def test_config_endpoint(monkeypatch):
    """Test config endpoint with mocked settings."""
    # Mock the settings attributes used in the config endpoint
    monkeypatch.setattr(main.settings, "app_name", "Test App")
    monkeypatch.setattr(main.settings, "app_version", "2.0.0")
    monkeypatch.setattr(main.settings, "environment", "test")
    monkeypatch.setattr(main.settings, "debug", True)
    monkeypatch.setattr(main.settings, "api_prefix", "/api/v2")
    monkeypatch.setattr(main.settings, "storage_type", "s3")
    monkeypatch.setattr(main.settings, "log_level", "DEBUG")
    monkeypatch.setattr(main.settings, "log_json", True)
    monkeypatch.setattr(main.settings, "max_upload_size", 20 * 1024 * 1024)  # 20MB
    
    client = TestClient(app)
    response = client.get("/config")
    assert response.status_code == 200
    assert response.json() == {
        "app_name": "Test App",
        "app_version": "2.0.0",
        "environment": "test",
        "debug": True,
        "api_prefix": "/api/v2",
        "storage_type": "s3",
        "log_level": "DEBUG",
        "log_json": True,
        "max_upload_size": 20 * 1024 * 1024,
    }