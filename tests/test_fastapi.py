import pytest
from fastapi.testclient import TestClient
from app.main import app  # Изменено на абсолютный импорт

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_readiness_probe():
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_liveness_probe():
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}