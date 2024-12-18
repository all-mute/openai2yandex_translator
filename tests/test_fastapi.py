import pytest
from fastapi.testclient import TestClient
from app.index import index

client = TestClient(index)

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

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Hello from Foundational Models Team! check .../docs for more info"}

@pytest.mark.skip(reason="idk")
def test_badge():
    response = client.get("/badge")
    assert response.status_code == 200
    assert response.headers["location"].startswith("https://img.shields.io/badge/status-healthy-green")

@pytest.mark.skip(reason="idk")
def test_badge_sha():
    response = client.get("/badge-sha")
    assert response.status_code == 200
    assert "sha-" in response.headers["location"]
    
@pytest.mark.skip(reason="idk")
def test_badge_ref():
    response = client.get("/badge-ref")
    assert response.status_code == 200
    assert "ref-" in response.headers["location"]

@pytest.mark.skip(reason="idk")
def test_non_existent_endpoint():
    response = client.get("/non-existent-endpoint")
    assert response.status_code == 405
    assert response.json() == {"detail": "Method Not Allowed"}


