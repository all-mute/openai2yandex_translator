import pytest
from fastapi.testclient import TestClient
from app.app import app  # Изменено на абсолютный импорт

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

@pytest.mark.skip(reason="idk")
def test_get_badge():
    response = client.get("/badge")
    assert response.status_code == 302  # Проверяем, что происходит редирект
    assert response.headers["Location"] == "https://img.shields.io/badge/status-online-brightgreen.svg"

def test_version():
    from app.app import app_version
    
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": app_version}  # Убедитесь, что версия соответствует ожидаемой
