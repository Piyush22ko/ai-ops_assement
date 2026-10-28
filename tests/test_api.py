from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={
            "email_text": "Congratulations! You won a free prize!"
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "result" in data


def test_empty_email_rejected():
    response = client.post(
        "/predict",
        json={"email_text": ""},
    )

    assert response.status_code == 422