from fastapi.testclient import TestClient

from exml.api import create_app
from exml.train import train_and_save


def test_api_health_smoke(tmp_path):
    train_and_save(model_name="logistic", out_dir=str(tmp_path))
    app = create_app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True
