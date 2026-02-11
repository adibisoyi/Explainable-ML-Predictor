from fastapi.testclient import TestClient

from exml.api import create_app
from exml.data import load_default_dataset
from exml.train import train_and_save


def test_api_health_smoke(tmp_path):
    train_and_save(model_name="logistic", out_dir=str(tmp_path))
    app = create_app(tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_api_auth_and_drift_endpoint(tmp_path):
    train_and_save(model_name="logistic", out_dir=str(tmp_path))
    app = create_app(tmp_path)
    sample = load_default_dataset().X.iloc[0].to_dict()

    with TestClient(app) as client:
        unauthorized = client.post("/predict", json=sample)
        assert unauthorized.status_code == 401

        invalid_key = client.post("/predict", json=sample, headers={"x-api-key": "bad-key"})
        assert invalid_key.status_code == 401

        predictor = client.post(
            "/predict",
            json=sample,
            headers={"x-api-key": "dev-predict-key", "x-request-id": "test-req-1"},
        )
        assert predictor.status_code == 200
        assert predictor.headers["x-request-id"] == "test-req-1"

        forbidden = client.post("/explain", json=sample, headers={"x-api-key": "dev-predict-key"})
        assert forbidden.status_code == 403

        drift = client.get("/monitoring/drift", headers={"x-api-key": "dev-admin-key"})
        assert drift.status_code == 200
        assert drift.json()["window_size"] >= 1
        assert drift.json()["status"] in {"stable", "drift_detected"}
