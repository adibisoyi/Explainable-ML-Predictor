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


def test_monitoring_summary_updates_after_prediction(tmp_path):
    train_and_save(model_name="logistic", out_dir=str(tmp_path))
    app = create_app(tmp_path)

    with TestClient(app) as client:
        initial = client.get("/monitoring/summary")
        payload = {
            "mean radius": 14.2,
            "mean texture": 20.1,
            "mean perimeter": 90.0,
            "mean area": 600.0,
            "mean smoothness": 0.1,
            "mean compactness": 0.1,
            "mean concavity": 0.1,
            "mean concave points": 0.05,
            "mean symmetry": 0.2,
            "mean fractal dimension": 0.06,
            "radius error": 0.3,
            "texture error": 1.2,
            "perimeter error": 2.0,
            "area error": 30.0,
            "smoothness error": 0.005,
            "compactness error": 0.02,
            "concavity error": 0.02,
            "concave points error": 0.01,
            "symmetry error": 0.02,
            "fractal dimension error": 0.003,
            "worst radius": 16.0,
            "worst texture": 25.0,
            "worst perimeter": 100.0,
            "worst area": 700.0,
            "worst smoothness": 0.12,
            "worst compactness": 0.2,
            "worst concavity": 0.2,
            "worst concave points": 0.1,
            "worst symmetry": 0.3,
            "worst fractal dimension": 0.08,
        }
        _ = client.post("/predict", json=payload)
        updated = client.get("/monitoring/summary")

    assert initial.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["total_predictions"] == initial.json()["total_predictions"] + 1
