from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from exml.config import (
    BACKGROUND_FILENAME,
    DEFAULT_ARTIFACT_DIR,
    METADATA_FILENAME,
    MODEL_FILENAME,
    TOP_K_DEFAULT,
)
from exml.explain import explain_single
from exml.monitoring import PredictionMonitor
from exml.schemas import BreastCancerFeatures, ExplainResponse, HealthResponse, PredictResponse


def create_app(artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> FastAPI:
    artifacts = Path(artifact_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.pipeline = joblib.load(artifacts / MODEL_FILENAME)
            app.state.background = joblib.load(artifacts / BACKGROUND_FILENAME)
            app.state.metadata = json.loads((artifacts / METADATA_FILENAME).read_text(encoding="utf-8"))
            app.state.load_error = None
        except Exception as exc:  # noqa: BLE001
            app.state.load_error = str(exc)
        yield

    app = FastAPI(title="Explainable ML Predictor", version="0.1.0", lifespan=lifespan)

    app.state.pipeline = None
    app.state.metadata = None
    app.state.background = None
    app.state.load_error = None
    app.state.monitor = PredictionMonitor(Path(os.getenv("EXML_PREDICTION_LOG", "artifacts/predictions.jsonl")))

    def _ensure_model_loaded() -> None:
        if app.state.pipeline is None or app.state.metadata is None or app.state.background is None:
            detail = "Model artifacts missing. Train first with `python -m exml.cli train --model logistic --out artifacts/`."
            if app.state.load_error:
                detail = f"{detail} Loader error: {app.state.load_error}"
            raise HTTPException(status_code=503, detail=detail)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = app.state.pipeline is not None and app.state.metadata is not None
        return HealthResponse(status="ok" if loaded else "not_ready", model_loaded=loaded)


    @app.get("/monitoring/summary")
    def monitoring_summary() -> dict:
        return app.state.monitor.summary()

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: BreastCancerFeatures) -> PredictResponse:
        _ensure_model_loaded()
        data = payload.model_dump(by_alias=True)
        frame = pd.DataFrame([data])[app.state.metadata["feature_names"]]
        predicted_class = int(app.state.pipeline.predict(frame)[0])
        predicted_probability = float(app.state.pipeline.predict_proba(frame)[0, 1])
        app.state.monitor.log_prediction(predicted_class, predicted_probability)
        return PredictResponse(
            predicted_class=predicted_class,
            predicted_probability=predicted_probability,
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(payload: BreastCancerFeatures) -> ExplainResponse:
        _ensure_model_loaded()
        data = payload.model_dump(by_alias=True)
        frame = pd.DataFrame([data])[app.state.metadata["feature_names"]]
        explanation = explain_single(
            pipeline=app.state.pipeline,
            background_df=app.state.background,
            input_df=frame,
            model_name=app.state.metadata["model_name"],
            top_k=TOP_K_DEFAULT,
        )
        return ExplainResponse(
            base_value=explanation.base_value,
            predicted_probability=explanation.predicted_probability,
            top_contributions=explanation.contributions,
        )

    return app


app = create_app()
