from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from exml.config import (
    BACKGROUND_FILENAME,
    DEFAULT_ARTIFACT_DIR,
    METADATA_FILENAME,
    MODEL_FILENAME,
    TOP_K_DEFAULT,
)
from exml.explain import explain_single
from exml.monitoring import DriftMonitor
from exml.observability import configure_logging, install_request_tracing
from exml.schemas import (
    BreastCancerFeatures,
    ContributionItem,
    DriftStatusResponse,
    ExplainResponse,
    HealthResponse,
    PredictResponse,
)
from exml.security import authorize_request, load_api_keys


def create_app(artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> FastAPI:
    artifacts = Path(artifact_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.pipeline = joblib.load(artifacts / MODEL_FILENAME)
            app.state.background = joblib.load(artifacts / BACKGROUND_FILENAME)
            app.state.metadata = json.loads((artifacts / METADATA_FILENAME).read_text(encoding="utf-8"))
            baseline_stats = app.state.metadata.get("feature_baseline", {})
            app.state.drift_monitor = DriftMonitor(baseline_stats=baseline_stats)
            app.state.load_error = None
        except Exception as exc:  # noqa: BLE001
            app.state.load_error = str(exc)
        yield

    configure_logging()
    app = FastAPI(title="Explainable ML Predictor", version="0.2.0", lifespan=lifespan)
    install_request_tracing(app)

    app.state.pipeline = None
    app.state.metadata = None
    app.state.background = None
    app.state.load_error = None
    app.state.api_keys = load_api_keys()
    app.state.drift_monitor = DriftMonitor(baseline_stats={})

    def _ensure_model_loaded() -> None:
        if app.state.pipeline is None or app.state.metadata is None or app.state.background is None:
            detail = (
                "Model artifacts missing. Train first with "
                "`python -m exml.cli train --model logistic --out artifacts/`."
            )
            if app.state.load_error:
                detail = f"{detail} Loader error: {app.state.load_error}"
            raise HTTPException(status_code=503, detail=detail)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = app.state.pipeline is not None and app.state.metadata is not None
        return HealthResponse(status="ok" if loaded else "not_ready", model_loaded=loaded)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: BreastCancerFeatures, request: Request) -> PredictResponse:
        _ensure_model_loaded()
        authorize_request(request, {"predictor", "admin"})
        data = payload.model_dump(by_alias=True)
        frame = pd.DataFrame([data])[app.state.metadata["feature_names"]]
        app.state.drift_monitor.update(frame)
        predicted_class = int(app.state.pipeline.predict(frame)[0])
        predicted_probability = float(app.state.pipeline.predict_proba(frame)[0, 1])
        return PredictResponse(
            predicted_class=predicted_class,
            predicted_probability=predicted_probability,
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(payload: BreastCancerFeatures, request: Request) -> ExplainResponse:
        _ensure_model_loaded()
        authorize_request(request, {"admin"})
        data = payload.model_dump(by_alias=True)
        frame = pd.DataFrame([data])[app.state.metadata["feature_names"]]
        app.state.drift_monitor.update(frame)
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
            top_contributions=[
                ContributionItem(
                    feature=item["feature"],
                    value=item["value"],
                    contribution=item["contribution"],
                )
                for item in explanation.contributions
            ],
        )

    @app.get("/monitoring/drift", response_model=DriftStatusResponse)
    def drift_status(request: Request) -> DriftStatusResponse:
        _ensure_model_loaded()
        authorize_request(request, {"admin"})
        snapshot = app.state.drift_monitor.snapshot()
        return DriftStatusResponse(**snapshot)

    return app


app = create_app()
