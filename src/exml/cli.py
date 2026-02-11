from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import uvicorn

from exml.api import create_app
from exml.config import (
    BACKGROUND_FILENAME,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_HOST,
    DEFAULT_PORT,
    METADATA_FILENAME,
    MODEL_FILENAME,
)
from exml.data import load_default_dataset
from exml.explain import explain_single
from exml.train import train_and_save


def _load_local_artifacts(artifact_dir: str):
    artifacts = Path(artifact_dir)
    pipeline = joblib.load(artifacts / MODEL_FILENAME)
    background = joblib.load(artifacts / BACKGROUND_FILENAME)
    metadata = json.loads((artifacts / METADATA_FILENAME).read_text(encoding="utf-8"))
    return pipeline, background, metadata


def cmd_train(args: argparse.Namespace) -> None:
    metadata = train_and_save(
        model_name=args.model,
        out_dir=args.out,
        csv_path=args.csv,
        target_column=args.target,
    )
    print("Training complete")
    print(json.dumps(metadata["metrics"], indent=2))


def cmd_sample_json(_: argparse.Namespace) -> None:
    sample = load_default_dataset().X.iloc[0].to_dict()
    print(json.dumps(sample, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    pipeline, _, metadata = _load_local_artifacts(args.artifacts)
    payload = json.loads(args.json)
    frame = pd.DataFrame([payload])[metadata["feature_names"]]
    result = {
        "predicted_class": int(pipeline.predict(frame)[0]),
        "predicted_probability": float(pipeline.predict_proba(frame)[0, 1]),
    }
    print(json.dumps(result, indent=2))


def cmd_explain(args: argparse.Namespace) -> None:
    pipeline, background, metadata = _load_local_artifacts(args.artifacts)
    payload = json.loads(args.json)
    frame = pd.DataFrame([payload])[metadata["feature_names"]]
    result = explain_single(
        pipeline=pipeline,
        background_df=background,
        input_df=frame,
        model_name=metadata["model_name"],
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "base_value": result.base_value,
                "predicted_probability": result.predicted_probability,
                "top_contributions": result.contributions,
            },
            indent=2,
        )
    )


def cmd_serve(args: argparse.Namespace) -> None:
    app = create_app(args.artifacts)
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explainable ML Predictor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save model artifacts")
    train_parser.add_argument("--model", choices=["logistic", "rf"], default="logistic")
    train_parser.add_argument("--out", default=str(DEFAULT_ARTIFACT_DIR))
    train_parser.add_argument("--csv", default=None)
    train_parser.add_argument("--target", default=None)
    train_parser.set_defaults(func=cmd_train)

    serve_parser = subparsers.add_parser("serve", help="Run FastAPI service")
    serve_parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR))
    serve_parser.add_argument("--host", default=DEFAULT_HOST)
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_parser.set_defaults(func=cmd_serve)

    predict_parser = subparsers.add_parser("predict", help="Predict one sample from JSON payload")
    predict_parser.add_argument("--json", required=True)
    predict_parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR))
    predict_parser.set_defaults(func=cmd_predict)

    explain_parser = subparsers.add_parser("explain", help="Explain one sample from JSON payload")
    explain_parser.add_argument("--json", required=True)
    explain_parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR))
    explain_parser.add_argument("--top-k", type=int, default=10)
    explain_parser.set_defaults(func=cmd_explain)

    sample_parser = subparsers.add_parser("sample-json", help="Print a valid sample payload")
    sample_parser.set_defaults(func=cmd_sample_json)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
