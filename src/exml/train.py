from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from exml.config import BACKGROUND_FILENAME, METADATA_FILENAME, MODEL_FILENAME
from exml.data import load_dataset
from exml.model import build_pipeline


def train_and_save(
    model_name: str,
    out_dir: str,
    csv_path: str | None = None,
    target_column: str | None = None,
) -> dict:
    dataset = load_dataset(csv_path=csv_path, target_column=target_column)
    X_train, X_val, y_train, y_val = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=42,
        stratify=dataset.y,
    )

    feature_names = list(X_train.columns)
    pipeline = build_pipeline(model_name=model_name, feature_names=feature_names)
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    val_proba = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "roc_auc": float(roc_auc_score(y_val, val_proba)),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, out_path / MODEL_FILENAME)
    joblib.dump(X_train.head(200), out_path / BACKGROUND_FILENAME)

    metadata = {
        "model_name": model_name,
        "feature_names": feature_names,
        "target_name": dataset.target_name,
        "metrics": metrics,
    }
    (out_path / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata
