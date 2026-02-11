from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    target_name: str


def load_default_dataset() -> DatasetBundle:
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.astype(int).copy()
    return DatasetBundle(X=X, y=y, target_name="target")


def load_dataset(csv_path: str | None = None, target_column: str | None = None) -> DatasetBundle:
    if csv_path is None:
        return load_default_dataset()

    frame = pd.read_csv(Path(csv_path))
    if frame.empty:
        raise ValueError("CSV is empty; provide rows with feature values and target.")

    target_col = target_column or "target"
    if target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    y = frame[target_col].astype(int)
    X = frame.drop(columns=[target_col])
    return DatasetBundle(X=X, y=y, target_name=target_col)
