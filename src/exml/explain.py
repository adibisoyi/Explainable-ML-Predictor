from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


class Contribution(TypedDict):
    feature: str
    value: float
    contribution: float


@dataclass
class PredictionExplanation:
    base_value: float
    predicted_probability: float
    contributions: list[Contribution]


def _binary_shap_values(raw_shap_values) -> np.ndarray:
    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) == 2:
            return np.array(raw_shap_values[1])
        return np.array(raw_shap_values[0])
    arr = np.array(raw_shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def explain_single(
    pipeline: Pipeline,
    background_df: pd.DataFrame,
    input_df: pd.DataFrame,
    model_name: str,
    top_k: int = 10,
) -> PredictionExplanation:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    background_transformed = preprocess.transform(background_df)
    input_transformed = preprocess.transform(input_df)

    if model_name == "rf":
        explainer = shap.TreeExplainer(model)
        shap_values = _binary_shap_values(explainer.shap_values(input_transformed))
        expected = explainer.expected_value
    else:
        explainer = shap.LinearExplainer(model, background_transformed)
        shap_values = _binary_shap_values(explainer.shap_values(input_transformed))
        expected = explainer.expected_value

    contributions_vector = shap_values[0]
    base_value = float(np.array(expected).reshape(-1)[-1])
    predicted_probability = float(pipeline.predict_proba(input_df)[0, 1])

    items: list[Contribution] = []
    for feature_name, feature_value, contribution in zip(
        input_df.columns,
        input_df.iloc[0].values,
        contributions_vector,
        strict=True,
    ):
        items.append(
            {
                "feature": str(feature_name),
                "value": float(feature_value),
                "contribution": float(contribution),
            }
        )

    items.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return PredictionExplanation(
        base_value=base_value,
        predicted_probability=predicted_probability,
        contributions=items[:top_k],
    )
