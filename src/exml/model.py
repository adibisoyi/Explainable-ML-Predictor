from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(model_name: str, feature_names: list[str]) -> Pipeline:
    if model_name not in {"logistic", "rf"}:
        raise ValueError("model_name must be one of: logistic, rf")

    if model_name == "logistic":
        model = LogisticRegression(max_iter=2000, random_state=42)
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), feature_names)],
            remainder="drop",
        )
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        preprocessor = ColumnTransformer(
            transformers=[("num", "passthrough", feature_names)],
            remainder="drop",
        )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
