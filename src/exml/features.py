from __future__ import annotations

import pandas as pd


def ensure_feature_order(frame: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in expected_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return frame[expected_columns]
