from __future__ import annotations

import logging
from collections import deque

import pandas as pd

logger = logging.getLogger("exml.monitoring")


class DriftMonitor:
    def __init__(self, baseline_stats: dict[str, dict[str, float]], window_size: int = 200, z_threshold: float = 3.0):
        self.baseline_stats = baseline_stats
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._window: deque[dict[str, float]] = deque(maxlen=window_size)

    def update(self, frame: pd.DataFrame) -> None:
        self._window.append(frame.iloc[0].to_dict())

    def snapshot(self) -> dict[str, object]:
        if not self._window:
            return {"status": "warming_up", "window_size": 0, "alerts": []}

        window_df = pd.DataFrame(self._window)
        alerts: list[dict[str, float | str]] = []
        for feature, stats in self.baseline_stats.items():
            if feature not in window_df.columns:
                continue
            baseline_mean = float(stats["mean"])
            baseline_std = float(stats["std"] or 1e-6)
            current_mean = float(window_df[feature].mean())
            z_score = float((current_mean - baseline_mean) / max(baseline_std, 1e-6))
            if abs(z_score) >= self.z_threshold:
                alerts.append(
                    {
                        "feature": feature,
                        "baseline_mean": round(baseline_mean, 6),
                        "current_mean": round(current_mean, 6),
                        "z_score": round(z_score, 4),
                    }
                )

        status = "drift_detected" if alerts else "stable"
        if alerts:
            logger.warning("drift_alert", extra={"alerts": alerts, "window_size": len(self._window)})

        return {
            "status": status,
            "window_size": len(self._window),
            "alerts": alerts,
            "z_threshold": self.z_threshold,
            "tracked_features": len(self.baseline_stats),
        }
