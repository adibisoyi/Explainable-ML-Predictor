from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictionMonitor:
    log_path: Path
    total_predictions: int = 0
    positive_predictions: int = 0

    def log_prediction(self, predicted_class: int, predicted_probability: float) -> None:
        self.total_predictions += 1
        if predicted_class == 1:
            self.positive_predictions += 1
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "predicted_class": predicted_class,
                        "predicted_probability": predicted_probability,
                    }
                )
                + "\n"
            )

    def summary(self) -> dict:
        positive_rate = self.positive_predictions / self.total_predictions if self.total_predictions else 0.0
        return {
            "total_predictions": self.total_predictions,
            "positive_predictions": self.positive_predictions,
            "positive_rate": round(positive_rate, 4),
            "log_path": str(self.log_path),
        }
