"""Shared inference contract between the research pipeline and backend platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class Prediction:
    """Single-image anomaly detection prediction matching ``docs/03_ARCHITECTURE.md``."""

    score: float
    anomaly_map: np.ndarray
    confidence: float | None = None
    scale_maps: dict[str, np.ndarray] | None = None
    latency_ms: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


class AnomalyModel(Protocol):
    """Protocol implemented by every baseline adapter and custom fusion model."""

    name: str

    def load(self, artifact_dir: Path, device: str) -> None:
        """Load trained weights/memory bank and calibration artifacts from ``artifact_dir``."""
        ...

    def predict(self, image: np.ndarray) -> Prediction:
        """Run inference on one preprocessed or raw RGB image array."""
        ...
