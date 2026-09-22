from __future__ import annotations

"""Dependency-free Goal 3 smoke-test classifier.

The real version can replace ``load_model`` and ``predict`` with a MedicalNet
3-D backbone and a binary classification head while keeping Goal3Result
unchanged.
"""

from pathlib import Path

import numpy as np

from data.structures import Series
from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.results import Goal3Result


class HeuristicGoal3Task(StudyTask[Goal3Result]):
    name = "goal3"

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.weights_path = Path(weights_path).expanduser() if weights_path else None
        self.model = None

    def load_model(self) -> None:
        if self.weights_path is not None and not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Goal3 weights were configured but not found: {self.weights_path}"
            )
        self.model = "heuristic-baseline"

    def predict(self, context: PipelineContext) -> Goal3Result:
        t1ce = _select_series(context.study.series, ("t1ce", "t1+c", "t1 enhanced", "t1"))
        flair = _select_series(context.study.series, ("flair", "t2flair", "t2"))
        probability = 0.5 * _lesion_score(t1ce.image) + 0.5 * _lesion_score(flair.image)
        return Goal3Result(tumor_probability=float(np.clip(probability, 0.0, 1.0)))


def _select_series(series: tuple[Series, ...], hints: tuple[str, ...]) -> Series:
    for hint in hints:
        for item in series:
            text = " ".join(
                (
                    item.modality or "",
                    str(item.metadata.get("SeriesDescription", "")),
                    item.series_uid,
                )
            ).lower()
            if hint in text:
                return item
    return series[0]


def _lesion_score(image: np.ndarray) -> float:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    spread = float(np.std(finite))
    if spread <= 1e-6:
        return 0.0
    # A high upper tail relative to scanner noise is a useful smoke-test
    # signal for the synthetic lesions. The result remains a probability.
    upper_tail = float(np.mean(finite > np.percentile(finite, 97.0)))
    contrast = float(np.clip((float(np.percentile(finite, 99.5)) - median) / (6.0 * spread), 0.0, 1.0))
    return float(np.clip(0.5 * upper_tail + 0.5 * contrast, 0.0, 1.0))
