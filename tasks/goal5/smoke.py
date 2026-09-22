from __future__ import annotations

"""A dependency-free Goal 5 segmentation smoke-test task.

The task thresholds each source image in its original grid.  The resulting
arrays satisfy the frozen Goal5Result contract and are suitable for exercising
Writer/Validator with synthetic data.  Replace this implementation with the
trained segmentation model and inverse-resampling step for the submission.
"""

import numpy as np
from pathlib import Path

from core.exceptions import MissingSeriesError
from data.structures import Series
from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.results import Goal5Result


class HeuristicGoal5Task(StudyTask[Goal5Result]):
    name = "goal5"

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.weights_path = Path(weights_path).expanduser() if weights_path else None
        self.model = None

    def load_model(self) -> None:
        if self.weights_path is not None and not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Goal5 weights were configured but not found: {self.weights_path}"
            )
        # Replace this with nnUNet/3D U-Net checkpoint loading later.  Keeping
        # it in the framework lifecycle makes the eventual replacement local
        # to this task and avoids changing the HTTP or output contract.
        self.model = "heuristic-baseline"

    def predict(self, context: PipelineContext) -> Goal5Result:
        if not context.study.series:
            raise MissingSeriesError(
                f"study {context.study.accession_number!r} has no series"
            )
        t1ce = _select_series(context.study.series, ("t1ce", "t1+c", "t1 enhanced", "t1"))
        flair = _select_series(context.study.series, ("flair", "t2flair", "t2"))

        flair_mask = _percentile_mask(flair.image, 88.0)
        core_mask = _percentile_mask(t1ce.image, 92.0) & flair_mask
        # Keep masks binary uint8 and exactly on the original source grids.
        return Goal5Result(
            core_mask=np.asarray(core_mask, dtype=np.uint8),
            core_source_series_uid=t1ce.series_uid,
            flair_mask=np.asarray(flair_mask, dtype=np.uint8),
            flair_source_series_uid=flair.series_uid,
        )


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


def _percentile_mask(image: np.ndarray, percentile: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    threshold = float(np.percentile(values[finite], percentile))
    return (finite & (values >= threshold)).astype(np.uint8)
