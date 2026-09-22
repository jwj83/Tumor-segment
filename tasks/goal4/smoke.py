from __future__ import annotations

"""A dependency-free Goal 4 smoke-test task.

This is deliberately small and deterministic.  It converts coarse image
statistics into valid ``Goal4Result`` values so that the complete output
contract can be exercised before a trained model is available.  It is not a
clinically meaningful predictor.
"""

import numpy as np
from pathlib import Path

from data.structures import Series
from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.results import BinaryResult, CategoricalResult, Goal4Result


class HeuristicGoal4Task(StudyTask[Goal4Result]):
    """Return format-valid attributes from simple percentile statistics."""

    name = "goal4"

    def __init__(self, weights_path: str | Path | None = None) -> None:
        # The smoke-test task has no learned parameters yet, but it keeps the
        # same lifecycle as the real task: load_model() is called once when
        # the service starts and predict() is called once per Study.
        self.weights_path = Path(weights_path).expanduser() if weights_path else None
        self.model = None

    def load_model(self) -> None:
        if self.weights_path is not None and not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Goal4 weights were configured but not found: {self.weights_path}"
            )
        # Replace this line with torch.load()/model.load_state_dict() when the
        # trained multi-head classifier is ready.
        self.model = "heuristic-baseline"

    def predict(self, context: PipelineContext) -> Goal4Result:
        t1ce = _select_series(context.study.series, ("t1ce", "t1+c", "t1 enhanced"))
        flair = _select_series(context.study.series, ("flair", "t2flair"))
        t2 = _select_series(context.study.series, ("t2",))

        enhancement_score = _signal_score(t1ce.image)
        flair_score = _signal_score(flair.image)
        t2_score = _signal_score(t2.image)
        irregular_score = float(np.clip((enhancement_score + flair_score) / 2.0, 0.0, 1.0))

        enhancement = _binary(enhancement_score)
        necrosis = _binary(max(0.0, enhancement_score - 0.35))
        cystic_change = _binary(max(0.0, flair_score - 0.45))
        hemorrhage = _binary(max(0.0, enhancement_score - 0.70))
        calcification = _binary(max(0.0, 0.35 - t2_score))
        margin_clear = _binary(1.0 - irregular_score)
        lobulation = _binary(irregular_score)

        morphology = _categorical(
            "Irregular" if irregular_score >= 0.5 else "Regular",
            ("Regular", "Irregular"),
            irregular_score,
        )
        who_grade_index = int(np.clip(round(enhancement_score * 3.0), 0, 3))
        who_grade = _categorical(
            str(who_grade_index + 1),
            ("1", "2", "3", "4"),
            enhancement_score,
        )
        enhancement_pattern = _categorical(
            "Ring" if enhancement_score >= 0.55 else "None",
            (
                "None",
                "Ring",
                "RimEnhancing",
                "Nodular",
                "GroundGlass",
                "Gyriform",
                "Multifocal",
                "Other",
            ),
            enhancement_score,
        )
        signal_t2wi = _signal_category(t2_score)
        signal_flair = _signal_category(flair_score)

        return Goal4Result(
            # ``Other`` is part of the baseline enum and is intentionally
            # conservative until the official scoring enum is confirmed.
            location="Other",
            morphology=morphology,
            who_grade=who_grade,
            enhancement=enhancement,
            enhancement_pattern=enhancement_pattern,
            necrosis=necrosis,
            cystic_change=cystic_change,
            hemorrhage=hemorrhage,
            calcification=calcification,
            margin_clear=margin_clear,
            lobulation=lobulation,
            signal_t2wi=signal_t2wi,
            signal_flair=signal_flair,
            conclusion=(
                "Local smoke-test heuristic; replace with the trained Goal 4 model."
            ),
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


def _signal_score(image: np.ndarray) -> float:
    """Estimate how much high-intensity signal is present, in [0, 1]."""
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    # The fraction above the median is stable across arbitrary scanner scales.
    median = float(np.median(finite))
    high_fraction = float(np.mean(finite > median))
    spread = float(np.std(finite))
    if spread <= 1e-6:
        return 0.0
    # Combining the high-tail fraction with normalized spread gives synthetic
    # BraTS examples a useful, deterministic variation while staying bounded.
    normalized_spread = float(np.clip(spread / (abs(float(np.mean(finite))) + spread), 0.0, 1.0))
    return float(np.clip(0.5 * high_fraction + 0.5 * normalized_spread, 0.0, 1.0))


def _binary(probability: float) -> BinaryResult:
    probability = float(np.clip(probability, 0.0, 1.0))
    return BinaryResult(present=probability >= 0.5, probability=probability)


def _categorical(predicted: str, labels: tuple[str, ...], score: float) -> CategoricalResult:
    score = float(np.clip(score, 0.0, 1.0))
    # Emit a one-hot distribution to satisfy the current validator while
    # retaining a deterministic predicted class.
    probabilities = {label: 1.0 if label == predicted else 0.0 for label in labels}
    return CategoricalResult(predicted=predicted, probabilities=probabilities)


def _signal_category(score: float) -> CategoricalResult:
    label = "High" if score >= 0.66 else "Iso" if score >= 0.33 else "Low"
    return _categorical(label, ("Low", "Iso", "High"), score)
