from __future__ import annotations

from core.exceptions import InvalidTaskResultError
from pipeline.context import PipelineContext
from tasks.results import BinaryResult, CategoricalResult


class PredictionAggregator:
    def build(self, context: PipelineContext) -> dict[str, object]:
        if not all(
            (
                context.goal1,
                context.goal2_stitched,
                context.goal3,
                context.goal4,
                context.goal5,
            )
        ):
            raise InvalidTaskResultError(
                f"incomplete results for {context.study.accession_number}"
            )

        goal4 = context.goal4
        goal5 = context.goal5
        prediction: dict[str, object] = {
            "AccessionNumber": context.study.accession_number,
            "IsNotHumanBodyProb": context.goal1.not_human_probability,
            "IsStitchedProb": context.goal2_stitched.stitched_probability,
            "ProcessingTime_ms": context.processing_time_ms,
            "SegmentationMaskURI": {
                "core": self._mask_uri(goal5.core_source_series_uid),
                "flair": self._mask_uri(goal5.flair_source_series_uid),
            },
            "Prediction": {
                "TumorProbability": context.goal3.tumor_probability,
                "Location": goal4.location,
                "Morphology": self._category(goal4.morphology),
                "WHO_Grade": self._category(goal4.who_grade),
                "Enhancement": self._binary(
                    goal4.enhancement,
                    "EnhancementProbability",
                ),
                "EnhancementPattern": self._category(goal4.enhancement_pattern),
                "Necrosis": self._binary(goal4.necrosis, "NecrosisProbability"),
                "CysticChange": self._binary(
                    goal4.cystic_change,
                    "CysticChangeProbability",
                ),
                "Hemorrhage": self._binary(
                    goal4.hemorrhage,
                    "HemorrhageProbability",
                ),
                "Calcification": self._binary(
                    goal4.calcification,
                    "CalcificationProbability",
                ),
                "Margin": {
                    "clear": goal4.margin_clear.present,
                    "MarginClearProbability": goal4.margin_clear.probability,
                },
                "Lobulation": self._binary(
                    goal4.lobulation,
                    "LobulationProbability",
                ),
                "Signal_T2WI": self._category(goal4.signal_t2wi),
                "Signal_FLAIR": self._category(goal4.signal_flair),
            },
            "Interpretation": {"Conclusion": goal4.conclusion},
        }
        if goal4.attention_map_uri:
            prediction["Interpretation"]["AttentionMapURI"] = goal4.attention_map_uri
        return prediction

    @staticmethod
    def _mask_uri(series_uid: str) -> str:
        return f"./{series_uid}/{series_uid}.nii.gz"

    @staticmethod
    def _category(result: CategoricalResult) -> dict[str, object]:
        return {
            "predicted": result.predicted,
            "probabilities": result.probabilities,
        }

    @staticmethod
    def _binary(result: BinaryResult, probability_key: str) -> dict[str, object]:
        return {
            "present": result.present,
            probability_key: result.probability,
        }

