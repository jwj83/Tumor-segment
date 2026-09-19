from __future__ import annotations

from data.structures import CompetitionDataset
from pipeline.context import PipelineContext
from tasks.base import DatasetTask
from tasks.results import DuplicatePair, DuplicateResult


class DummyDuplicateTask(DatasetTask[DuplicateResult]):
    name = "goal2_duplicate"

    def predict(
        self,
        dataset: CompetitionDataset,
        contexts: dict[str, PipelineContext],
    ) -> DuplicateResult:
        accessions = sorted(contexts)
        if len(accessions) < 2:
            return DuplicateResult(pairs=())
        return DuplicateResult(
            pairs=(
                DuplicatePair(
                    left_accession=accessions[0],
                    right_accession=accessions[1],
                    probability=0.0,
                ),
            )
        )
