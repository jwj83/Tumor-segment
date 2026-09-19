from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from data.structures import CompetitionDataset

if False:  # pragma: no cover - imports for type checkers without a cycle
    from pipeline.context import PipelineContext


ResultT = TypeVar("ResultT")


class StudyTask(ABC, Generic[ResultT]):
    name: str

    def load_model(self) -> None:
        """Load weights once. Dummy and stateless tasks may do nothing."""

    @abstractmethod
    def predict(self, context: "PipelineContext") -> ResultT:
        raise NotImplementedError


class DatasetTask(ABC, Generic[ResultT]):
    name: str

    def load_model(self) -> None:
        """Load weights once. Dummy and stateless tasks may do nothing."""

    @abstractmethod
    def predict(
        self,
        dataset: CompetitionDataset,
        contexts: dict[str, "PipelineContext"],
    ) -> ResultT:
        raise NotImplementedError

