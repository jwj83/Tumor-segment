from __future__ import annotations

import math
import time
from dataclasses import dataclass

from core.exceptions import InvalidTaskResultError
from data.structures import CompetitionDataset
from pipeline.context import PipelineContext
from tasks.base import DatasetTask, StudyTask
from tasks.dummy.dataset_tasks import DummyDuplicateTask
from tasks.dummy.study_tasks import (
    DummyGoal1Task,
    DummyGoal3Task,
    DummyGoal4Task,
    DummyGoal5Task,
    DummyStitchedTask,
)
from tasks.results import DuplicateResult


@dataclass(frozen=True)
class StudyTaskBinding:
    context_field: str
    task: StudyTask[object]


class InferencePipeline:
    def __init__(
        self,
        study_tasks: tuple[StudyTaskBinding, ...] | None = None,
        duplicate_task: DatasetTask[DuplicateResult] | None = None,
    ) -> None:
        self.study_tasks = study_tasks or (
            StudyTaskBinding("goal1", DummyGoal1Task()),
            StudyTaskBinding("goal2_stitched", DummyStitchedTask()),
            StudyTaskBinding("goal3", DummyGoal3Task()),
            StudyTaskBinding("goal5", DummyGoal5Task()),
            StudyTaskBinding("goal4", DummyGoal4Task()),
        )
        self.duplicate_task = duplicate_task or DummyDuplicateTask()
        for binding in self.study_tasks:
            binding.task.load_model()
        self.duplicate_task.load_model()

    def run(
        self,
        dataset: CompetitionDataset,
    ) -> tuple[dict[str, PipelineContext], DuplicateResult]:
        contexts: dict[str, PipelineContext] = {}
        for study in dataset.studies:
            started = time.perf_counter()
            context = PipelineContext(study=study)
            for binding in self.study_tasks:
                result = binding.task.predict(context)
                self._validate_result_numbers(result, binding.task.name)
                setattr(context, binding.context_field, result)
            context.processing_time_ms = round((time.perf_counter() - started) * 1000)
            contexts[study.accession_number] = context

        duplicates = self.duplicate_task.predict(dataset, contexts)
        self._validate_result_numbers(duplicates, self.duplicate_task.name)
        return contexts, duplicates

    @classmethod
    def _validate_result_numbers(cls, value: object, task_name: str) -> None:
        """Reject non-finite or out-of-range probability-like values."""
        from dataclasses import fields, is_dataclass

        def walk(item: object, field_name: str = "") -> None:
            if is_dataclass(item):
                for descriptor in fields(item):
                    walk(getattr(item, descriptor.name), descriptor.name)
            elif isinstance(item, dict):
                for key, child in item.items():
                    walk(child, str(key))
            elif isinstance(item, (tuple, list)):
                for child in item:
                    walk(child, field_name)
            elif isinstance(item, float):
                if not math.isfinite(item):
                    raise InvalidTaskResultError(
                        f"{task_name} produced non-finite {field_name}"
                    )
                if "prob" in field_name.lower() and not 0.0 <= item <= 1.0:
                    raise InvalidTaskResultError(
                        f"{task_name} produced out-of-range {field_name}: {item}"
                    )

        walk(value)
