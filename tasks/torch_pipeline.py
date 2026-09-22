"""Competition pipeline backed by the trainable PyTorch Goal 3/4/5 Tasks.

Use this factory after installing the approved PyTorch environment:

    COMPETITION_PIPELINE_FACTORY=tasks.torch_pipeline:build_pipeline
"""

from __future__ import annotations

import os
from pathlib import Path

from pipeline.inference import InferencePipeline, StudyTaskBinding
from tasks.dummy.dataset_tasks import DummyDuplicateTask
from tasks.dummy.study_tasks import DummyGoal1Task, DummyStitchedTask
from tasks.goal3.task import Goal3Task
from tasks.goal4.task import Goal4Task
from tasks.goal5.task import Goal5Task


def build_pipeline() -> InferencePipeline:
    checkpoint_root = Path(
        os.environ.get(
            "COMPETITION_CHECKPOINT_ROOT",
            "/2026aicompetition/workspace/checkpoint",
        )
    )
    return InferencePipeline(
        study_tasks=(
            StudyTaskBinding("goal1", DummyGoal1Task()),
            StudyTaskBinding("goal2_stitched", DummyStitchedTask()),
            StudyTaskBinding("goal3", Goal3Task(_checkpoint_path("GOAL3_CHECKPOINT", checkpoint_root / "goal3.pt"))),
            StudyTaskBinding("goal5", Goal5Task(_checkpoint_path("GOAL5_CHECKPOINT", checkpoint_root / "goal5.pt"))),
            StudyTaskBinding("goal4", Goal4Task(_checkpoint_path("GOAL4_CHECKPOINT", checkpoint_root / "goal4.pt"))),
        ),
        duplicate_task=DummyDuplicateTask(),
    )


def _checkpoint_path(variable: str, default: Path) -> str | None:
    """Use the fixed team path when present, while allowing local random smoke tests."""
    configured = os.environ.get(variable)
    if configured:
        return configured
    return str(default) if default.is_file() else None
