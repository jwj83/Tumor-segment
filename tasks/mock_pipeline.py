"""Local smoke-test pipeline using deterministic Goal 3/4/5 heuristics.

Enable it with:

    COMPETITION_PIPELINE_FACTORY=tasks.mock_pipeline:build_pipeline \
      python scripts/local_eval.py ...

It intentionally retains the existing Goal 1/2 and duplicate baselines.
"""

from __future__ import annotations

import os

from pipeline.inference import InferencePipeline, StudyTaskBinding
from tasks.dummy.dataset_tasks import DummyDuplicateTask
from tasks.dummy.study_tasks import (
    DummyGoal1Task,
    DummyStitchedTask,
)
from tasks.goal3.smoke import HeuristicGoal3Task
from tasks.goal4.smoke import HeuristicGoal4Task
from tasks.goal5.smoke import HeuristicGoal5Task


def build_pipeline() -> InferencePipeline:
    goal3_weights = os.environ.get("GOAL3_WEIGHTS") or None
    goal4_weights = os.environ.get("GOAL4_WEIGHTS") or None
    goal5_weights = os.environ.get("GOAL5_WEIGHTS") or None
    return InferencePipeline(
        study_tasks=(
            StudyTaskBinding("goal1", DummyGoal1Task()),
            StudyTaskBinding("goal2_stitched", DummyStitchedTask()),
            StudyTaskBinding("goal3", HeuristicGoal3Task(goal3_weights)),
            # Goal 4 can inspect the same source series independently.  Keep
            # Goal 5 first to match the production pipeline ordering.
            StudyTaskBinding("goal5", HeuristicGoal5Task(goal5_weights)),
            StudyTaskBinding("goal4", HeuristicGoal4Task(goal4_weights)),
        ),
        duplicate_task=DummyDuplicateTask(),
    )
