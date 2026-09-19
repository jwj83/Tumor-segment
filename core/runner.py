from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from app.callback import CompetitionCallback
from core.config import Settings
from core.registry import build_pipeline
from data.loader import DatasetLoader
from observability.competition_logger import CompetitionLogger
from output.validator import OutputValidator
from output.writer import OutputWriter
from pipeline.inference import InferencePipeline


@dataclass(frozen=True)
class EvaluationJob:
    request_id: str
    evaluation_id: str
    dataset_path: Path


class EvaluationRunner:
    def __init__(
        self,
        settings: Settings,
        *,
        loader: DatasetLoader | None = None,
        pipeline: InferencePipeline | None = None,
        validator: OutputValidator | None = None,
        callback: CompetitionCallback | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader or DatasetLoader()
        self.pipeline = pipeline or build_pipeline(settings.pipeline_factory)
        self.writer = OutputWriter(settings.answer_root)
        self.validator = validator or OutputValidator()
        self.callback = callback or CompetitionCallback(
            settings.callback_url,
            settings.callback_timeout_seconds,
            settings.callback_attempts,
        )
        self.logger = CompetitionLogger(settings.log_root)

    def run(self, job: EvaluationJob, *, send_callback: bool = True) -> Path:
        started = time.perf_counter()
        data_source = str(job.dataset_path)
        self.logger.write(
            request_id=job.request_id,
            evaluation_id=job.evaluation_id,
            phase="test",
            message="evaluation_started",
            data_source=data_source,
        )
        staging: Path | None = None
        try:
            dataset = self.loader.load(job.dataset_path)
            contexts, duplicates = self.pipeline.run(dataset)
            staging = self.writer.write_staging(
                job.evaluation_id,
                contexts,
                duplicates,
            )
            self.validator.validate(staging, dataset)
            output_dir = self.writer.publish(staging, job.evaluation_id)
            staging = None
            duration_ms = round((time.perf_counter() - started) * 1000)
            self.logger.write(
                request_id=job.request_id,
                evaluation_id=job.evaluation_id,
                phase="test",
                message="evaluation_completed",
                data_source=data_source,
                duration_ms=duration_ms,
                study_count=len(dataset.studies),
                pred_path=str(output_dir),
            )
            if send_callback:
                self.callback.send_success(
                    job.request_id,
                    job.evaluation_id,
                    output_dir,
                )
            return output_dir
        except Exception as exc:
            self.logger.write(
                request_id=job.request_id,
                evaluation_id=job.evaluation_id,
                phase="test",
                message="evaluation_failed",
                data_source=data_source,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if staging and staging.exists():
                shutil.rmtree(staging)
            raise
