from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from core.config import Settings
from core.exceptions import OutputValidationError
from core.runner import EvaluationJob, EvaluationRunner
from data.loader import DatasetLoader
from output.validator import OutputValidator


class StreamingTest(unittest.TestCase):
    def test_loader_reads_only_the_current_study(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path = Path(temporary) / "dataset"
            self._make_dataset(dataset_path)

            with patch("data.loader.nib.load", wraps=nib.load) as load:
                studies = DatasetLoader().iter_studies(dataset_path)
                self.assertEqual(0, load.call_count)
                self.assertEqual("ACC001", next(studies).accession_number)
                self.assertEqual(2, load.call_count)
                self.assertEqual("ACC002", next(studies).accession_number)
                self.assertEqual(4, load.call_count)

    def test_runner_validates_before_loading_the_next_study(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset"
            self._make_dataset(dataset_path)
            events: list[str] = []

            class RecordingLoader(DatasetLoader):
                def iter_studies(self, path: str | Path):
                    for study in super().iter_studies(path):
                        events.append(f"load {study.accession_number}")
                        yield study

            class RecordingValidator(OutputValidator):
                def validate_study(self, directory: Path, study: object) -> None:
                    super().validate_study(directory, study)
                    events.append(f"validate {study.accession_number}")

            runner = EvaluationRunner(
                self._settings(root),
                loader=RecordingLoader(),
                validator=RecordingValidator(),
            )
            runner.run(
                EvaluationJob("request-order", "evaluation-order", dataset_path),
                send_callback=False,
            )

            self.assertEqual(
                [
                    "load ACC001",
                    "validate ACC001",
                    "load ACC002",
                    "validate ACC002",
                ],
                events,
            )

    def test_failed_study_removes_staging_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset"
            self._make_dataset(dataset_path)

            class FailingValidator(OutputValidator):
                def validate_study(self, directory: Path, study: object) -> None:
                    raise OutputValidationError("expected failure")

            settings = self._settings(root)
            runner = EvaluationRunner(settings, validator=FailingValidator())
            with self.assertRaises(OutputValidationError):
                runner.run(
                    EvaluationJob("request-fail", "evaluation-fail", dataset_path),
                    send_callback=False,
                )

            self.assertFalse((settings.answer_root / "evaluation-fail").exists())
            self.assertEqual([], list(settings.answer_root.glob(".*.tmp-*")))

    @staticmethod
    def _settings(root: Path) -> Settings:
        return Settings(
            workspace=root / "workspace",
            answer_root=root / "workspace" / "answer",
            log_root=root / "workspace" / "logs",
            callback_url=None,
        )

    @staticmethod
    def _make_dataset(root: Path) -> None:
        for accession in ("ACC001", "ACC002"):
            for series_uid in ("FLAIR", "T1CE"):
                directory = root / accession / series_uid
                directory.mkdir(parents=True)
                nib.save(
                    nib.Nifti1Image(
                        np.zeros((2, 3, 4), dtype=np.float32),
                        np.eye(4),
                    ),
                    str(directory / f"{series_uid}.nii.gz"),
                )


if __name__ == "__main__":
    unittest.main()
