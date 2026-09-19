from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np

from data.structures import Series
from pipeline.aggregator import PredictionAggregator
from pipeline.context import PipelineContext
from tasks.results import DuplicatePair, DuplicateResult


class OutputWriter:
    def __init__(
        self,
        answer_root: Path,
        aggregator: PredictionAggregator | None = None,
    ) -> None:
        self.answer_root = answer_root
        self.aggregator = aggregator or PredictionAggregator()

    def write_staging(
        self,
        evaluation_id: str,
        contexts: dict[str, PipelineContext],
        duplicates: DuplicateResult,
    ) -> Path:
        safe_evaluation_id = _safe_component(evaluation_id, "evaluation_id")
        self.answer_root.mkdir(parents=True, exist_ok=True)
        final_dir = self.answer_root / safe_evaluation_id
        if final_dir.exists():
            raise FileExistsError(f"evaluation output already exists: {final_dir}")

        staging = self.answer_root / f".{safe_evaluation_id}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        self._write_duplicate_pairs(staging, contexts, duplicates)
        for accession, context in sorted(contexts.items()):
            accession_dir = staging / _safe_component(accession, "accession_number")
            accession_dir.mkdir()
            self._write_masks(accession_dir, context)
            self._write_json(
                accession_dir / "prediction.json",
                self.aggregator.build(context),
            )
        return staging

    def publish(self, staging: Path, evaluation_id: str) -> Path:
        final_dir = self.answer_root / _safe_component(evaluation_id, "evaluation_id")
        if final_dir.exists():
            raise FileExistsError(f"evaluation output already exists: {final_dir}")
        staging.rename(final_dir)
        return final_dir

    def _write_masks(self, accession_dir: Path, context: PipelineContext) -> None:
        result = context.goal5
        if result is None:
            raise ValueError("Goal5 result is missing")

        planned: dict[Path, tuple[np.ndarray, Series]] = {}
        for mask, series_uid in (
            (result.core_mask, result.core_source_series_uid),
            (result.flair_mask, result.flair_source_series_uid),
        ):
            series = context.study.series_by_uid(series_uid)
            safe_uid = _safe_component(series_uid, "series_uid")
            path = accession_dir / safe_uid / f"{safe_uid}.nii.gz"
            if path in planned and not np.array_equal(planned[path][0], mask):
                raise ValueError(
                    f"core and flair masks target the same series but differ: {series_uid}"
                )
            planned[path] = (mask, series)

        for path, (mask, series) in planned.items():
            path.parent.mkdir()
            image = nib.Nifti1Image(
                np.asarray(mask, dtype=np.uint8),
                np.asarray(series.affine, dtype=np.float64),
            )
            image.set_data_dtype(np.uint8)
            nib.save(image, str(path))

    def _write_duplicate_pairs(
        self,
        staging: Path,
        contexts: dict[str, PipelineContext],
        duplicates: DuplicateResult,
    ) -> None:
        pairs = self._normalize_pairs(duplicates.pairs)
        if not pairs:
            accessions = sorted(contexts)
            if len(accessions) < 2:
                raise ValueError(
                    "duplicate_pairs.jsonl requires at least two studies for one valid pair"
                )
            pairs = [DuplicatePair(accessions[0], accessions[1], 0.0)]

        path = staging / "duplicate_pairs.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for pair in pairs:
                line = {
                    "StudyUID": pair.left_accession,
                    "StudyUID_dup": pair.right_accession,
                    "PairProb": pair.probability,
                }
                handle.write(
                    json.dumps(line, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _normalize_pairs(pairs: tuple[DuplicatePair, ...]) -> list[DuplicatePair]:
        best: dict[tuple[str, str], DuplicatePair] = {}
        for pair in pairs:
            key = tuple(sorted((pair.left_accession, pair.right_accession)))
            if key[0] == key[1]:
                continue
            current = best.get(key)
            if current is None or pair.probability > current.probability:
                best[key] = DuplicatePair(key[0], key[1], pair.probability)

        counts: dict[str, int] = {}
        selected: list[DuplicatePair] = []
        for pair in sorted(
            best.values(),
            key=lambda item: (-item.probability, item.left_accession, item.right_accession),
        ):
            if counts.get(pair.left_accession, 0) >= 200:
                continue
            if counts.get(pair.right_accession, 0) >= 200:
                continue
            selected.append(pair)
            counts[pair.left_accession] = counts.get(pair.left_accession, 0) + 1
            counts[pair.right_accession] = counts.get(pair.right_accession, 0) + 1
        return selected

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def _safe_component(value: str, label: str) -> str:
    text = str(value).strip()
    if (
        not text
        or text in {".", ".."}
        or Path(text).is_absolute()
        or re.search(r"[\\/\x00-\x1f]", text)
    ):
        raise ValueError(f"unsafe {label}: {value!r}")
    return text
