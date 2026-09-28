"""Sequence-level lazy dataset with accession-level split and aggregation metadata."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from training.resize import resize_volume


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def canonical_modality(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", value.lower())
    if "flair" in text:
        return "flair"
    if "t1ce" in text or "t1c" in text or "enh" in text:
        return "t1ce"
    if "t2" in text:
        return "t2"
    if "t1" in text:
        return "t1"
    return "unknown"


@dataclass(frozen=True)
class SequenceRecord:
    accession: str
    series_uid: str
    modality: str
    path: str
    label: float
    split: str = "train"


def build_sequence_records(file_index: str | Path, labels_csv: str | Path, label_column: str) -> list[SequenceRecord]:
    rows = _read_csv(Path(file_index))
    labels = {
        (row.get("AccessionNumber", ""), row.get("SeriesUid", "")): row.get(label_column, "")
        for row in _read_csv(Path(labels_csv))
    }
    records: list[SequenceRecord] = []
    for row in rows:
        if row.get("status") not in {"series_and_mask_found", "series_found_mask_missing"}:
            continue
        key = (row.get("AccessionNumber", ""), row.get("SeriesUid", ""))
        raw_label = labels.get(key, "")
        if not row.get("series_path") or raw_label == "":
            continue
        try:
            label = float(raw_label)
        except ValueError:
            continue
        records.append(
            SequenceRecord(
                accession=key[0], series_uid=key[1],
                modality=canonical_modality(row.get("SeriesType", "")),
                path=row["series_path"], label=label,
            )
        )
    return records


def assign_accession_splits(records: list[SequenceRecord], val_fraction: float, test_fraction: float, seed: int) -> None:
    import random
    rng = random.Random(seed)
    by_accession: dict[str, list[SequenceRecord]] = {}
    for record in records:
        by_accession.setdefault(record.accession, []).append(record)
    accessions = list(by_accession)
    rng.shuffle(accessions)
    n_test = round(len(accessions) * test_fraction)
    n_val = round(len(accessions) * val_fraction)
    for index, accession in enumerate(accessions):
        split = "test" if index < n_test else "val" if index < n_test + n_val else "train"
        for record in by_accession[accession]:
            object.__setattr__(record, "split", split)


def normalize(array: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(array, dtype=np.float32), copy=True)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.float32)
    lo, hi = np.percentile(values[finite], (1, 99))
    return np.clip((values - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)


class SequenceNiftiDataset(Dataset):
    def __init__(self, records: list[SequenceRecord], split: str, target_shape: tuple[int, int, int] = (32, 32, 16)):
        self.records = [record for record in records if record.split == split]
        self.target_shape = target_shape

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = normalize(np.asarray(nib.load(record.path).dataobj, dtype=np.float32))
        image = resize_volume(image, self.target_shape, is_mask=False)
        return {
            "image": torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "accession": record.accession,
            "series_uid": record.series_uid,
            "modality": record.modality,
        }
