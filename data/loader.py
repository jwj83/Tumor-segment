from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import nibabel as nib
import numpy as np

from core.exceptions import InvalidInputError
from data.structures import CompetitionDataset, Series, Study


_NIFTI_SUFFIXES = (".nii", ".nii.gz")
_MASK_HINTS = ("mask", "seg", "label", "roi")


def _nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def _clean_identifier(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return re.sub(r"[\\/\x00-\x1f]", "_", text)


class DatasetLoader:
    """Load a NIfTI tree into the competition domain model."""

    def load(self, dataset_path: str | Path) -> CompetitionDataset:
        """Compatibility entry point for callers that need the full dataset."""
        root = Path(dataset_path).expanduser().resolve()
        return CompetitionDataset(
            root,
            tuple(self.iter_studies(root)),
            {"format": "nifti"},
        )

    def iter_studies(self, dataset_path: str | Path) -> Iterator[Study]:
        """Discover file paths, then load one study at a time."""
        root = Path(dataset_path).expanduser().resolve()
        if not root.is_dir():
            raise InvalidInputError(f"dataset_path is not a directory: {root}")

        nifti_files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name.lower().endswith(_NIFTI_SUFFIXES)
            and not any(hint in path.name.lower() for hint in _MASK_HINTS)
        )
        if not nifti_files:
            raise InvalidInputError(f"no readable NIfTI images under {root}")
        yield from self._iter_nifti(root, nifti_files)

    def _iter_nifti(
        self,
        root: Path,
        files: Iterable[Path],
    ) -> Iterator[Study]:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in files:
            relative = path.relative_to(root)
            accession = relative.parts[0] if len(relative.parts) > 1 else _nifti_stem(path)
            grouped[_clean_identifier(accession, "study")].append(path)

        for accession, paths in sorted(grouped.items()):
            yield Study(
                accession_number=accession,
                series=tuple(
                    self._read_nifti_series(root, accession, path) for path in paths
                ),
            )

    def _read_nifti_series(
        self,
        root: Path,
        accession: str,
        path: Path,
    ) -> Series:
        relative = path.relative_to(root)
        if len(relative.parts) == 1 or path.parent == root / relative.parts[0]:
            series_uid = _nifti_stem(path)
        else:
            series_uid = path.parent.name
        sidecar = path.with_name(_nifti_stem(path) + ".json")
        try:
            metadata: dict[str, Any] = {}
            if sidecar.is_file():
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))

            image = nib.load(str(path))
            array = np.asanyarray(image.dataobj)
            array = np.squeeze(array)
            if array.ndim != 3:
                raise InvalidInputError(f"NIfTI image must be 3-D: {path} -> {array.shape}")

            uid = _clean_identifier(
                metadata.get("SeriesInstanceUID"),
                series_uid,
            )
            description = str(
                metadata.get("SeriesDescription")
                or metadata.get("ProtocolName")
                or series_uid
            )
            return Series(
                series_uid=uid,
                modality=description,
                image=np.asarray(array),
                affine=np.asarray(image.affine, dtype=np.float64),
                source_path=path,
                metadata=metadata,
            )
        except Exception as exc:
            detail = (
                str(exc)
                if isinstance(exc, InvalidInputError)
                else f"{type(exc).__name__}: {exc}"
            )
            raise InvalidInputError(
                f"cannot load NIfTI study={accession!r} series={series_uid!r} "
                f"path={path}: {detail}"
            ) from exc
