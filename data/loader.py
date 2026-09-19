from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pydicom

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
    """Load either a NIfTI tree or a DICOM tree into one domain model."""

    def load(self, dataset_path: str | Path) -> CompetitionDataset:
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
        if nifti_files:
            return self._load_nifti(root, nifti_files)

        return self._load_dicom(root)

    def _load_nifti(
        self,
        root: Path,
        files: Iterable[Path],
    ) -> CompetitionDataset:
        grouped: dict[str, list[Series]] = defaultdict(list)
        for path in files:
            relative = path.relative_to(root)
            accession = relative.parts[0] if len(relative.parts) > 1 else _nifti_stem(path)
            if len(relative.parts) == 1 or path.parent == root / accession:
                series_uid = _nifti_stem(path)
            else:
                series_uid = path.parent.name
            sidecar = path.with_name(_nifti_stem(path) + ".json")
            metadata: dict[str, Any] = {}
            if sidecar.is_file():
                try:
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise InvalidInputError(f"invalid NIfTI sidecar {sidecar}: {exc}") from exc

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
            grouped[_clean_identifier(accession, "study")].append(
                Series(
                    series_uid=uid,
                    modality=description,
                    image=np.asarray(array),
                    affine=np.asarray(image.affine, dtype=np.float64),
                    source_path=path,
                    metadata=metadata,
                )
            )

        studies = tuple(
            Study(accession_number=accession, series=tuple(series))
            for accession, series in sorted(grouped.items())
        )
        return CompetitionDataset(root, studies, {"format": "nifti"})

    def _load_dicom(self, root: Path) -> CompetitionDataset:
        groups: dict[tuple[str, str], list[tuple[Path, Any]]] = defaultdict(list)
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            try:
                header = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    force=True,
                )
            except (OSError, pydicom.errors.InvalidDicomError):
                continue
            if not hasattr(header, "SeriesInstanceUID") or not hasattr(header, "Rows"):
                continue

            accession = _clean_identifier(
                getattr(header, "AccessionNumber", None),
                path.relative_to(root).parts[0],
            )
            series_uid = _clean_identifier(
                getattr(header, "SeriesInstanceUID", None),
                path.parent.name,
            )
            groups[(accession, series_uid)].append((path, header))

        if not groups:
            raise InvalidInputError(f"no readable NIfTI or DICOM images under {root}")

        by_study: dict[str, list[Series]] = defaultdict(list)
        for (accession, series_uid), entries in sorted(groups.items()):
            by_study[accession].append(self._read_dicom_series(series_uid, entries))

        studies = tuple(
            Study(accession_number=accession, series=tuple(series))
            for accession, series in sorted(by_study.items())
        )
        return CompetitionDataset(root, studies, {"format": "dicom"})

    def _read_dicom_series(
        self,
        series_uid: str,
        entries: list[tuple[Path, Any]],
    ) -> Series:
        first_header = entries[0][1]
        orientation = np.asarray(
            getattr(first_header, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]),
            dtype=np.float64,
        )
        row_direction = orientation[:3]
        column_direction = orientation[3:]
        slice_direction = np.cross(row_direction, column_direction)

        def order(entry: tuple[Path, Any]) -> float:
            header = entry[1]
            if hasattr(header, "ImagePositionPatient"):
                return float(
                    np.dot(
                        np.asarray(header.ImagePositionPatient, dtype=np.float64),
                        slice_direction,
                    )
                )
            return float(getattr(header, "InstanceNumber", 0))

        entries.sort(key=order)
        datasets = [pydicom.dcmread(str(path)) for path, _ in entries]
        slices = []
        for dataset in datasets:
            try:
                pixels = dataset.pixel_array.astype(np.float32)
            except Exception as exc:
                raise InvalidInputError(
                    f"cannot decode DICOM pixels for series {series_uid}: {exc}"
                ) from exc
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            slices.append(pixels * slope + intercept)

        shapes = {item.shape for item in slices}
        if len(shapes) != 1:
            raise InvalidInputError(f"inconsistent DICOM slice shapes in {series_uid}")
        image = np.stack(slices, axis=-1)

        spacing = np.asarray(
            getattr(datasets[0], "PixelSpacing", [1.0, 1.0]),
            dtype=np.float64,
        )
        origin = np.asarray(
            getattr(datasets[0], "ImagePositionPatient", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if len(datasets) > 1 and hasattr(datasets[1], "ImagePositionPatient"):
            slice_step = np.asarray(
                datasets[1].ImagePositionPatient,
                dtype=np.float64,
            ) - origin
        else:
            slice_step = slice_direction * float(
                getattr(datasets[0], "SpacingBetweenSlices", None)
                or getattr(datasets[0], "SliceThickness", 1.0)
            )

        affine_lps = np.eye(4, dtype=np.float64)
        affine_lps[:3, 0] = column_direction * spacing[0]
        affine_lps[:3, 1] = row_direction * spacing[1]
        affine_lps[:3, 2] = slice_step
        affine_lps[:3, 3] = origin
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        affine = lps_to_ras @ affine_lps

        description = str(
            getattr(datasets[0], "SeriesDescription", None)
            or getattr(datasets[0], "ProtocolName", None)
            or getattr(datasets[0], "Modality", "MR")
        )
        metadata = {
            "SeriesDescription": description,
            "Modality": str(getattr(datasets[0], "Modality", "")),
            "source_file_count": len(datasets),
        }
        return Series(
            series_uid=series_uid,
            modality=description,
            image=image,
            affine=affine,
            source_path=entries[0][0].parent,
            metadata=metadata,
        )
