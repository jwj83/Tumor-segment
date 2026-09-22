from __future__ import annotations

"""Generate a small BraTS-shaped NIfTI dataset for local pipeline smoke tests.

The files intentionally use modest dimensions so ten studies can be created in
seconds on a laptop.  Each study has T1, T1ce, T2 and FLAIR source series plus
an optional ``{accession}_seg.nii.gz`` label.  The label is ignored by the
competition loader (``seg`` is a reserved mask hint) and is useful later when
the real Goal 5 model is trained.

Example::

    python scripts/generate_brats_mock.py --output ./brats_mock --studies 10
"""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES = ("t1", "t1ce", "t2", "flair")
DESCRIPTIONS = {
    "t1": "T1",
    "t1ce": "T1ce",
    "t2": "T2",
    "flair": "FLAIR",
}


def generate_dataset(
    output: Path,
    *,
    studies: int = 10,
    shape: tuple[int, int, int] = (32, 32, 16),
    seed: int = 20260922,
    overwrite: bool = False,
) -> Path:
    if studies < 2:
        raise ValueError("at least two studies are required for duplicate_pairs.jsonl")
    if any(d < 4 for d in shape):
        raise ValueError(f"shape is too small: {shape}")
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is non-empty: {output}; pass --overwrite to replace it"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    affine = np.diag((1.0, 1.0, 1.5, 1.0)).astype(np.float64)
    manifest: dict[str, object] = {
        "format": "synthetic-brats-smoke-test",
        "seed": seed,
        "shape": list(shape),
        "studies": [],
    }
    for index in range(1, studies + 1):
        accession = f"BRATS_SYN_{index:03d}"
        study_dir = output / accession
        study_dir.mkdir()

        center = np.array(
            (
                rng.integers(shape[0] // 3, 2 * shape[0] // 3),
                rng.integers(shape[1] // 3, 2 * shape[1] // 3),
                rng.integers(shape[2] // 3, 2 * shape[2] // 3),
            ),
            dtype=np.float32,
        )
        radii = np.array(
            (
                rng.integers(max(2, shape[0] // 8), max(3, shape[0] // 5)),
                rng.integers(max(2, shape[1] // 8), max(3, shape[1] // 5)),
                rng.integers(max(1, shape[2] // 8), max(2, shape[2] // 4)),
            ),
            dtype=np.float32,
        )
        grid = np.indices(shape, dtype=np.float32)
        normalized = ((grid - center[:, None, None, None]) / radii[:, None, None, None]) ** 2
        lesion = np.sum(normalized, axis=0) <= 1.0
        core = lesion & (np.sum(normalized, axis=0) <= 0.42)

        series_manifest = []
        for modality in MODALITIES:
            uid = f"{accession}_{modality.upper()}"
            series_dir = study_dir / uid
            series_dir.mkdir()
            # Scanner-like low-amplitude background with modality-specific
            # lesion contrast.  Values are finite float32 and affine is stable.
            image = rng.normal(0.0, 0.08, size=shape).astype(np.float32)
            image += np.float32({"t1": 0.55, "t1ce": 0.60, "t2": 0.70, "flair": 0.65}[modality])
            contrast = {"t1": 0.35, "t1ce": 0.95, "t2": 1.10, "flair": 1.35}[modality]
            image[lesion] += np.float32(contrast)
            if modality == "t1ce":
                image[core] += np.float32(0.55)

            image_path = series_dir / f"{uid}.nii.gz"
            nib.save(nib.Nifti1Image(image, affine), str(image_path))
            sidecar = {
                "SeriesInstanceUID": uid,
                "SeriesDescription": DESCRIPTIONS[modality],
                "Modality": "MR",
                "Synthetic": True,
            }
            (series_dir / f"{uid}.json").write_text(
                json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            series_manifest.append({"series_uid": uid, "modality": modality})

        # Optional BraTS-style segmentation label.  The loader deliberately
        # excludes filenames containing ``seg`` from image input discovery.
        label = np.zeros(shape, dtype=np.uint8)
        label[lesion] = 2
        label[core] = 4
        nib.save(
            nib.Nifti1Image(label, affine),
            str(study_dir / f"{accession}_seg.nii.gz"),
        )
        goal4_target = {
            "location": (
                "RightFrontal",
                "LeftTemporal",
                "RightParietal",
                "LeftOccipital",
            )[(index - 1) % 4],
            "morphology": "Irregular" if index % 2 else "Regular",
            "who_grade": ((index - 1) % 4) + 1,
            "enhancement": bool(index % 2),
            "enhancement_pattern": "RimEnhancing" if index % 2 else "None",
            "necrosis": bool(index % 3 == 0),
            "cystic_change": bool(index % 4 == 0),
            "hemorrhage": bool(index % 5 == 0),
            "calcification": False,
            "margin_clear": bool(index % 2 == 0),
            "lobulation": bool(index % 2),
            "signal_t2wi": "High",
            "signal_flair": "High",
        }
        (study_dir / f"{accession}_goal4.json").write_text(
            json.dumps(goal4_target, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["studies"].append(
            {
                "accession": accession,
                "series": series_manifest,
                "goal4": goal4_target,
            }
        )

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--studies", type=int, default=10)
    parser.add_argument("--shape", type=int, nargs=3, default=(32, 32, 16), metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = generate_dataset(
        args.output,
        studies=args.studies,
        shape=tuple(args.shape),
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"generated {args.studies} synthetic studies under {path}")


if __name__ == "__main__":
    main()
