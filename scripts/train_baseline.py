from __future__ import annotations

"""Train the first Goal 3/4/5 PyTorch baseline on NIfTI studies.

This script is intentionally small: it is a starting point for tomorrow's
training run, not a claimed competition recipe.  The produced checkpoints are
loaded by ``tasks.torch_pipeline`` without changing the inference contract.
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.loader import DatasetLoader
from data.structures import Study
from tasks.goal3.model import Goal3Model
from tasks.goal4.model import Goal4Model
from tasks.goal5.model import Goal5Model
from tasks.torch_tasks import (
    INPUT_MODALITIES,
    LOCATION_LABELS,
    MORPHOLOGY_LABELS,
    PATTERN_LABELS,
    SIGNAL_LABELS,
    WHO_LABELS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("goal3", "goal4", "goal5", "all"), default="all")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()
    _seed_everything(args.seed)
    device = torch.device(args.device)
    studies = tuple(DatasetLoader().iter_studies(args.dataset))
    if not studies:
        raise ValueError("dataset contains no studies")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_path or args.output_dir / "training.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = ("goal3", "goal4", "goal5") if args.task == "all" else (args.task,)
    for task in tasks:
        path = _train(task, studies, args.dataset.resolve(), args.output_dir, args.epochs, args.lr, device, log_path)
        print(f"saved {task} checkpoint: {path}")


def _train(task: str, studies: tuple[Study, ...], root: Path, output: Path, epochs: int, lr: float, device: torch.device, log_path: Path) -> Path:
    if task == "goal3":
        model = Goal3Model().to(device)
    elif task == "goal4":
        model = Goal4Model().to(device)
    else:
        model = Goal5Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for study in studies:
            x = _study_tensor(study, device)
            optimizer.zero_grad(set_to_none=True)
            if task == "goal3":
                label = _segmentation_label(root, study)
                target = torch.tensor([float(np.any(label > 0))], device=device)
                loss = nn.functional.binary_cross_entropy_with_logits(model(x), target)
            elif task == "goal4":
                outputs = model(x)
                target = _goal4_target(root, study)
                loss = _goal4_loss(outputs, target, device)
            else:
                labels = _segmentation_label(root, study)
                target = torch.from_numpy(np.stack((labels == 4, labels > 0)).astype(np.float32))[None].to(device)
                target = F.interpolate(target, size=x.shape[2:], mode="nearest")
                loss = nn.functional.binary_cross_entropy_with_logits(model(x), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"{task} epoch={epoch} loss={sum(losses) / len(losses):.5f}")
        _write_log(
            log_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "step": epoch * len(studies),
                "phase": "train",
                "loss": sum(losses) / len(losses),
                "lr": lr,
                "mode": "training",
                "data_source": str(root),
                "checkpoint": str(output / f"{task}.pt"),
                "task": task,
            },
        )
    path = output / f"{task}.pt"
    torch.save({"model": model.state_dict(), "task": task, "input_shape": list(x.shape[2:])}, path)
    return path


def _goal4_loss(outputs: dict[str, torch.Tensor], target: dict[str, object], device: torch.device) -> torch.Tensor:
    loss = torch.zeros((), device=device)
    categorical = {
        "location": LOCATION_LABELS,
        "morphology": MORPHOLOGY_LABELS,
        "who_grade": WHO_LABELS,
        "enhancement_pattern": PATTERN_LABELS,
        "signal_t2wi": SIGNAL_LABELS,
        "signal_flair": SIGNAL_LABELS,
    }
    for name, labels in categorical.items():
        index = labels.index(str(target[name]))
        loss = loss + F.cross_entropy(outputs[name], torch.tensor([index], device=device))
    for name in ("enhancement", "necrosis", "cystic_change", "hemorrhage", "calcification", "margin_clear", "lobulation"):
        value = torch.tensor([[float(bool(target[name]))]], device=device)
        loss = loss + F.binary_cross_entropy_with_logits(outputs[name], value)
    return loss


def _study_tensor(study: Study, device: torch.device) -> torch.Tensor:
    arrays = []
    for modality in INPUT_MODALITIES:
        series = _select_series(study, modality)
        values = np.asarray(series.image, dtype=np.float32)
        finite = np.isfinite(values)
        values = np.nan_to_num(values, copy=True)
        if finite.any():
            lo, hi = np.percentile(values[finite], (1.0, 99.0))
            values = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        else:
            values.fill(0.0)
        arrays.append(np.asarray(values, dtype=np.float32))
    tensor = torch.from_numpy(np.stack(arrays).astype(np.float32, copy=False)).unsqueeze(0).to(device)
    return F.interpolate(tensor, size=(32, 32, 16), mode="trilinear", align_corners=False)


def _segmentation_label(root: Path, study: Study) -> np.ndarray:
    path = root / study.accession_number / f"{study.accession_number}_seg.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"missing training label: {path}")
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.uint8)


def _goal4_target(root: Path, study: Study) -> dict[str, object]:
    path = root / study.accession_number / f"{study.accession_number}_goal4.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Goal4 training label: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _select_series(study: Study, modality: str):
    for series in study.series:
        text = " ".join((series.modality or "", series.series_uid)).lower()
        if modality in text:
            return series
    return study.series[0]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_log(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
