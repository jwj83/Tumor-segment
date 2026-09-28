#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/2026aicompetition/datasets/training/annotation}"
ANN_ROOT="${ANN_ROOT:-$DATA_ROOT}"
OUT_ROOT="${OUT_ROOT:-./runs}"
mkdir -p "$OUT_ROOT"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-40}"

RUN_NNUNET="${RUN_NNUNET:-0}"
if [[ "$RUN_NNUNET" == "1" ]]; then
  : "${NNUNET_RAW:?Set NNUNET_RAW before RUN_NNUNET=1}"
  : "${NNUNET_DATASET_ABNORMAL:?Set NNUNET_DATASET_ABNORMAL, e.g. 501}"
  : "${NNUNET_DATASET_CORE:?Set NNUNET_DATASET_CORE, e.g. 502}"
  ABNORMAL_OUT="$NNUNET_RAW/Dataset${NNUNET_DATASET_ABNORMAL}_GliomaAbnormal"
  CORE_OUT="$NNUNET_RAW/Dataset${NNUNET_DATASET_CORE}_GliomaCore"
else
  ABNORMAL_OUT="$OUT_ROOT/nnunet_abnormal"
  CORE_OUT="$OUT_ROOT/nnunet_core"
fi

python scripts/read_training_annotations.py \
  --series-type "$ANN_ROOT/SeriesType.xlsx" \
  --annotation "$ANN_ROOT/脑胶质瘤标注结果-训练集.xlsx" \
  --out-dir "$OUT_ROOT/readout"
python scripts/scan_training_files.py \
  --index "$OUT_ROOT/readout/sample_index.csv" \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_ROOT/file_check"
python scripts/summarize_study_modalities.py \
  --index "$OUT_ROOT/file_check/file_index.csv" \
  --out "$OUT_ROOT/study_modalities.json"

python -m training.train_medicalnet \
  --file-index "$OUT_ROOT/file_check/file_index.csv" \
  --labels-csv "$OUT_ROOT/readout/series_merged.csv" \
  --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" \
  --output "$OUT_ROOT/goal3_medicalnet"

python -m training.train_goal4 \
  --file-index "$OUT_ROOT/file_check/file_index.csv" \
  --labels-csv "$OUT_ROOT/readout/series_merged.csv" \
  --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" \
  --output "$OUT_ROOT/goal4_medicalnet"

python scripts/export_nnunet_dataset.py \
  --index "$OUT_ROOT/file_check/file_index.csv" \
  --labels-csv "$OUT_ROOT/readout/series_merged.csv" \
  --out-dir "$ABNORMAL_OUT" --target abnormal --link
python scripts/export_nnunet_dataset.py \
  --index "$OUT_ROOT/file_check/file_index.csv" \
  --labels-csv "$OUT_ROOT/readout/series_merged.csv" \
  --out-dir "$CORE_OUT" --target core --link

if [[ "$RUN_NNUNET" == "1" ]]; then
  nnUNetv2_plan_and_preprocess -d "$NNUNET_DATASET_ABNORMAL" --verify_dataset_integrity
  nnUNetv2_train "$NNUNET_DATASET_ABNORMAL" 3d_fullres 0 --npz
  nnUNetv2_plan_and_preprocess -d "$NNUNET_DATASET_CORE" --verify_dataset_integrity
  nnUNetv2_train "$NNUNET_DATASET_CORE" 3d_fullres 0 --npz
fi
