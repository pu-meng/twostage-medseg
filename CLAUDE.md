# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **two-stage medical image segmentation** pipeline for liver and tumor segmentation (MSD Task03 Liver dataset). Stage 1 segments the liver from a full CT volume; Stage 2 segments tumors within the liver ROI crop. Both stages use DynUNet from MONAI.

**External dependency:** `medseg_project` (at `/home/pumengyu/medseg_project`) is a separate library that must be on `sys.path`. All scripts accept `--medseg_root` to add it dynamically. Key imports from it:
- `medseg.data.dataset_offline` — `load_pt_paths`, `split_three_ways`
- `medseg.data.transforms_offline` — `build_train_transforms`, `build_val_transforms`
- `medseg.engine.train_eval` — `train_one_epoch_sigmoid_binary`, `validate_sliding_window`
- `medseg.engine.adaptive_loss` — `train_one_epoch_binary_learnable`, `LearnableWeightedLoss`
- `medseg.models.build_model` — `build_model`
- `medseg.utils.ckpt` — `load_ckpt`, `load_ckpt_full`, `save_ckpt_full`
- `medseg.utils.io_utils`, `medseg.utils.train_utils`

**Data format:** Preprocessed `.pt` files in a flat directory (e.g. `/home/pumengyu/Task03_Liver_pt`). Each file is a dict `{"image": [1,D,H,W] float32, "label": [1,D,H,W] int64}` where label values are `0=background, 1=liver, 2=tumor`.

## Commands

### Train Stage 2 (tumor ROI model)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet \
  --model dynunet \
  --epochs 300 \
  --batch_size 1 \
  --lr 3e-3 \
  --patch 96 96 96 \
  --val_patch 96 96 96 \
  --amp \
  --loss dicefocal \
  --overlap 0.5 \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --init_ckpt /path/to/stage1_best.pt
```

### Evaluate two-stage pipeline
```bash
python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /path/to/stage1_best.pt \
  --stage2_ckpt /path/to/stage2_best.pt \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 \
  --stage2_patch 96 96 96 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --split test \
  --save_vis --vis_n 10 \
  --margin 24 \
  --save_dir /path/to/output
```

### Run inference only (no GT required)
```bash
python scripts/infer_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /path/to/stage1_best.pt \
  --stage2_ckpt /path/to/stage2_best.pt \
  --save_dir ./twostage_outputs
```

## Architecture

### Two-stage inference flow
1. **Stage 1** — sliding window inference on full volume → liver mask (binary, argmax over 2 classes)
2. Filter largest connected component (`metrics/filter.py`) to remove false positives
3. Compute liver bounding box with margin (`twostage/roi_utils.py:compute_bbox_from_mask`)
4. **Stage 2** — sliding window inference on the cropped liver ROI → tumor mask (binary)
5. Paste tumor prediction back into full volume coordinates (`roi_utils.py:paste_3d`)
6. Post-process: intersect tumor with liver mask, remove small components (< `min_tumor_size` voxels)
7. Combine into final label: `0=bg, 1=liver, 2=tumor`

### Training (Stage 2 specifics — `twostage/dataset_tumor_roi.py`)
- `TumorROIDataset` crops each case to liver bbox on-the-fly using GT liver mask
- **bbox_jitter**: randomly shifts bbox boundaries ±`bbox_max_shift` voxels to simulate Stage 1 prediction error
- **random_margin**: randomly samples margin from `[margin_min, margin_max]` to simulate ROI scale variation
- Label remapping: `label==2 → 1` (tumor is class 1 inside the ROI)
- `repeats` parameter multiplies dataset length (each case appears N times per epoch)

### Experiment outputs
Each training run creates a timestamped directory under `{exp_root}/{exp_name}/train/{MM-DD-HH-MM-SS}/` containing:
- `config.json` — full hyperparameter config
- `best.pt` / `last.pt` — model checkpoints (via `medseg.utils.ckpt`)
- `log.csv` / `log.txt` — per-epoch metrics (loss, tumor dice, best)
- `extra_log.csv` — learnable loss weights (if `--learnable_loss`)
- `diag.txt` / `diag_summary.txt` — data diagnostics (split stats, data leakage check, label stats)
- `metrics.json` / `report.txt` — final summary

### Key modules in this repo
- `twostage/dataset_tumor_roi.py` — Stage 2 dataset with online liver crop, bbox jitter, and random margin
- `twostage/roi_utils.py` — `compute_bbox_from_mask`, `crop_3d`, `paste_3d`, `bbox_to_dict`
- `twostage/train_logger.py` — CSV/TXT training logger (`TrainLoggerTwoStage`)
- `twostage/train_eval_tumor.py` — `tumor_metrics_from_val_result` helper
- `metrics/DiagLogger.py` — startup diagnostics: dataset stats, data leakage check, label distribution
- `metrics/filter.py` — `filter_largest_component` (removes noise from liver predictions)
- `scripts/train_tumor_roi.py` — main Stage 2 training script
- `scripts/eval_twostage.py` — full two-stage evaluation with metrics, CSV, optional PNG visualizations
- `scripts/infer_twostage.py` — inference-only script (saves `*_twostage.pt` per case)

### Loss options
`--loss` choices: `dicece` (default), `dicefocal`, `tversky`. With `--learnable_loss`, alpha weights between loss components are learned automatically via `LearnableWeightedLoss`.
