#!/bin/bash

# --init_ckpt : 只加载权重，optimizer/scheduler/epoch 全部重置（接着上个实验跑用这个）
# --resume    : 完整恢复，用于中断后继续同一个 run

# ============================================================
# 实验五 — pred_bbox ROI 干净对比
# 控制变量：只换 ROI 来源（pred_bbox），不加 hard mining
# repeats=6，~552 batches/epoch
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_predbbox_roi_clean \
  --model dynunet \
  --epochs 300 \
  --batch_size 2 \
  --lr 3e-3 \
  --patch 96 96 96 \
  --val_patch 96 96 96 \
  --sw_batch_size 2 \
  --val_every 3 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --amp \
  --loss dicefocal \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --random_margin --margin_min 8 --margin_max 24 \
  --val_overlap 0.5 \
  --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt


# ============================================================
# 实验六 — pred_bbox ROI + Hard Mining
# 与实验五并行跑，都用 ROIJITTER_CKPT 初始化
# repeats=6，小肿瘤(<500) 额外 x4，无肿瘤额外 x2
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_predbbox_roi_hardmine \
  --model dynunet \
  --epochs 300 \
  --batch_size 2 \
  --lr 3e-3 \
  --patch 96 96 96 \
  --val_patch 96 96 96 \
  --sw_batch_size 2 \
  --val_every 3 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --amp \
  --loss dicefocal \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --random_margin --margin_min 8 --margin_max 24 \
  --small_tumor_thresh 500 \
  --small_tumor_repeat_scale 4 \
  --no_tumor_repeat_scale 2 \
  --val_overlap 0.5 \
  --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt


# ============================================================
# 实验八 — pred_bbox ROI + 大肿瘤过采样（实验六改进版）
# 与实验五唯一变量：large_tumor_thresh=50000，large_tumor_repeat_scale=3
# 大肿瘤(>=50k voxel) ×3，其余 ×1，总 indices≈948，预计 ~41h
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_predbbox_roi_hardmine \
  --model dynunet \
  --epochs 300 \
  --batch_size 2 \
  --lr 3e-4 \
  --patch 96 96 96 \
  --val_patch 96 96 96 \
  --sw_batch_size 2 \
  --val_every 3 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --amp \
  --loss dicefocal \
  --val_overlap 0.5 \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --bbox_jitter \
  --bbox_max_shift 8 \
  --random_margin \
  --margin_min 8 \
  --margin_max 20 \
  --large_tumor_thresh 50000 \
  --large_tumor_repeat_scale 3 \
  --use_pred_bbox \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage1_patch 144 144 144 \
  --stage1_model dynunet \
  --pred_bbox_cache /home/pumengyu/Task03_Liver_json/pred_bbox_stage1.json \
  --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt


# ============================================================
# eval 模板（替换 EXPNAME 和 TIMESTAMP 后使用）
# 固定参数：--tta --min_tumor_size 100
# ============================================================

# 验证pred_bbox ROI+hardmining的效果,和纯pred_bbox ROI对比

CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage2_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox_roi_hardmine/train/03-28-09-59-18/best.pt \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 \
  --split test \
  --tta \
  --min_tumor_size 100

验证纯pred_bbox ROI的效果

CUDA_VISIBLE_DEVICES=1 python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage2_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox_roi_clean/train/03-28-09-49-40/best.pt \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 \
  --split test \
  --tta \
  --min_tumor_size 100

