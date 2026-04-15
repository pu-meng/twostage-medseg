#!/bin/bash
# ============================================================
# 历史训练实验（已完成 / 已停止，仅供参考，不要重跑）
# ============================================================
MEDSEG=/home/PuMengYu/MSD_LiverTumorSeg/medseg_project
DATA=/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt
DATA_ROI=/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_roi
EXP=/home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage

STAGE1_CKPT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt
ROIJITTER_CKPT=$EXP/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt


# ============================================================
# 实验一 ✅ — tumor_dynunet_roi_jitter（当前最强基线，val 0.4697）
# GT bbox + bbox_jitter + random_margin，92 cases，repeats=6
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA \
  --exp_root $EXP \
  --exp_name tumor_dynunet_roi_jitter \
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
  --overlap 0.5 \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --bbox_jitter --bbox_max_shift 8 \
  --random_margin --margin_min 8 --margin_max 20 \
  --init_ckpt $STAGE1_CKPT


# ============================================================
# 实验二 ❌ — tumor_dynunet_predbbox（停止，IO 瓶颈 50min/epoch）
# Stage1 pred_bbox，在线裁剪，全量数据（130GB），太慢
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA \
  --exp_root $EXP \
  --exp_name tumor_dynunet_predbbox \
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
  --overlap 0.5 \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --use_pred_bbox \
  --stage1_ckpt $STAGE1_CKPT \
  --stage1_patch 144 144 144 \
  --random_margin --margin_min 8 --margin_max 24 \
  --small_tumor_thresh 500 \
  --small_tumor_repeat_scale 4 \
  --no_tumor_repeat_scale 2 \
  --init_ckpt $ROIJITTER_CKPT


# ============================================================
# 实验三 ❌ — tumor_dynunet_predbbox_roi（停止，repeats=6 太慢，345h）
# 换用预裁剪 ROI（30GB），速度解决了，但 repeats=6 还是太慢
# 同时混了 pred_bbox + hard mining 两个变量，无法做消融
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA_ROI \
  --exp_root $EXP \
  --exp_name tumor_dynunet_predbbox_roi \
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
  --overlap 0.5 \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --random_margin --margin_min 8 --margin_max 24 \
  --small_tumor_thresh 500 \
  --small_tumor_repeat_scale 4 \
  --no_tumor_repeat_scale 2 \
  --init_ckpt $ROIJITTER_CKPT


# ============================================================
# baseline: dynunet_singlestage（单阶段对比，3 类直接预测）
# 展示两阶段相对单阶段的提升，跑一次留作论文参考
# ============================================================

CUDA_VISIBLE_DEVICES=1 python -m scripts.train \
  --task liver \
  --exp_name dynunet_singlestage \
  --model dynunet \
  --preprocessed_root $DATA \
  --num_classes 3 \
  --epochs 300 \
  --batch_size 1 \
  --lr 3e-3 \
  --patch 144 144 144 \
  --val_patch 144 144 144 \
  --sw_batch_size 1 \
  --val_every 5 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --early_ratios 0.1 0.9 \
  --late_ratios 0.1 0.9 \
  --amp \
  --loss dicefocal \
  --overlap 0.5 \
  --repeats 3
# note: 无 --merge_label12_to1，无 --init_ckpt，从头训练
# exp_root 默认 /home/PuMengYu/MSD_LiverTumorSeg/experiments（不是 twostage 子目录）
