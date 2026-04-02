#!/bin/bash

# --init_ckpt : 只加载权重，optimizer/scheduler/epoch 全部重置（接着上个实验跑用这个）
# --resume    : 完整恢复，用于中断后继续同一个 run

# ============================================================
# Stage 1 重训 — 三分类（背景/肝脏/肿瘤）
# 原 Stage1 只输出肝脏（num_classes=2, merge_label12_to1），
# 新版输出肝脏+粗糙肿瘤（num_classes=3），用于缩小 Stage2 ROI
# ============================================================

CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --task liver \
  --exp_name dynunet_liver_tumor_stage1 \
  --model dynunet \
  --data_root /home/pumengyu/Task03_Liver \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --num_classes 3 \
  --epochs 200 \
  --batch_size 1 \
  --lr 0.003 \
  --patch 144 144 144 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 6 \
  --num_workers 2 \
  --cache_rate 0.0 \
  --early_ratios 0.2 0.8 \
  --late_ratios 0.1 0.9 \
  --amp \
  --loss dicece \
  --overlap 0.5 \
  --prefetch_factor 4 \
  --repeats 3


# ============================================================
# 实验十 — 粗糙肿瘤通道（Stage1三分类 + Stage2双通道CT+粗糙肿瘤）
# Stage1 ckpt 换成新训练的 dynunet_liver_tumor_stage1/best.pt
# pred_bbox_cache 需重新生成（新格式含 tumor bbox，换新路径）
# init_ckpt: 实验八最优 0.4451
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_coarse_tumor_2ch \
  --model dynunet \
  --epochs 300 \
  --batch_size 2 \
  --lr 3e-4 \
  --patch 128 128 128 \
  --val_patch 128 128 128 \
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
  --use_pred_bbox \
  --use_coarse_tumor \
  --stage1_out_channels 3 \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_tumor_stage1/train/TIMESTAMP/best.pt \
  --stage1_patch 144 144 144 \
  --stage1_model dynunet \
  --pred_bbox_cache /home/pumengyu/Task03_Liver_json/pred_bbox_stage1_3ch.json \
  --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox_roi_hardmine/train/03-28-09-59-18/best.pt

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
# 实验九A — pred_bbox ROI + repeats=12 + patch=128
# 全部等频，epoch 更长，等价于更多数据
# init_ckpt: 实验八最优 0.4451
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_predbbox_roi_rep12_p128 \
  --model dynunet \
  --epochs 300 \
  --batch_size 2 \
  --lr 3e-4 \
  --patch 128 128 128 \
  --val_patch 128 128 128 \
  --sw_batch_size 2 \
  --val_every 3 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --amp \
  --loss dicefocal \
  --val_overlap 0.5 \
  --repeats 12 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --bbox_jitter \
  --bbox_max_shift 8 \
  --random_margin \
  --margin_min 8 \
  --margin_max 20 \
  --use_pred_bbox \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage1_patch 144 144 144 \
  --stage1_model dynunet \
  --pred_bbox_cache /home/pumengyu/Task03_Liver_json/pred_bbox_stage1.json \
  --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox_roi_hardmine/train/03-28-09-59-18/best.pt


# ============================================================
# 实验九B — pred_bbox ROI + 大肿瘤×6 + patch=128
# 大肿瘤(>=50k voxel) ×6，其余等频，重点照顾大肿瘤
# init_ckpt: 实验八最优 0.4451
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_predbbox_roi_largetx6_p128 \
  --model dynunet \
  --epochs 300 \
  --batch_size 2 \
  --lr 3e-4 \
  --patch 128 128 128 \
  --val_patch 128 128 128 \
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
  --large_tumor_repeat_scale 6 \
  --use_pred_bbox \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage1_patch 144 144 144 \
  --stage1_model dynunet \
  --pred_bbox_cache /home/pumengyu/Task03_Liver_json/pred_bbox_stage1.json \
  --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox_roi_hardmine/train/03-28-09-59-18/best.pt


# ============================================================
# 实验九 — pred_bbox ROI + 两通道输入（CT + liver_mask）
# 与实验五唯一变量：--two_channel（in_channels 1→2）
# 训练时 Ch2 = GT liver mask，推理时 Ch2 = Stage1 预测 liver_mask
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_predbbox_roi_2ch \
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
  --two_channel \
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

