#!/bin/bash
# ============================================================
# 公共路径变量
# ============================================================
MEDSEG=/home/pumengyu/medseg_project
DATA=/home/pumengyu/Task03_Liver_pt
DATA_ROI=/home/pumengyu/Task03_Liver_roi        # 预裁剪肝脏 ROI，30GB，加载快
EXP=/home/pumengyu/experiments/twostage

STAGE1_CKPT=/home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt
ROIJITTER_CKPT=$EXP/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt

# --init_ckpt : 只加载权重，optimizer/scheduler/epoch 全部重置（接着上个实验跑用这个）
# --resume    : 完整恢复，用于中断后继续同一个 run

# ============================================================
# 实验五 🏃 — pred_bbox ROI 干净对比（训练中，GPU 0）
# 控制变量：只换 ROI 来源（pred_bbox），不加 hard mining
# repeats=6，~552 batches/epoch，预计 ~33h
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA \
  --exp_root $EXP \
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
  --init_ckpt $ROIJITTER_CKPT


# ============================================================
# 实验六 ⏳ — pred_bbox ROI + Hard Mining（GPU 空出立马启动，GPU 1）
# 与实验五并行跑，不等实验五完成，都用 ROIJITTER_CKPT 初始化
# repeats=6（与实验五基础一致），小肿瘤(<500) 额外 x4，无肿瘤额外 x2
# steps_per_epoch > 实验五（hard case 增加采样量），config.json 里会记录实际值
# 预计 ~50h
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA_ROI \
  --exp_root $EXP \
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
  --init_ckpt $ROIJITTER_CKPT


# ============================================================
# eval 模板（替换 EXPNAME 和 TIMESTAMP 后使用）
# 固定参数：--tta --min_tumor_size 100
# ============================================================

# STAGE2_CKPT=$EXP/EXPNAME/train/TIMESTAMP/best.pt

# CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
#   --medseg_root $MEDSEG \
#   --preprocessed_root $DATA \
#   --stage1_ckpt $STAGE1_CKPT \
#   --stage2_ckpt $STAGE2_CKPT \
#   --stage1_model dynunet --stage2_model dynunet \
#   --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
#   --stage2_sw_batch_size 2 \
#   --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
#   --overlap 0.5 \
#   --split test \
#   --tta \
#   --min_tumor_size 100
