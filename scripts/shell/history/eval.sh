#!/bin/bash
# ============================================================
# 历史 eval 命令（已完成，仅供参考）
# 当前 eval 模板在 运行.sh 末尾
# ============================================================
MEDSEG=/home/PuMengYu/MSD_LiverTumorSeg/medseg_project
DATA=/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt
EXP=/home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage

STAGE1_CKPT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt
ROIJITTER_CKPT=$EXP/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt


# ============================================================
# eval: 实验一 无 TTA（baseline，tumor Dice 0.5816）
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/eval_twostage.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA \
  --stage1_ckpt $STAGE1_CKPT \
  --stage2_ckpt $ROIJITTER_CKPT \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 \
  --split test \
  --margin 12 \
  --min_tumor_size 100


# ============================================================
# eval: 实验一 + TTA（当前最优，tumor Dice 0.5965）
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/eval_twostage.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA \
  --stage1_ckpt $STAGE1_CKPT \
  --stage2_ckpt $ROIJITTER_CKPT \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 \
  --split test \
  --tta \
  --margin 12 \
  --min_tumor_size 100 \
  --save_dir $EXP/tumor_dynunet_roi_jitter/eval_tta


# ============================================================
# eval: ensemble（实验一 + 另一个模型，权重各 0.5）
# TIMESTAMP 替换为实际目录名后使用
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA \
  --stage1_ckpt $STAGE1_CKPT \
  --stage2_ckpt $EXP/EXPNAME/train/TIMESTAMP/best.pt \
  --stage2_ckpt_b $ROIJITTER_CKPT \
  --ensemble_weight_b 0.5 \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 \
  --split test \
  --tta \
  --min_tumor_size 100
