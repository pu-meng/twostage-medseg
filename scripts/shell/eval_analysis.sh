#!/bin/bash
# 对两个 ckpt 分别跑 test + val，保存可视化和 pred_pt
# 运行方式: bash scripts/shell/eval_analysis.sh

STAGE1_CKPT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt
MEDSEG=/home/PuMengYu/MSD_LiverTumorSeg/medseg_project
DATA=/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt

# ---------- ckpt A: 03-29-13-33-35/best.pt (原始最优) ----------
CKPT_A=/home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/tumor_dynunet_predbbox_roi_largetx6_p128/train/03-29-13-33-35/best.pt
SAVE_A=/home/PuMengYu/MSD_LiverTumorSeg/experiments/analysis/ckpt_A

# ---------- ckpt B: 03-30-23-47-04/last.pt (resume后当前last) ----------
CKPT_B=/home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/tumor_dynunet_predbbox_roi_largetx6_p128/train/03-30-23-47-04/last.pt
SAVE_B=/home/PuMengYu/MSD_LiverTumorSeg/experiments/analysis/ckpt_B

COMMON="
  --medseg_root $MEDSEG
  --preprocessed_root $DATA
  --stage1_ckpt $STAGE1_CKPT
  --stage1_model dynunet
  --stage2_model dynunet
  --stage1_patch 144 144 144
  --stage2_patch 128 128 128
  --stage2_sw_batch_size 2
  --val_ratio 0.2 --test_ratio 0.1 --seed 0
  --overlap 0.5
  --min_tumor_size 100
  --save_vis
  --save_pred_pt
  --vis_n 999
"

echo "===== ckpt_A test ====="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  $COMMON \
  --stage2_ckpt $CKPT_A \
  --split test \
  --save_dir $SAVE_A/test

echo "===== ckpt_A val ====="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  $COMMON \
  --stage2_ckpt $CKPT_A \
  --split val \
  --save_dir $SAVE_A/val

echo "===== ckpt_B test ====="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  $COMMON \
  --stage2_ckpt $CKPT_B \
  --split test \
  --save_dir $SAVE_B/test

echo "===== ckpt_B val ====="
CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  $COMMON \
  --stage2_ckpt $CKPT_B \
  --split val \
  --save_dir $SAVE_B/val

echo "===== Done ====="
