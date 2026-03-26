#!/bin/bash
# ============================================================
# common path variables
# ============================================================
MEDSEG=/home/pumengyu/medseg_project
DATA=/home/pumengyu/Task03_Liver_pt
EXP=/home/pumengyu/experiments/twostage

STAGE1_CKPT=/home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt
ROIJITTER_CKPT=$EXP/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt

# ============================================================
# training workflow
# ============================================================
# do not re-run 300 epochs from scratch every time you change code!
# use --init_ckpt to warm-start from the previous best.pt.
#
# --init_ckpt : loads model weights only (shape-matched layers),
#               optimizer / scheduler / epoch are all reset
# --resume    : full checkpoint resume, use when continuing the same run
#
# chain:
#   first run   --init_ckpt $STAGE1_CKPT   (transfer from liver model)
#   later runs  --init_ckpt previous best.pt


# ============================================================
# exp1: tumor_dynunet_roi_jitter (做目前为止最好, val dice 0.4697)
# GT bbox + bbox_jitter + random_margin, full training set
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
# exp2: tumor_dynunet_predbbox (next experiment)
# 使用 Stage1 预测的 bbox 代替 GT bbox -> no domain gap
# small tumor (<500 voxels) oversampled x4 to reduce missed detections
# warm-start from roi_jitter best.pt
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
# eval: single model
# replace TIMESTAMP with the actual run directory name
# save_dir is auto-derived from stage2_ckpt, no need to set manually
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA \
  --stage1_ckpt $STAGE1_CKPT \
  --stage2_ckpt $EXP/tumor_dynunet_predbbox/train/TIMESTAMP/best.pt \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 \
  --split test \
  --save_vis --vis_n 10 \
  --margin 12


# ============================================================
# eval: 集成两个模型 (two models combined)
# stage2_ckpt = new model, stage2_ckpt_b = roi_jitter as second model
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  --medseg_root $MEDSEG \
  --preprocessed_root $DATA \
  --stage1_ckpt $STAGE1_CKPT \
  --stage2_ckpt $EXP/tumor_dynunet_predbbox/train/TIMESTAMP/best.pt \
  --stage2_ckpt_b $ROIJITTER_CKPT \
  --ensemble_weight_b 0.5 \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 \
  --split test \
  --save_vis --vis_n 10 \
  --margin 12


# ============================================================
# historical experiments (reference only, do not run)
# ============================================================

# tumor_dynunet_balanced (val dice ~0.49, did not beat roi_jitter)
# reason: lr=8e-4 restart caused oscillation, later lr=1e-4 also failed to improve

# tumor_dynunet_augv2 (val dice 0.3157, stopped)
# reason: no_tumor_repeat_scale=4 too aggressive, model became overly conservative

# ============================================================
# baseline: dynunet_singlestage
# 全图直接预测 bg/liver/tumor（3类），不裁 ROI，不分两阶段
# 对比用：证明 two-stage 比 single-stage 好多少
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

# 注意：不加 --merge_label12_to1，不加 --init_ckpt，从头训练
# exp_root 默认是 /home/pumengyu/experiments（不是 twostage 子目录）


目前最好的指标参数

/home/pumengyu/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt


# ============================================================
# eval: TTA 对比实验（同一个 checkpoint，唯一区别是有无 --tta）
# 先跑无TTA baseline，再跑TTA，对比纯收益
# ============================================================

# step1: 无TTA baseline


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
  --margin 12

# step2: 加TTA（save_dir 单独指定，避免覆盖 step1 结果）
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
  --tta \
  --save_dir $EXP/tumor_dynunet_roi_jitter/eval_tta