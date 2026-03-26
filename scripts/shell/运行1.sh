#!/bin/bash
cd /home/pumengyu/twostage_medseg

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
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
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage1_patch 144 144 144 \
  --random_margin \
  --margin_min 8 \
  --margin_max 24 \
  --small_tumor_thresh 500 \
  --small_tumor_repeat_scale 4 \
  --no_tumor_repeat_scale 2 \
  --pred_bbox_cache /home/pumengyu/Task03_Liver_json/pred_bbox_stage1.json \
  --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox/train/03-26-16-38-04/last.pt
