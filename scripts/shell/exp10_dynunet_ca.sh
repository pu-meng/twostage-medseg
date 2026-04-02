#!/bin/bash
# ============================================================
# 实验十 — DynUNet + CBAM3D 注意力（dynunet_ca）
# 唯一变量：--model dynunet_ca（vs 实验八的 dynunet）
# 其余参数与实验八完全一致，确保对比公平
#
# CBAM3D 位置：skip_layers(decoder 32ch 特征) → output_block 之间
# 参数增量：+1198（+0.007%）
# 兼容：AMP / torch.compile / sliding_window_inference
#
# init_ckpt：实验八 last.pt（0.6212 test Dice，当前最优）
# 注意：init_ckpt 用裸 DynUNet ckpt 热启动 DynUNetWithCA 时，
#       train_tumor_roi.py 的 load_init_weights 会通过 shape 匹配
#       自动加载 backbone 权重（key 前缀 backbone. 不同会被跳过）。
#       因此只有完整加载效果。如需精确热启动请用 model.load_backbone_weights()。
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_ca_predbbox_roi_largetx6_p128 \
  --model dynunet_ca \
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
  --large_tumor_thresh 50000 \
  --large_tumor_repeat_scale 3 \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage1_patch 144 144 144 \
  --stage1_model dynunet \
  --pred_bbox_cache /home/pumengyu/Task03_Liver_json/pred_bbox_stage1.json \
  --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox_roi_largetx6_p128/train/03-30-23-47-04/last.pt \
  --seed 0
