#!/bin/bash
# ============================================================
# 实验十 — DynUNet + 2D/3D 特征融合 + Skip Attention Gate
#
# 对应论文：Lu Meng et al., "Two-Stage Liver and Tumor Segmentation
#           Algorithm Based on Convolutional Neural Network", Diagnostics 2021
#
# 两个核心改动（均在 dynunet_ca.py 实现）：
#   1. SliceWise2DBranch（L1 32ch skip）：
#      Conv3d(1,3,3) 模拟 slice-wise 2D 特征，与 3D 特征 concat 融合
#      对应论文 Figure 5A 的 2D U-Net 分支
#   2. AttGate3D（全部 4 层 skip，32/64/128/256ch）：
#      (1 + sigmoid(θ_x + φ_g)) × x  —— 残差注意力门控
#      对应论文 Figure 6 + Eq.3，g = 来自更深层 decoder 的 gating signal
#
# 参数增量：+0.127M（base 16.54M → 16.67M）
# init_ckpt：实验八 last.pt（test Dice 0.6212，当前最优）
#            load_init_weights 自动加 backbone. 前缀匹配，118/139 参数加载
#            新增的 21 个参数（att_gate + slice2d）随机初始化
#
# 其余超参与实验八完全一致，确保对比公平：
#   patch=128, repeats=6, large_tumor_thresh=50000 ×3
#   lr=3e-4, loss=dicefocal, epochs=300
# ============================================================
# scripts/shell/exp10_dynunet_ca.sh

CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
  --medseg_root /home/PuMengYu/MSD_LiverTumorSeg/medseg_project \
  --preprocessed_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt \
  --exp_root /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage \
  --exp_name tumor_dynunet_ca_predbbox_roi_largetx6_p128 \
  --model dynunet_ca \
  --epochs 300 \
  --batch_size 1 \
  --lr 3e-4 \
  --patch 128 128 128 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 3 \
  --num_workers 2 \
  --prefetch_factor 2 \
  --amp \
  --loss dicefocal \
  --val_overlap 0.0 \
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
  --stage1_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage1_patch 144 144 144 \
  --stage1_model dynunet \
  --pred_bbox_cache /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_json/pred_bbox_stage1.json \
  --init_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/tumor_dynunet_predbbox_roi_largetx6_p128/train/03-30-23-47-04/last.pt \
  --seed 0


# ============================================================
# eval 模板（训练完后替换 TIMESTAMP 使用）
# ============================================================

# CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
#   --medseg_root /home/PuMengYu/MSD_LiverTumorSeg/medseg_project \
#   --preprocessed_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt \
#   --stage1_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
#   --stage2_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/tumor_dynunet_ca_predbbox_roi_largetx6_p128/train/TIMESTAMP/best.pt \
#   --stage1_model dynunet --stage2_model dynunet_ca \
#   --stage1_patch 144 144 144 --stage2_patch 128 128 128 \
#   --stage2_sw_batch_size 2 \
#   --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
#   --overlap 0.5 \
#   --split test \
#   --tta \
#   --min_tumor_size 100
