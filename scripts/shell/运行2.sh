#!/bin/bash

# ============================================================
# 实验十一 — dynunet_focaltversky_smallmine_p128
# 全力小肿瘤攻坚：focaltversky + small tumor hard mining + 优化配置
#
# 与实验八(largetx6_p128)核心差异：
#   - loss: dicefocal → focaltversky (beta=0.7加大FN惩罚, gamma=0.75)
#   - small_tumor_thresh=5000, small_tumor_repeat_scale=4 (极小肿瘤×4)
#   - no_tumor_repeat_scale=2 (无肿瘤×2)
#   - tumor_ratios=0.02 0.98 (98%从肿瘤体素附近切patch)
#   - repeats=8 (从6提升)
#   - val_overlap=0.25 (修复之前0.0导致best.pt选取失真)
#   - batch_size=2
# init_ckpt: tumor_dynunet_predbbox_roi_largetx6_p128/train/03-30-23-47-04/last.pt
# ============================================================

# ============================================================
# 实验十一 finetune — dynunet_focaltversky_smallmine_p128_finetune
# 在实验十一 best.pt (epoch78, score=0.6481) 基础上小学习率继续训练
# 改动: lr 3e-4 → 1e-4, epochs 300 → 100, init_ckpt → 实验十一 best.pt
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name dynunet_focaltversky_smallmine_p128_finetune \
  --model dynunet \
  --epochs 100 \
  --batch_size 3 \
  --lr 1e-4 \
  --patch 128 128 128 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 3 \
  --num_workers 10 \
  --prefetch_factor 4 \
  --amp \
  --loss focaltversky \
  --val_overlap 0.25 \
  --repeats 8 \
  --tumor_ratios 0.02 0.98 \
  --margin 8 \
  --bbox_jitter \
  --bbox_max_shift 8 \
  --random_margin \
  --margin_min 8 \
  --margin_max 20 \
  --use_pred_bbox \
  --small_tumor_thresh 5000 \
  --small_tumor_repeat_scale 4 \
  --no_tumor_repeat_scale 2 \
  --large_tumor_thresh 50000 \
  --large_tumor_repeat_scale 3 \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage1_patch 144 144 144 \
  --stage1_model dynunet \
  --pred_bbox_cache /home/pumengyu/Task03_Liver_json/pred_bbox_stage1.json \
  --init_ckpt /home/pumengyu/experiments/twostage/dynunet_focaltversky_smallmine_p128/train/04-06-12-56-02/best.pt \
  --seed 0

# ============================================================
# 旧：实验十一原始训练（已中断，98epoch, best=0.6481）
# ============================================================
# CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
#   --medseg_root /home/pumengyu/medseg_project \
#   --preprocessed_root /home/pumengyu/Task03_Liver_pt \
#   --exp_root /home/pumengyu/experiments/twostage \
#   --exp_name dynunet_focaltversky_smallmine_p128 \
#   --model dynunet \
#   --epochs 300 \
#   --batch_size 3 \
#   --lr 3e-4 \
#   --patch 128 128 128 \
#   --val_patch 96 96 96 \
#   --sw_batch_size 1 \
#   --val_every 3 \
#   --num_workers 10 \
#   --prefetch_factor 4 \
#   --amp \
#   --loss focaltversky \
#   --val_overlap 0.25 \
#   --repeats 8 \
#   --tumor_ratios 0.02 0.98 \
#   --margin 8 \
#   --bbox_jitter \
#   --bbox_max_shift 8 \
#   --random_margin \
#   --margin_min 8 \
#   --margin_max 20 \
#   --use_pred_bbox \
#   --small_tumor_thresh 5000 \
#   --small_tumor_repeat_scale 4 \
#   --no_tumor_repeat_scale 2 \
#   --large_tumor_thresh 50000 \
#   --large_tumor_repeat_scale 3 \
#   --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
#   --stage1_patch 144 144 144 \
#   --stage1_model dynunet \
#   --pred_bbox_cache /home/pumengyu/Task03_Liver_json/pred_bbox_stage1.json \
#   --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox_roi_largetx6_p128/train/03-30-23-47-04/last.pt \
#   --seed 0


# ============================================================
# 旧：dynunet_dicece_nnunet_p128 中途验证集 eval
# stage2 best: 04-05-01-14-36 (训练到110epoch, best=0.5749, 不理想，已放弃)
# ============================================================
#
# 实验:dynunet_dicece_nnunet_p128 中途验证集 eval
# stage1: dynunet_liver_only (两分类, 03-14-01-11-56)
# stage2: dynunet_dicece_nnunet_p128 best.pt (04-05-01-14-36)
# ============================================================

# CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
#   --medseg_root /home/pumengyu/medseg_project \
#   --preprocessed_root /home/pumengyu/Task03_Liver_pt \
#   --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
#   --stage2_ckpt /home/pumengyu/experiments/twostage/dynunet_dicece_nnunet_p128/train/04-05-01-14-36/best.pt \
#   --stage1_model dynunet --stage2_model dynunet \
#   --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
#   --stage2_sw_batch_size 1 \
#   --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
#   --overlap 0.5 --split test \
#   --min_tumor_size 100

# ============================================================
# 实验十一 eval — dynunet_focaltversky_smallmine_p128
# stage2 best: 04-06-12-56-02 (epoch5 中途, best tumor dice=0.1484)
# 数据划分: split_two_with_monitor (与训练一致, test=nnUNet fold0 19个case)
# ============================================================

# ============================================================
# 实验十一 eval — 自适应连通域阈值测试
# 新增 --comp_prob_thresh 0.4：体积小但高置信度的连通域也保留
# ============================================================

# CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
#   --medseg_root /home/pumengyu/medseg_project \
#   --preprocessed_root /home/pumengyu/Task03_Liver_pt \
#   --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
#   --stage2_ckpt /home/pumengyu/experiments/twostage/dynunet_focaltversky_smallmine_p128/train/04-06-12-56-02/best.pt \
#   --stage1_model dynunet --stage2_model dynunet \
#   --stage1_patch 144 144 144 --stage2_patch 128 128 128 \
#   --stage2_sw_batch_size 1 \
#   --overlap 0.25 --margin 8 \
#   --small_tumor_low_thresh 0.2 \
#   --comp_prob_thresh 0.4 \
#   --split test

CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage2_ckpt /home/pumengyu/experiments/twostage/dynunet_focaltversky_smallmine_p128/train/04-06-12-56-02/best.pt \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 128 128 128 \
  --stage2_sw_batch_size 1 \
  --overlap 0.25 --margin 8 \
  --small_tumor_low_thresh 0.2 \
  --split test

============================================================
实验十二 — dynunet_focaltversky_smallmine_zoom_p128
在实验十一基础上增加 small_tumor_zoom_in：
  小肿瘤(< 5000 vox)训练时ROI放大2倍，增强空间特征学习
init_ckpt: 实验十一 best.pt
============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name dynunet_focaltversky_smallmine_zoom_p128 \
  --model dynunet \
  --epochs 150 \
  --batch_size 3 \
  --lr 1e-4 \
  --patch 128 128 128 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 3 \
  --num_workers 10 \
  --prefetch_factor 4 \
  --amp \
  --loss focaltversky \
  --val_overlap 0.25 \
  --repeats 8 \
  --tumor_ratios 0.02 0.98 \
  --margin 8 \
  --bbox_jitter \
  --bbox_max_shift 8 \
  --random_margin \
  --margin_min 8 \
  --margin_max 20 \
  --use_pred_bbox \
  --small_tumor_thresh 5000 \
  --small_tumor_repeat_scale 4 \
  --no_tumor_repeat_scale 2 \
  --large_tumor_thresh 50000 \
  --large_tumor_repeat_scale 3 \
  --small_tumor_zoom_thresh 5000 \
  --small_tumor_zoom_factor 2.0 \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage1_patch 144 144 144 \
  --stage1_model dynunet \
  --pred_bbox_cache /home/pumengyu/Task03_Liver_json/pred_bbox_stage1.json \
  --init_ckpt /home/pumengyu/experiments/twostage/dynunet_focaltversky_smallmine_p128/train/04-06-12-56-02/best.pt \
  --seed 42
