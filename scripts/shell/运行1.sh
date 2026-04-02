!/bin/bash

============================================================
实验十完整流程：Stage1重训（三分类）→ Stage2训练 → eval
说明：
  Step1 在 /home/pumengyu/medseg_project 目录下运行
  Step2/Step3 在 /home/pumengyu/twostage_medseg 目录下运行
  Step2 的 --stage1_ckpt 需要在 Step1 完成后填入实际 TIMESTAMP
============================================================


============================================================
Step 1 — Stage1 重训（三分类：背景/肝脏/肿瘤）
cd /home/pumengyu/medseg_project
============================================================

CUDA_VISIBLE_DEVICES=1 python -m scripts.train \
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
# Step 2 — Stage2 训练（双通道：CT + 粗糙肿瘤mask）
# cd /home/pumengyu/twostage_medseg
# 填入 Step1 训练完后的实际 TIMESTAMP
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/train_tumor_roi.py \
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
# Step 3 — eval
# cd /home/pumengyu/twostage_medseg
# 填入 Step1 和 Step2 训练完后的实际 TIMESTAMP
# ============================================================

CUDA_VISIBLE_DEVICES=1 python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage2_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox_roi_largetx6_p128/train/03-30-23-47-04/last.pt \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_out_channels 3 \
  --stage1_patch 144 144 144 --stage2_patch 128 128 128 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 \
  --split test \
  --use_coarse_tumor \
  --tta \
  --min_tumor_size 100


# ============================================================
# 实验九B eval — pred_bbox ROI + 大肿瘤×6 + patch=128
# stage1: dynunet_liver_only (两分类)
# stage2: tumor_dynunet_predbbox_roi_largetx6_p128 last.pt
# ============================================================

CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage2_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_predbbox_roi_largetx6_p128/train/03-29-13-33-35/best.pt \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 128 128 128 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 \
  --split test \
  --save_vis \
  --tta \
  --min_tumor_size 100
