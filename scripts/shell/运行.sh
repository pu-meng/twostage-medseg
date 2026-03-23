python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments_twostage \
  --exp_name tumor_roi_dynunet \
  --model dynunet \
  --epochs 200 \
  --batch_size 1 \
  --lr 0.003 \
  --patch 96 96 96 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 6 \
  --num_workers 2 \
  --amp \
  --loss dicece \
  --overlap 0.5 \
  --repeats 3 \
  --tumor_ratios 0.2 0.8 \
  --margin 12 \
  --init_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt






python -m twostage.eval_twostage \
  --project_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/liver_stage1/train/xxx/best.pt \
  --stage2_ckpt /home/pumengyu/experiments_twostage/tumor_roi_dynunet/train/xxx/best.pt \
  --stage1_model dynunet \
  --stage2_model dynunet \
  --stage1_patch 144 144 144 \
  --stage2_patch 96 96 96 \
  --stage1_sw_batch_size 1 \
  --stage2_sw_batch_size 1 \
  --overlap 0.5 \
  --margin 12 \
  --save_dir /home/pumengyu/experiments_twostage_eval \
  --save_pred_pt




训练命令

two-stage  的tumor的训练命令
workdir = os.path.join(args.exp_root, args.exp_name, "train", timestamp)

CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet \
  --model dynunet \
  --epochs 300 \
  --batch_size 1 \
  --lr 0.003 \
  --patch 96 96 96 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 3 \
  --num_workers 2 \
  --amp \
  --loss dicefocal \
  --overlap 0.5 \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --init_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt




python scripts/eval_twostage.py \
--medseg_root /home/pumengyu/medseg_project \
--preprocessed_root /home/pumengyu/Task03_Liver_pt \
--stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
--stage2_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet/train/03-17-09-22-42/best.pt \
--stage1_model dynunet \
--stage2_model dynunet \
--stage1_patch 144 144 144 \
--stage2_patch 96 96 96 \
--val_ratio 0.2 \
--test_ratio 0.1 \
--seed 0 \
--split test \
--save_vis \
--vis_n 10 \
--margin 24 \
--save_dir /home/pumengyu/experiments/twostage/tumor_dynunet/eval


我们对train的roi进行改造




  --margin 12 \
  --bbox_jitter \
  --bbox_max_shift 8 \
  --random_margin \
  --margin_min 8 \
  --margin_max 20



CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_roi_jitter \
  --model dynunet \
  --epochs 300 \
  --batch_size 1 \
  --lr 0.003 \
  --patch 96 96 96 \
  --val_patch 96 96 96 \
  --sw_batch_size 1 \
  --val_every 3 \
  --num_workers 2 \
  --amp \
  --loss dicefocal \
  --overlap 0.5 \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --bbox_jitter \
  --bbox_max_shift 8 \
  --random_margin \
  --margin_max 20 \
  --init_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt



CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_roi_jitter \
  --model dynunet \
  --epochs 300 \
  --batch_size 2 \
  --lr 0.003 \
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
  --bbox_jitter \
  --bbox_max_shift 8 \
  --random_margin \
  --margin_max 20 \
  --resume /home/pumengyu/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-10-47-09/last.pt \
  --init_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt

                                                                                          


# ============================================================
训练模式说明
============================================================

【锤炼链 —— 推荐工作流】

  不要每次改代码都从头 300 epoch 重跑！
  用 --init_ckpt 指向上一次的 best.pt 热启动，不断积累。

  第一次（模式一/二）：--init_ckpt 指向 stage1 肝脏权重（迁移学习起点）
       ↓ 训完得到 best.pt (A)
  第二次修改后：--init_ckpt 指向 best.pt (A)
       ↓ 训完得到 best.pt (B)
  第三次 fine-tune(模式三)：--init_ckpt 指向 best.pt (B)
       ↓ ...

  --init_ckpt 只加载模型权重（形状匹配才载入，换架构也安全）
  optimizer / scheduler / epoch 全部重置 → 可以自由改 LR 和训练策略
  --resume 是断点续训（接着跑同一次，不改任何参数时用）

============================================================
【模式一】标准训练（无任何 hard mining，原始行为）
第一次跑：--init_ckpt 填 stage1 肝脏权重
后续迭代：--init_ckpt 填上一次肿瘤模型的 best.pt
============================================================



CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_standard \
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
  --init_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt


============================================================
【模式二】差异化 oversampling（小肿瘤/无肿瘤多采样）
小肿瘤(<5000 voxels) repeat×3，无肿瘤 repeat×4，正常 repeat×1
repeats 6 是基础重复次数，困难样本在此基础上乘以对应的 scale 系数
第一次跑：--init_ckpt 填 stage1 肝脏权重
后续迭代：--init_ckpt 填上一次肿瘤模型的 best.pt
============================================================


CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_hardmining \
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
  --small_tumor_thresh 5000 \
  --small_tumor_repeat_scale 3 \
  --no_tumor_repeat_scale 4 \
  --init_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt


============================================================
【模式三】困难样本 fine-tune（在已训练好的 Model A 基础上二次强化）
重点训练无肿瘤+小肿瘤 case，混入少量普通样本防止灾难遗忘
--normal_case_ratio 0.2 = 从普通样本里随机取 20% 混入
--init_ckpt 填 Model A 的 best.pt 路径
============================================================


CUDA_VISIBLE_DEVICES=0 python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --exp_root /home/pumengyu/experiments/twostage \
  --exp_name tumor_dynunet_hardfinetune \
  --model dynunet \
  --epochs 60 \
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
  --overlap 0.5 \
  --repeats 6 \
  --tumor_ratios 0.05 0.95 \
  --margin 8 \
  --bbox_jitter --bbox_max_shift 8 \
  --random_margin --margin_min 8 --margin_max 20 \
  --hard_cases_only \
  --small_tumor_thresh 5000 \
  --normal_case_ratio 0.2 \
  --init_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt


# ============================================================
# eval 命令
# ============================================================

# 【eval A】单模型评估（Model A）
python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage2_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --split test \
  --save_vis --vis_n 10 \
  --margin 24 \
  --save_dir /home/pumengyu/experiments/twostage/eval_modelA

# 【eval B】单模型评估（Model B fine-tune，看有无遗忘）
python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage2_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_hardfinetune/train/TIMESTAMP/best.pt \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --split test \
  --save_vis --vis_n 10 \
  --margin 24 \
  --save_dir /home/pumengyu/experiments/twostage/eval_modelB

# 【eval A+B ensemble】两模型融合（weight_b=0.5 各占一半）
python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage2_ckpt /home/pumengyu/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt \
  --stage2_ckpt_b /home/pumengyu/experiments/twostage/tumor_dynunet_hardfinetune/train/TIMESTAMP/best.pt \
  --ensemble_weight_b 0.5 \
  --stage1_model dynunet --stage2_model dynunet \
  --stage1_patch 144 144 144 --stage2_patch 96 96 96 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --split test \
  --save_vis --vis_n 10 \
  --margin 24 \
  --save_dir /home/pumengyu/experiments/twostage/eval_ensemble