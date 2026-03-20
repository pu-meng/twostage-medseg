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

python scripts/train_tumor_roi.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /data/Task03_Liver_pt \
  --exp_name tumor_roi_dynunet_jitter \
  --model dynunet \
  --epochs 200 \
  --batch_size 1 \
  --lr 3e-3 \
  --patch 96 96 96 \
  --val_patch 96 96 96 \
  --tumor_ratios 0.2 0.8 \
  --margin 12 \
  --bbox_jitter \
  --bbox_max_shift 8 \
  --random_margin \
  --margin_min 8 \
  --margin_max 20