CUDA_VISIBLE_DEVICES=0 /home/PuMengYu/anaconda3/envs/medseg/bin/python scripts/auto_postprocess_sweep.py \
  --medseg_root /home/PuMengYu/MSD_LiverTumorSeg/medseg_project \
  --preprocessed_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_roi \
  --stage1_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_tumor_stage1/train/03-29-21-29-13/best.pt \
  --stage2_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/focaltversky_deep_p160_sgd_ratio04/train/04-13-08-47-34/best.pt
