                                                                                                           
  # 先跑：无TTA（baseline）                                                                                  
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_twostage.py \                                                   
    --medseg_root /home/PuMengYu/MSD_LiverTumorSeg/medseg_project \                                                            
    --preprocessed_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt \                                                     
    --stage1_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
    --stage2_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt \
    --stage1_model dynunet \
    --stage2_model dynunet \         
    --stage1_patch 144 144 144 \
    --stage2_patch 96 96 96 \                                                     
    --val_ratio 0.2 \
    --test_ratio 0.1 \
    --seed 0 \             
    --split test \
    --margin 12                                                                                 
                                                            
  # 再跑：加TTA                                                                                              
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_twostage.py \  
    --medseg_root /home/PuMengYu/MSD_LiverTumorSeg/medseg_project \
    --preprocessed_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt \                                                     
    --stage1_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
    --stage2_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt \
    --stage1_model dynunet --stage2_model dynunet \                                                          
    --stage1_patch 144 144 144 --stage2_patch 96 96 96 \                                                     
    --val_ratio 0.2 --test_ratio 0.1 --seed 0 \                                                              
    --split test --margin 12 \                                                                               
    --tta
                                                                                                             
  同一个checkpoint，唯一区别就是有没有--tta，结果就能说明TTA的纯收益。  