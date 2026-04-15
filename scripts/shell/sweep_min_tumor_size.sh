#!/bin/bash
# 扫描 min_tumor_size 后处理阈值，在 val set 上找最优值
# 使用当前最优 checkpoint（roi_jitter + TTA = 0.5965）
#  bash scripts/shell/sweep_min_tumor_size.sh 2>&1 | tee /tmp/sweep_min_tumor_size.log  
# GPU 1

STAGE1_CKPT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt
STAGE2_CKPT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/tumor_dynunet_roi_jitter/train/03-22-11-44-00/best.pt
MEDSEG_ROOT=/home/PuMengYu/MSD_LiverTumorSeg/medseg_project
PREPROCESSED_ROOT=/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt
SAVE_DIR=/home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/tumor_dynunet_roi_jitter/eval_min_tumor_size_sweep

for SIZE in 0 100 200 500 1000; do
    echo "========================================"
    echo "[sweep] min_tumor_size=${SIZE}"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
        --medseg_root     $MEDSEG_ROOT \
        --preprocessed_root $PREPROCESSED_ROOT \
        --stage1_ckpt     $STAGE1_CKPT \
        --stage2_ckpt     $STAGE2_CKPT \
        --stage1_model    dynunet \
        --stage2_model    dynunet \
        --stage1_patch    144 144 144 \
        --stage2_patch    96 96 96 \
        --val_ratio       0.2 \
        --test_ratio      0.1 \
        --seed            0 \
        --split           test \
        --margin          12 \
        --tta \
        --min_tumor_size  $SIZE \
        --save_dir        $SAVE_DIR/size_${SIZE}
done

echo ""
echo "========================================"
echo "[sweep] 汇总各阈值 Tumor Dice（val set）："
echo "========================================"
for SIZE in 0 100 200 500 1000; do
    METRICS=$SAVE_DIR/size_${SIZE}/*/metrics.json
    DICE=$(python3 -c "
import json, glob
files = glob.glob('$SAVE_DIR/size_${SIZE}/*/metrics.json')
if files:
    d = json.load(open(files[0]))
    print(f\"{d['tumor']['Dice']['mean']:.4f}\")
else:
    print('N/A')
" 2>/dev/null)
    echo "  min_tumor_size=${SIZE}: Tumor Dice = ${DICE}"
done
