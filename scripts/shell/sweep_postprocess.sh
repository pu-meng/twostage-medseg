#!/bin/bash
# ============================================================
# 后处理参数 Sweep — focaltversky_smallmine_zoom_p160
# GPU: 0
# 目标：在 0.7135 基础上通过后处理进一步提升 tumor dice
#
# 变量说明：
#   prob_threshold    : Stage2 概率图二值化阈值（默认0.3）
#   min_tumor_size    : 连通域最小体积，小于此值依赖 comp_prob_thresh 决定去留
#   comp_prob_thresh  : 小连通域保留的平均概率下限
#   small_tumor_low_thresh : 极小肿瘤(200~1000vox)激进低阈值（0=关闭）
# ============================================================

STAGE1_CKPT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt
STAGE2_CKPT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/focaltversky_smallmine_zoom_p160/train/04-12-21-15-06/best.pt
MEDSEG=/home/PuMengYu/MSD_LiverTumorSeg/medseg_project
PREPROC=/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_roi

BASE_ARGS="
  --medseg_root $MEDSEG
  --preprocessed_root $PREPROC
  --stage1_ckpt $STAGE1_CKPT
  --stage2_ckpt $STAGE2_CKPT
  --stage1_model dynunet --stage2_model dynunet
  --stage1_patch 144 144 144 --stage2_patch 128 128 128
  --stage2_sw_batch_size 1
  --overlap 0.25 --margin 8
  --val_ratio 0.2 --test_ratio 0.1 --seed 0
  --split test
"

run_eval() {
    local tag=$1; shift
    echo ""
    echo "========================================================"
    echo " 运行: $tag"
    echo "========================================================"
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_twostage.py \
        $BASE_ARGS \
        --save_dir /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/focaltversky_smallmine_zoom_p160/sweep/$tag \
        "$@"
    echo ">>> $tag 完成"
}

# ─── baseline（复现当前0.7135）─────────────────────────────────
run_eval "baseline_pt03_min100_cpt05_slt02" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

# ─── prob_threshold sweep（主要针对 liver_12 FP爆炸）───────────
run_eval "pt035_min100_cpt05_slt02" \
    --prob_threshold 0.35 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

run_eval "pt040_min100_cpt05_slt02" \
    --prob_threshold 0.4 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

run_eval "pt045_min100_cpt05_slt02" \
    --prob_threshold 0.45 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

# ─── min_tumor_size sweep（过滤小FP连通域）────────────────────
run_eval "pt03_min200_cpt05_slt02" \
    --prob_threshold 0.3 \
    --min_tumor_size 200 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

run_eval "pt03_min300_cpt05_slt02" \
    --prob_threshold 0.3 \
    --min_tumor_size 300 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

run_eval "pt03_min500_cpt05_slt02" \
    --prob_threshold 0.3 \
    --min_tumor_size 500 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

# ─── comp_prob_thresh sweep─────────────────────────────────────
run_eval "pt03_min100_cpt04_slt02" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.4 \
    --small_tumor_low_thresh 0.2

run_eval "pt03_min100_cpt06_slt02" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.6 \
    --small_tumor_low_thresh 0.2

# ─── small_tumor_low_thresh sweep──────────────────────────────
run_eval "pt03_min100_cpt05_slt00" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.0

run_eval "pt03_min100_cpt05_slt01" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.1

run_eval "pt03_min100_cpt05_slt015" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.15

# ─── 组合：高阈值 + 大最小体积（激进去FP）─────────────────────
run_eval "pt04_min300_cpt05_slt01" \
    --prob_threshold 0.4 \
    --min_tumor_size 300 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.1

run_eval "pt035_min200_cpt04_slt015" \
    --prob_threshold 0.35 \
    --min_tumor_size 200 \
    --comp_prob_thresh 0.4 \
    --small_tumor_low_thresh 0.15

# ─── 汇总结果──────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Sweep 完成，汇总各组 tumor dice："
echo "========================================================"
for d in /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/focaltversky_smallmine_zoom_p160/sweep/*/report.txt; do
    tag=$(basename $(dirname $d))
    dice=$(grep "Dice: mean=" $d | grep -v Liver | awk '{print $2}' | cut -d= -f2)
    recall=$(grep "Recall: mean=" $d | awk '{print $2}' | cut -d= -f2)
    fdr=$(grep "FDR: mean=" $d | awk '{print $2}' | cut -d= -f2)
    printf "%-45s  dice=%-6s  recall=%-6s  fdr=%s\n" "$tag" "$dice" "$recall" "$fdr"
done
