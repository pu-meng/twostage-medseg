#!/bin/bash
# ============================================================
# 后处理参数 Sweep2 — focaltversky_smallmine_zoom_p160
# GPU: 0
#
# sweep1 发现问题：用的是 patch=128/overlap=0.25，复现出来是0.679而非0.7135
# sweep2 修正：全部改用 patch=160/overlap=0.5（与跑出0.7135的命令一致）
#
# 本轮目标：
#   1. 裸模型（--no_postprocess）：看后处理到底贡献了多少
#   2. 正确 baseline 复现 0.7135
#   3. 在正确基础上补充 sweep1 未覆盖的组合
# ============================================================

STAGE1_CKPT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt
STAGE2_CKPT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/focaltversky_smallmine_zoom_p160/train/04-12-21-15-06/best.pt
MEDSEG=/home/PuMengYu/MSD_LiverTumorSeg/medseg_project
PREPROC=/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_roi
SWEEP_ROOT=/home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/focaltversky_smallmine_zoom_p160/sweep2

BASE_ARGS="
  --medseg_root $MEDSEG
  --preprocessed_root $PREPROC
  --stage1_ckpt $STAGE1_CKPT
  --stage2_ckpt $STAGE2_CKPT
  --stage1_model dynunet --stage2_model dynunet
  --stage1_patch 144 144 144
  --stage2_patch 160 160 160
  --stage2_sw_batch_size 1
  --overlap 0.5
  --margin 8
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
        --save_dir $SWEEP_ROOT/$tag \
        "$@"
    echo ">>> $tag 完成"
}

# ─── 1. 裸模型（无任何后处理）──────────────────────────────────
# 看 fill_holes + 肝脏约束 + 连通域过滤 总共贡献了多少 dice
run_eval "no_postprocess" \
    --no_postprocess \
    --prob_threshold 0.3

# ─── 2. 正确 baseline（复现 0.7135）────────────────────────────
run_eval "baseline_pt03_slt02" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

# ─── 3. slt 单独 sweep（固定其他为默认）────────────────────────
# sweep1 漏掉了 slt 的几组，补上
run_eval "slt00" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.0

run_eval "slt01" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.1

run_eval "slt015" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.15

run_eval "slt025" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.25

# ─── 4. prob_threshold sweep（正确 patch/overlap）───────────────
run_eval "pt035_slt02" \
    --prob_threshold 0.35 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

run_eval "pt040_slt02" \
    --prob_threshold 0.4 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

run_eval "pt045_slt02" \
    --prob_threshold 0.45 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

# ─── 5. min_tumor_size sweep（正确 patch/overlap）──────────────
run_eval "min200_slt02" \
    --prob_threshold 0.3 \
    --min_tumor_size 200 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

run_eval "min500_slt02" \
    --prob_threshold 0.3 \
    --min_tumor_size 500 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2

# ─── 6. last.pt 裸模型 vs 后处理对比──────────────────────────
# 看 last.pt 是否比 best.pt 更好（训练验证 dice 选的是 best，但 last 可能泛化更好）
run_eval "last_no_postprocess" \
    --no_postprocess \
    --prob_threshold 0.3 \
    --stage2_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/focaltversky_smallmine_zoom_p160/train/04-12-21-15-06/last.pt

run_eval "last_baseline" \
    --prob_threshold 0.3 \
    --min_tumor_size 100 \
    --comp_prob_thresh 0.5 \
    --small_tumor_low_thresh 0.2 \
    --stage2_ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/twostage/focaltversky_smallmine_zoom_p160/train/04-12-21-15-06/last.pt

# ─── 汇总──────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Sweep2 完成，汇总（按 tumor dice 排序）："
echo "========================================================"
find $SWEEP_ROOT -name "report.txt" | sort | while read f; do
    tag=$(echo $f | sed "s|$SWEEP_ROOT/||" | sed 's|/.*||')
    dice=$(grep "Dice: mean=" $f | grep -v Liver | awk -F= '{print $2}' | awk '{print $1}')
    recall=$(grep "Recall: mean=" $f | awk -F= '{print $2}' | awk '{print $1}')
    fdr=$(grep "FDR: mean=" $f | awk -F= '{print $2}' | awk '{print $1}')
    printf "%-35s  dice=%-8s  recall=%-8s  fdr=%s\n" "$tag" "$dice" "$recall" "$fdr"
done | sort -t= -k2 -rn
