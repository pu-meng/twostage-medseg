"""
检测后处理（min_tumor_size + liver_filled 约束）对某个 case 的影响。
用法：
  python scripts/debug_case_postprocess.py \
    --pt /path/to/liver_121_twostage.pt \
    --min_tumor_sizes 0 10 50 100 200
"""
import argparse
import scipy.ndimage as ndi
import torch
import numpy as np


def dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0
    return 2 * inter / union


def apply_postprocess(stage2_np, liver_filled_np, min_tumor_size):
    """复现 eval_twostage.py 的后处理流程，返回最终肿瘤 mask。"""
    # 1. 约束在 liver_filled 内
    tumor = (stage2_np == 1) & (liver_filled_np == 1)
    # 2. 去除小连通域
    if min_tumor_size > 0:
        labeled, n = ndi.label(tumor)
        for i in range(1, n + 1):
            if (labeled == i).sum() < min_tumor_size:
                tumor[labeled == i] = False
    return tumor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True, help="twostage pred .pt 文件路径")
    p.add_argument(
        "--min_tumor_sizes",
        type=int,
        nargs="+",
        default=[0, 10, 50, 100, 200],
        help="要测试的 min_tumor_size 列表",
    )
    args = p.parse_args()

    data = torch.load(args.pt, map_location="cpu", weights_only=False)
    label = data["label"][0].numpy()          # [D,H,W]
    stage2 = data["stage2_tumor_pred"][0].numpy()
    liver_filled = data["stage1_liver_filled"][0].numpy()

    gt_tumor = label == 2
    gt_liver = label >= 1

    print(f"Case: {data['meta']['case_name']}")
    print(f"GT tumor voxels : {gt_tumor.sum()}")
    print(f"GT tumor size cat: {data['meta'].get('tumor_size_cat', 'N/A')}")
    print()

    # --- Stage2 原始预测统计 ---
    stage2_raw = stage2 == 1
    raw_inter = (stage2_raw & gt_tumor).sum()
    print(f"Stage2 raw pred voxels: {stage2_raw.sum()}")
    print(f"Stage2 raw overlap with GT tumor: {raw_inter}  ({raw_inter/max(gt_tumor.sum(),1)*100:.1f}% recall)")
    print()

    # --- liver_filled 约束的影响 ---
    tumor_in_liver = stage2_raw & (liver_filled == 1)
    removed_by_liver = stage2_raw & (liver_filled == 0)
    overlap_removed = (removed_by_liver & gt_tumor).sum()
    print(f"Removed by liver_filled constraint: {removed_by_liver.sum()} voxels")
    print(f"  of which overlap with GT tumor  : {overlap_removed}  ← 后处理1误删的GT体素")
    print()

    # --- 连通域分析（liver_filled 约束后） ---
    labeled, n_comp = ndi.label(tumor_in_liver)
    comp_sizes = sorted(
        [(i + 1, int((labeled == i + 1).sum())) for i in range(n_comp)],
        key=lambda x: -x[1],
    )
    print(f"连通域数量（liver_filled 约束后）: {n_comp}")
    print("各连通域大小 | 与GT肿瘤重叠:")
    for comp_id, sz in comp_sizes:
        comp_mask = labeled == comp_id
        overlap = int((comp_mask & gt_tumor).sum())
        print(f"  component {comp_id:2d}: {sz:6d} voxels  GT overlap={overlap}")
    print()

    # --- 不同 min_tumor_size 下的 Dice ---
    print(f"{'min_tumor_size':>15} | {'pred_voxels':>12} | {'tumor_dice':>10} | {'recall':>8} | {'precision':>10}")
    print("-" * 65)
    for mts in args.min_tumor_sizes:
        pred = apply_postprocess(stage2, liver_filled, mts)
        d = dice(pred, gt_tumor)
        recall = (pred & gt_tumor).sum() / max(gt_tumor.sum(), 1)
        prec = (pred & gt_tumor).sum() / max(pred.sum(), 1)
        print(f"{mts:>15} | {pred.sum():>12} | {d:>10.4f} | {recall:>8.4f} | {prec:>10.4f}")


if __name__ == "__main__":
    main()
