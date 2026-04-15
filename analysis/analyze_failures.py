"""
analyze_failures.py
读取 eval 保存的 pred_pt，对每个 case 深度分析：
- 肿瘤大小分布 vs Dice
- FP/FN 分析
- 无肿瘤 case 的假阳性情况
- 可视化 worst cases

用法:
  python scripts/analyze_failures.py \
    --pred_dir /home/PuMengYu/MSD_LiverTumorSeg/experiments/analysis/ckpt_A/test/03-29-13-33-35/pred_pt \
    --out_dir  /home/PuMengYu/MSD_LiverTumorSeg/experiments/analysis/ckpt_A/test/analysis
  
    
    python scripts/analyze_failures.py \
     --pred_dir /home/PuMengYu/MSD_LiverTumorSeg/experiments/analysis/ckpt_B/test/03-30-23-47-04/pred_pt \
     --out_dir  /home/PuMengYu/MSD_LiverTumorSeg/experiments/analysis/ckpt_B/test/analysis

    

"""
import argparse
import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def dice(pred, gt):
    pred = pred.bool()
    gt = gt.bool()
    inter = (pred & gt).sum().float()
    denom = pred.sum().float() + gt.sum().float()
    if denom == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return (2 * inter / denom).item()


def compute_case_stats(data):
    label = data["label"][0].long()       # [D,H,W]
    final_pred = data["final_pred"][0].long()

    gt_tumor = label == 2
    pred_tumor = final_pred == 2
    gt_liver = label > 0
    pred_liver = final_pred > 0

    gt_tumor_voxels = gt_tumor.sum().item()
    pred_tumor_voxels = pred_tumor.sum().item()

    tumor_dice = dice(pred_tumor, gt_tumor)
    liver_dice = dice(pred_liver, gt_liver)

    # FP/FN voxels
    fp = (pred_tumor & ~gt_tumor).sum().item()
    fn = (~pred_tumor & gt_tumor).sum().item()
    tp = (pred_tumor & gt_tumor).sum().item()

    # GT 肿瘤连通域数量和大小
    labeled_gt, n_gt = ndimage.label(gt_tumor.numpy())
    gt_cc_sizes = [int(ndimage.sum(gt_tumor.numpy(), labeled_gt, i+1)) for i in range(n_gt)]

    # Pred 肿瘤连通域数量和大小
    labeled_pred, n_pred = ndimage.label(pred_tumor.numpy())
    pred_cc_sizes = [int(ndimage.sum(pred_tumor.numpy(), labeled_pred, i+1)) for i in range(n_pred)]

    return {
        "gt_tumor_voxels": gt_tumor_voxels,
        "pred_tumor_voxels": pred_tumor_voxels,
        "tumor_dice": tumor_dice,
        "liver_dice": liver_dice,
        "tp": tp, "fp": fp, "fn": fn,
        "n_gt_cc": n_gt,
        "gt_cc_sizes": gt_cc_sizes,
        "n_pred_cc": n_pred,
        "pred_cc_sizes": pred_cc_sizes,
        "has_tumor": gt_tumor_voxels > 0,
    }


def plot_scatter_size_vs_dice(cases, out_dir):
    """GT肿瘤大小 vs Tumor Dice 散点图，按有无肿瘤着色"""
    with_tumor = [(c["gt_tumor_voxels"], c["tumor_dice"]) for c in cases if c["has_tumor"]]
    no_tumor   = [(c["gt_tumor_voxels"], c["tumor_dice"]) for c in cases if not c["has_tumor"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    if with_tumor:
        xs, ys = zip(*with_tumor)
        ax.scatter(xs, ys, c="steelblue", label="has tumor", alpha=0.8, s=60)
        for (x, y), c in zip(with_tumor, [c for c in cases if c["has_tumor"]]):
            ax.annotate(c["case_name"][:10], (x, y), fontsize=6, alpha=0.6)
    if no_tumor:
        xs, ys = zip(*no_tumor)
        ax.scatter(xs, ys, c="tomato", label="no tumor", marker="x", s=60)

    ax.set_xlabel("GT Tumor Voxels")
    ax.set_ylabel("Tumor Dice")
    ax.set_title("GT Tumor Size vs Tumor Dice")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "size_vs_dice.png"), dpi=120)
    plt.close()


def plot_fp_fn(cases, out_dir):
    """每个 case 的 FP/FN/TP 条形图"""
    names = [c["case_name"][:12] for c in cases]
    tp = [c["tp"] for c in cases]
    fp = [c["fp"] for c in cases]
    fn = [c["fn"] for c in cases]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(names)*0.8), 5))
    ax.bar(x - width, tp, width, label="TP", color="steelblue")
    ax.bar(x,          fp, width, label="FP", color="tomato")
    ax.bar(x + width,  fn, width, label="FN", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Voxels")
    ax.set_title("Per-case TP / FP / FN")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fp_fn_per_case.png"), dpi=120)
    plt.close()


def visualize_worst_cases(cases, pred_dir, out_dir, n=5):
    """对 Dice 最差的 n 个 case 画三视图对比"""
    with_tumor = [c for c in cases if c["has_tumor"]]
    worst = sorted(with_tumor, key=lambda c: c["tumor_dice"])[:n]

    for info in worst:
        pt_path = os.path.join(pred_dir, f"{info['case_name']}_twostage.pt")
        if not os.path.exists(pt_path):
            continue
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        image = data["image"][0].float()        # [D,H,W]
        label = data["label"][0].long()
        final_pred = data["final_pred"][0].long()

        gt_tumor = (label == 2)
        pred_tumor = (final_pred == 2)

        # 选肿瘤重心切片
        coords = torch.nonzero(gt_tumor, as_tuple=False)
        if coords.numel() > 0:
            center = coords.float().mean(0)
            z = int(center[0].item())
            y = int(center[1].item())
            x = int(center[2].item())
        else:
            D, H, W = image.shape
            z, y, x = D//2, H//2, W//2

        D, H, W = image.shape
        z = max(0, min(z, D-1))
        y = max(0, min(y, H-1))
        x = max(0, min(x, W-1))

        # 三视图
        views = [
            ("Axial",    image[z],    label[z],    final_pred[z]),
            ("Coronal",  image[:,y,:], label[:,y,:], final_pred[:,y,:]),
            ("Sagittal", image[:,:,x], label[:,:,x], final_pred[:,:,x]),
        ]

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        fig.suptitle(
            f"{info['case_name']}  Tumor Dice={info['tumor_dice']:.4f}\n"
            f"GT={info['gt_tumor_voxels']} vox  Pred={info['pred_tumor_voxels']} vox  "
            f"FP={info['fp']}  FN={info['fn']}",
            fontsize=10
        )

        cmap_ct = "gray"
        for row, (view_name, ct_slice, gt_slice, pred_slice) in enumerate(views):
            ct_np = ct_slice.numpy()
            gt_np = gt_slice.numpy()
            pred_np = pred_slice.numpy()

            # CT
            axes[row, 0].imshow(ct_np, cmap=cmap_ct, aspect="auto")
            axes[row, 0].set_title(f"{view_name} CT", fontsize=8)
            axes[row, 0].axis("off")

            # GT overlay
            axes[row, 1].imshow(ct_np, cmap=cmap_ct, aspect="auto")
            gt_liver_mask = (gt_np > 0).astype(float)
            gt_tumor_mask = (gt_np == 2).astype(float)
            axes[row, 1].imshow(gt_liver_mask, cmap="Blues", alpha=0.25, aspect="auto", vmin=0, vmax=1)
            axes[row, 1].imshow(gt_tumor_mask, cmap="Reds",  alpha=0.5,  aspect="auto", vmin=0, vmax=1)
            axes[row, 1].set_title("GT", fontsize=8)
            axes[row, 1].axis("off")

            # Pred overlay
            axes[row, 2].imshow(ct_np, cmap=cmap_ct, aspect="auto")
            pred_liver_mask = (pred_np > 0).astype(float)
            pred_tumor_mask = (pred_np == 2).astype(float)
            axes[row, 2].imshow(pred_liver_mask, cmap="Blues", alpha=0.25, aspect="auto", vmin=0, vmax=1)
            axes[row, 2].imshow(pred_tumor_mask, cmap="Reds",  alpha=0.5,  aspect="auto", vmin=0, vmax=1)
            axes[row, 2].set_title("Pred", fontsize=8)
            axes[row, 2].axis("off")

        red_patch   = mpatches.Patch(color="red",  alpha=0.5, label="Tumor")
        blue_patch  = mpatches.Patch(color="blue", alpha=0.25, label="Liver")
        fig.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"worst_{info['case_name']}_dice{info['tumor_dice']:.3f}.png")
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  saved: {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(args.pred_dir, "*_twostage.pt")))
    if not pt_files:
        print(f"No *_twostage.pt found in {args.pred_dir}")
        return

    print(f"Found {len(pt_files)} cases")
    cases = []
    for pt_path in pt_files:
        case_name = os.path.basename(pt_path).replace("_twostage.pt", "")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        stats = compute_case_stats(data)
        stats["case_name"] = case_name
        cases.append(stats)

    # ---- 打印汇总表格 ----
    print(f"\n{'case':<20} {'gt_vox':>8} {'pred_vox':>9} {'dice':>6} {'fp':>8} {'fn':>8} {'n_gt_cc':>7} {'gt_cc_sizes'}")
    print("-" * 90)
    for c in sorted(cases, key=lambda x: x["tumor_dice"]):
        gt_sizes_str = str(sorted(c["gt_cc_sizes"], reverse=True)[:5])
        print(f"{c['case_name']:<20} {c['gt_tumor_voxels']:>8} {c['pred_tumor_voxels']:>9} "
              f"{c['tumor_dice']:>6.4f} {c['fp']:>8} {c['fn']:>8} {c['n_gt_cc']:>7}  {gt_sizes_str}")

    # ---- 类别统计 ----
    has_tumor = [c for c in cases if c["has_tumor"]]
    no_tumor  = [c for c in cases if not c["has_tumor"]]
    print(f"\n有肿瘤 cases: {len(has_tumor)}")
    if has_tumor:
        dices = [c["tumor_dice"] for c in has_tumor]
        print(f"  Dice mean={np.mean(dices):.4f}  std={np.std(dices):.4f}  min={np.min(dices):.4f}  max={np.max(dices):.4f}")
        small = [c for c in has_tumor if c["gt_tumor_voxels"] < 5000]
        large = [c for c in has_tumor if c["gt_tumor_voxels"] >= 5000]
        if small:
            print(f"  小肿瘤(<5000 vox) {len(small)} cases  Dice mean={np.mean([c['tumor_dice'] for c in small]):.4f}")
        if large:
            print(f"  大肿瘤(>=5000 vox) {len(large)} cases  Dice mean={np.mean([c['tumor_dice'] for c in large]):.4f}")

    print(f"\n无肿瘤 cases: {len(no_tumor)}")
    if no_tumor:
        fp_list = [c["fp"] for c in no_tumor]
        print(f"  FP voxels: mean={np.mean(fp_list):.0f}  max={np.max(fp_list):.0f}")
        false_pos_cases = [c for c in no_tumor if c["fp"] > 0]
        print(f"  有假阳性的 cases: {len(false_pos_cases)}/{len(no_tumor)}")

    # ---- 图表 ----
    print("\n生成图表...")
    plot_scatter_size_vs_dice(cases, args.out_dir)
    plot_fp_fn(cases, args.out_dir)
    visualize_worst_cases(cases, args.pred_dir, args.out_dir, n=5)

    print(f"\n分析结果保存至: {args.out_dir}")


if __name__ == "__main__":
    main()
