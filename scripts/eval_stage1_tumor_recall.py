"""
快速评估 stage1 三分类模型在 test set 上的 tumor recall。
直接用 Task03_Liver_roi 的 pt 文件（已 crop 到 liver 范围），
在 ROI 上做滑窗推理，统计 tumor recall/precision/dice。

用法:
CUDA_VISIBLE_DEVICES=0 python scripts/eval_stage1_tumor_recall.py \
  --ckpt /home/PuMengYu/MSD_LiverTumorSeg/experiments/dynunet_liver_tumor_stage1/train/03-29-21-29-13/best.pt \
  --roi_pt_root /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_roi \
  --model dynunet \
  --patch 144 144 144 \
  --sw_batch_size 2 \
  --overlap 0.5 \
  --val_ratio 0.2 \
  --test_ratio 0.1 \
  --seed 0 \
  --split val

  
"""
from __future__ import annotations

import argparse
import os
import sys
import random
import numpy as np
import torch
from monai.inferers.utils import sliding_window_inference

sys.path.insert(0, "/home/PuMengYu/MSD_LiverTumorSeg/medseg_project")
sys.path.insert(0, "/home/PuMengYu/MSD_LiverTumorSeg/twostage_medseg")

from medseg.models.build_model import build_model
from twostage_medseg.metrics.metrics_utils import compute_metrics


def load_ckpt_flexible(path, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    dst = model.state_dict()
    matched = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    dst.update(matched)
    model.load_state_dict(dst, strict=False)
    print(f"[load] {len(matched)}/{len(dst)} params matched from {path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_three_ways(items, val_ratio=0.2, test_ratio=0.1, seed=0):
    rng = random.Random(seed)
    items = sorted(items)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    te = shuffled[:n_test]
    va = shuffled[n_test:n_test + n_val]
    tr = shuffled[n_test + n_val:]
    return tr, va, te


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--roi_pt_root", type=str, default="/home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_roi")
    p.add_argument("--model", type=str, default="dynunet")
    p.add_argument("--patch", type=int, nargs=3, default=[144, 144, 144])
    p.add_argument("--sw_batch_size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--min_tumor_size", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    import glob
    all_pts = sorted(glob.glob(os.path.join(args.roi_pt_root, "*.pt")))
    tr, va, te = split_three_ways(all_pts, args.val_ratio, args.test_ratio, args.seed)
    pt_paths = {"train": tr, "val": va, "test": te, "all": all_pts}[args.split]

    print(f"split={args.split}  n_cases={len(pt_paths)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, in_channels=1, out_channels=3,
                        img_size=tuple(args.patch)).to(device)
    load_ckpt_flexible(args.ckpt, model, map_location=device)
    model.eval()
    print(f"ckpt loaded: {args.ckpt}")

    rows = []
    with torch.no_grad():
        for pt_path in pt_paths:
            case_name = os.path.basename(pt_path).replace(".pt", "")
            data = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
            image = data["image"].float()   # [1, D, H, W]
            label = data["label"][0].long() # [D, H, W]

            x = image.unsqueeze(0).to(device)  # [1,1,D,H,W]
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
                logits = sliding_window_inference(
                    x, tuple(args.patch), args.sw_batch_size, model, args.overlap
                )
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = torch.as_tensor(logits).float()

            # softmax prob for tumor (channel 2)
            prob = torch.softmax(logits[0], dim=0)  # [3, D, H, W]
            tumor_prob = prob[2].cpu()               # [D, H, W]  ← 软概率

            pred = torch.argmax(logits[0], dim=0).cpu()  # [D, H, W]
            tumor_pred = pred == 2

            gt_tumor = label == 2
            gt_has_tumor = gt_tumor.sum().item() > 0

            m = compute_metrics(tumor_pred, gt_tumor)

            prob_max = float(tumor_prob.max().item())
            prob_mean_in_gt = float(tumor_prob[gt_tumor].mean().item()) if gt_has_tumor else float("nan")

            rows.append({
                "case": case_name,
                "gt_tumor_vox": int(gt_tumor.sum().item()),
                "pred_tumor_vox": int(tumor_pred.sum().item()),
                "dice": round(m["Dice"], 4),
                "recall": round(m["Recall"], 4),
                "precision": round(m["Precision"], 4),
                "FDR": round(m["FDR"], 4),
                "prob_max": round(prob_max, 4),
                "prob_mean_in_gt": round(prob_mean_in_gt, 4) if gt_has_tumor else "N/A",
            })
            print(f"  {case_name:12s}  gt={int(gt_tumor.sum()):>8,}  "
                  f"dice={m['Dice']:.3f}  recall={m['Recall']:.3f}  "
                  f"prec={m['Precision']:.3f}  prob_max={prob_max:.3f}  "
                  f"prob_mean_in_gt={prob_mean_in_gt:.3f}" if gt_has_tumor else
                  f"  {case_name:12s}  gt={int(gt_tumor.sum()):>8,}  (no tumor)")

    # 汇总
    has_tumor = [r for r in rows if r["gt_tumor_vox"] > 0]
    if has_tumor:
        mean_dice = np.mean([r["dice"] for r in has_tumor])
        mean_recall = np.mean([r["recall"] for r in has_tumor])
        mean_prec = np.mean([r["precision"] for r in has_tumor])
        mean_fdr = np.mean([r["FDR"] for r in has_tumor])
        print("\n" + "=" * 60)
        print(f"有肿瘤 case: n={len(has_tumor)}")
        print(f"  Dice:      {mean_dice:.4f}")
        print(f"  Recall:    {mean_recall:.4f}   ← 关键：这个决定stage2的recall上限")
        print(f"  Precision: {mean_prec:.4f}")
        print(f"  FDR:       {mean_fdr:.4f}")
        print()
        print("Recall 低的 case（recall < 0.5）：")
        low_recall = sorted([r for r in has_tumor if r["recall"] < 0.5],
                            key=lambda x: x["recall"])
        for r in low_recall:
            print(f"  {r['case']:12s}  recall={r['recall']:.3f}  "
                  f"dice={r['dice']:.3f}  gt={r['gt_tumor_vox']:,}")


if __name__ == "__main__":
    main()
