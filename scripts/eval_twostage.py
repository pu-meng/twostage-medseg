# eval_twostage_simple.py
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import scipy.ndimage as ndi

import matplotlib.pyplot as plt
import sys
import time
from typing import Dict, List

import torch
from monai.inferers.utils import sliding_window_inference

from medseg.data.dataset_offline import split_three_ways

# twostage_medseg/metrics/filter.py
from metrics.filter import filter_largest_component


def add_medseg_to_syspath(medseg_root: str) -> None:
    medseg_root = os.path.abspath(medseg_root)
    if medseg_root not in sys.path:
        sys.path.insert(0, medseg_root)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--medseg_root", type=str, required=True)
    p.add_argument("--preprocessed_root", type=str, required=True)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)

    p.add_argument("--stage1_model", type=str, default="dynunet")
    p.add_argument("--stage2_model", type=str, default="dynunet")

    p.add_argument("--stage1_patch", type=int, nargs=3, default=[144, 144, 144])
    p.add_argument("--stage2_patch", type=int, nargs=3, default=[96, 96, 96])

    p.add_argument("--stage1_sw_batch_size", type=int, default=1)
    p.add_argument("--stage2_sw_batch_size", type=int, default=1)

    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--margin", type=int, default=12)

    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
    )

    p.add_argument("--n", type=int, default=0, help="只跑前N个case, 0=全部")
    p.add_argument("--save_pred_pt", action="store_true", help="是否保存预测结果")
    p.add_argument("--save_dir", type=str, default="./experiments_twostage_eval")
    p.add_argument("--save_vis", action="store_true", help="保存可视化png")
    p.add_argument("--vis_n", type=int, default=10, help="最多保存前N个case的可视化")
    p.add_argument("--min_tumor_size", type=int, default=50)

    return p.parse_args()


def safe_case_name(pt_path: str) -> str:
    name = os.path.basename(pt_path)
    if name.endswith(".pt"):
        name = name[:-3]
    return name


def pick_slice_indices(mask_or_label: torch.Tensor) -> Dict[str, int]:
    """
    选三个方向上更有信息量的切片:
    - 如果有前景, 取前景中心
    - 如果没有前景, 取体积中心
    输入: [D,H,W]
    返回: {"axial": z, "coronal": y, "sagittal": x}
    """
    assert mask_or_label.ndim == 3

    D, H, W = mask_or_label.shape
    coords = torch.nonzero(mask_or_label > 0, as_tuple=False)

    if coords.numel() == 0:
        return {
            "axial": D // 2,
            "coronal": H // 2,
            "sagittal": W // 2,
        }

    center = coords.float().mean(dim=0)
    z = int(round(center[0].item()))
    y = int(round(center[1].item()))
    x = int(round(center[2].item()))

    z = max(0, min(z, D - 1))
    y = max(0, min(y, H - 1))
    x = max(0, min(x, W - 1))

    return {
        "axial": z,
        "coronal": y,
        "sagittal": x,
    }


def save_case_visualization(
    save_path: str,
    image: torch.Tensor,  # [1,D,H,W]
    label: torch.Tensor | None,  # [1,D,H,W] or None
    pred1: torch.Tensor,  # [D,H,W]
    tumor_full: torch.Tensor,  # [D,H,W]
    final_pred: torch.Tensor,  # [D,H,W]
    case_name: str,
) -> None:
    """
    保存 one-case 可视化:
    行 = axial/coronal/sagittal
    列 = image / gt / liver_pred / tumor_pred / final_pred
    """
    image3d = image[0].cpu()
    gt3d = label[0].cpu() if label is not None else None

    # 优先根据GT选切片, 没GT就根据预测选
    ref = gt3d if gt3d is not None else final_pred
    idxs = pick_slice_indices(ref)

    def get_views(vol: torch.Tensor, idxs: Dict[str, int]):
        # 返回 3 个方向切片, 并转成常见显示方向
        axial = vol[idxs["axial"], :, :]  # [H,W]
        coronal = vol[:, idxs["coronal"], :]  # [D,W]
        sagittal = vol[:, :, idxs["sagittal"]]  # [D,H]

        return [
            axial.numpy(),
            coronal.numpy(),
            sagittal.numpy(),
        ]

    img_views = get_views(image3d, idxs)
    gt_views = get_views(gt3d, idxs) if gt3d is not None else [None, None, None]
    liver_views = get_views(pred1, idxs)
    tumor_views = get_views(tumor_full, idxs)
    final_views = get_views(final_pred, idxs)

    row_names = ["axial", "coronal", "sagittal"]
    col_names = ["image", "gt", "stage1_liver", "stage2_tumor", "final_pred"]

    fig, axes = plt.subplots(3, 5, figsize=(18, 10))

    for r in range(3):
        for c in range(5):
            ax = axes[r, c]
            ax.axis("off")

            if c == 0:
                ax.imshow(img_views[r], cmap="gray")
            elif c == 1:
                if gt_views[r] is not None:
                    ax.imshow(gt_views[r], cmap="gray", vmin=0, vmax=2)
                else:
                    ax.text(0.5, 0.5, "No GT", ha="center", va="center")
            elif c == 2:
                ax.imshow(liver_views[r], cmap="gray", vmin=0, vmax=1)
            elif c == 3:
                ax.imshow(tumor_views[r], cmap="gray", vmin=0, vmax=1)
            elif c == 4:
                ax.imshow(final_views[r], cmap="gray", vmin=0, vmax=2)

            if r == 0:
                ax.set_title(col_names[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(row_names[r], fontsize=11)

    fig.suptitle(case_name, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def load_pt_paths(preprocessed_root: str) -> List[str]:
    pt_paths = sorted(glob.glob(os.path.join(preprocessed_root, "*.pt")))
    if len(pt_paths) == 0:
        raise FileNotFoundError(f"no .pt found in {preprocessed_root}")
    return pt_paths


def dice_binary(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.bool()
    gt = gt.bool()

    inter = (pred & gt).sum().item()
    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum + gt_sum == 0:
        return 0.0
    return 2.0 * inter / (pred_sum + gt_sum + 1e-8)


def summarize_metrics_list(metrics_list: List[Dict], keys: List[str]) -> Dict:
    return {k: summarize_metric([m[k] for m in metrics_list]) for k in keys}


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    # pred = bool(pred) #bool()会把整个tensor变成True/False
    # gt = bool(gt)
    pred = pred.bool()
    gt = gt.bool()  # .bool()是逐元素转换,保持tensor形状不变
    TP = (pred & gt).sum().item()
    FP = (pred & ~gt).sum().item()
    FN = (~pred & gt).sum().item()
    TN = (~pred & ~gt).sum().item()
    dice = dice_binary(pred, gt)
    jaccard = TP / (TP + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    FPR = FP / (FP + TN + 1e-8)
    FDR = FP / (FP + TP + 1e-8)
    NPV = TN / (TN + FN + 1e-8)
    ACC = (TP + TN) / (TP + FP + FN + TN + 1e-8)
    FNR = FN / (TP + FN + 1e-8)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Dice": dice,
        "Jaccard": jaccard,
        "Precision": precision,
        "Recall": recall,
        "FPR": FPR,
        "FDR": FDR,
        "FNR": FNR,
        "NPV": NPV,
        "ACC": ACC,
    }


def summarize_metric(xs: List[float]) -> Dict[str, float]:
    if len(xs) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    x = torch.tensor(xs, dtype=torch.float32)
    return {
        "mean": round(float(x.mean().item()), 4),
        "std": round(float(x.std(unbiased=False).item()), 4),
        "min": round(float(x.min().item()), 4),
        "max": round(float(x.max().item()), 4),
    }


def compute_bbox_from_mask(mask: torch.Tensor, margin: int = 12):
    """
    mask: [D,H,W] bool
    return: ((z0,z1),(y0,y1),(x0,x1)) 右边界开区间
    """
    if mask.sum().item() == 0:
        return None

    coords = torch.nonzero(mask, as_tuple=False)
    zmin, ymin, xmin = coords.min(dim=0).values.tolist()
    zmax, ymax, xmax = coords.max(dim=0).values.tolist()

    D, H, W = mask.shape

    z0 = max(0, zmin - margin)
    y0 = max(0, ymin - margin)
    x0 = max(0, xmin - margin)

    z1 = min(D, zmax + 1 + margin)
    y1 = min(H, ymax + 1 + margin)
    x1 = min(W, xmax + 1 + margin)

    return ((z0, z1), (y0, y1), (x0, x1))


def crop_3d(x: torch.Tensor, bbox):
    """
    x: [1,D,H,W] or [D,H,W]
    """
    (z0, z1), (y0, y1), (x0, x1) = bbox
    if x.ndim == 4:
        return x[:, z0:z1, y0:y1, x0:x1]
    elif x.ndim == 3:
        return x[z0:z1, y0:y1, x0:x1]
    else:
        raise ValueError(f"unsupported ndim={x.ndim}")


def paste_3d(dst: torch.Tensor, src: torch.Tensor, bbox):
    """
    dst: [D,H,W]
    src: [d,h,w]
    """
    (z0, z1), (y0, y1), (x0, x1) = bbox
    out = dst.clone()
    out[z0:z1, y0:y1, x0:x1] = src
    return out


def bbox_to_dict(bbox):
    if bbox is None:
        return None
    (z0, z1), (y0, y1), (x0, x1) = bbox
    return {
        "z0": int(z0),
        "z1": int(z1),
        "y0": int(y0),
        "y1": int(y1),
        "x0": int(x0),
        "x1": int(x1),
    }


def build_final_pred_from_liver_tumor(
    liver_mask: torch.Tensor,
    tumor_mask: torch.Tensor,
) -> torch.Tensor:
    """
    output:
      0 bg
      1 liver
      2 tumor
    """
    final_pred = torch.zeros_like(liver_mask, dtype=torch.long)
    final_pred[liver_mask] = 1
    final_pred[tumor_mask] = 2
    return final_pred


def main():
    args = parse_args()
    add_medseg_to_syspath(args.medseg_root)

    from medseg.models.build_model import build_model
    from medseg.utils.ckpt import load_ckpt

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = time.strftime("%m-%d-%H-%M-%S")
    workdir = os.path.join(args.save_dir, f"eval_{timestamp}")
    os.makedirs(workdir, exist_ok=True)

    pred_dir = None
    if args.save_pred_pt:
        pred_dir = os.path.join(workdir, "pred_pt")
        os.makedirs(pred_dir, exist_ok=True)
    vis_dir = None
    if args.save_vis:
        vis_dir = os.path.join(workdir, "vis_png")
        os.makedirs(vis_dir, exist_ok=True)

    all_pt = load_pt_paths(args.preprocessed_root)
    tr, va, te = split_three_ways(
        all_pt,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if args.split == "train":
        pt_paths = tr
    elif args.split == "val":
        pt_paths = va
    elif args.split == "test":
        pt_paths = te
    elif args.split == "all":
        pt_paths = all_pt
    else:
        raise ValueError(f"unsupported split: {args.split}")
    liver_metrics_list: List[float] = []
    tumor_metrics_list: List[float] = []

    if args.n > 0:
        pt_paths = pt_paths[: args.n]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stage1 = build_model(
        args.stage1_model,
        in_channels=1,
        out_channels=2,
        img_size=tuple(args.stage1_patch),
    ).to(device)

    stage2 = build_model(
        args.stage2_model,
        in_channels=1,
        out_channels=2,
        img_size=tuple(args.stage2_patch),
    ).to(device)

    load_ckpt(args.stage1_ckpt, stage1, optimizer=None, map_location=device)
    load_ckpt(args.stage2_ckpt, stage2, optimizer=None, map_location=device)

    stage1.eval()
    stage2.eval()

    print(f"[eval] split={args.split}")
    print(
        f"[eval] val_ratio={args.val_ratio} test_ratio={args.test_ratio} seed={args.seed}"
    )
    print(f"[eval] n_cases={len(pt_paths)}")
    print(f"[eval] device={device}")

    rows: List[Dict] = []
    liver_dices: List[float] = []
    tumor_dices: List[float] = []

    time_start = time.time()

    with torch.no_grad():
        for case_idx, pt_path in enumerate(pt_paths, start=1):
            case_name = safe_case_name(pt_path)
            data = torch.load(
                pt_path, map_location="cpu", weights_only=False, mmap=True
            )

            image = data["image"].float()  # [1,D,H,W]
            label = data.get("label", None)  # [1,D,H,W]

            # -----------------------------
            # stage1: liver segmentation
            # -----------------------------
            x = image.unsqueeze(0).to(device)  # [1,1,D,H,W]

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device == "cuda"),
            ):
                logits1 = sliding_window_inference(
                    inputs=x,
                    roi_size=tuple(args.stage1_patch),
                    sw_batch_size=args.stage1_sw_batch_size,
                    predictor=stage1,
                    overlap=args.overlap,
                )

            if isinstance(logits1, (tuple, list)):
                logits1 = logits1[0]

            logits1 = torch.as_tensor(logits1)
            pred1 = torch.argmax(logits1.float(), dim=1)[0].cpu()  # [D,H,W]
            liver_mask = pred1 == 1

            liver_mask = filter_largest_component(liver_mask)
            pred1_filtered = liver_mask.long()

            # -----------------------------
            # stage2: tumor segmentation in liver ROI
            # -----------------------------
            if liver_mask.sum().item() == 0:
                bbox = None
                tumor_full = torch.zeros_like(pred1, dtype=torch.long)
            else:
                bbox = compute_bbox_from_mask(liver_mask, margin=args.margin)
                image_roi = crop_3d(image, bbox)  # [1,d,h,w]
                x_roi = image_roi.unsqueeze(0).to(device)  # [1,1,d,h,w]

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(device == "cuda"),
                ):
                    logits2 = sliding_window_inference(
                        inputs=x_roi,
                        roi_size=tuple(args.stage2_patch),
                        sw_batch_size=args.stage2_sw_batch_size,
                        predictor=stage2,
                        overlap=args.overlap,
                    )

                if isinstance(logits2, (tuple, list)):
                    logits2 = logits2[0]

                logits2 = torch.as_tensor(logits2)
                pred2 = torch.argmax(logits2.float(), dim=1)[0].cpu()  # [d,h,w]

                tumor_full = torch.zeros_like(pred1, dtype=torch.long)
                tumor_full = paste_3d(tumor_full, pred2.long(), bbox)

            tumor_mask = tumor_full == 1
            tumor_mask = tumor_mask & liver_mask

            # 删除很小的假阳性块
            labeled, num = ndi.label(tumor_mask.cpu().numpy())  # type:ignore

            sizes = ndi.sum(tumor_mask.cpu().numpy(), labeled, range(1, num + 1))

            clean = torch.zeros_like(tumor_mask)

            for comp_idx, s in enumerate(sizes):
                if s > args.min_tumor_size:  # 50 个 voxel 以下认为是噪声
                    clean[labeled == (comp_idx + 1)] = 1

            tumor_mask = clean.bool()

            # final 0/1/2 prediction

            final_pred = build_final_pred_from_liver_tumor(
                liver_mask=liver_mask,
                tumor_mask=tumor_mask,
            )
            pt_path = os.path.basename(pt_path)
            row: Dict = {
                "case_name": case_name,
                "source_pt": pt_path,
                "pred_liver_voxels": int(liver_mask.sum().item()),
                "pred_tumor_voxels": int(tumor_mask.sum().item()),
                "bbox": bbox_to_dict(bbox),
            }

            if label is not None:
                gt = label[0].long()  # [D,H,W], 0/1/2
                gt_liver = gt > 0
                gt_tumor = gt == 2

                liver_metrics = compute_metrics(final_pred > 0, gt_liver)
                tumor_metrics = compute_metrics(final_pred == 2, gt_tumor)

                row["liver_dice"] = round(liver_metrics["Dice"], 4)
                row["tumor_dice"] = round(tumor_metrics["Dice"], 4)
                row["tumor_jaccard"] = round(tumor_metrics["Jaccard"], 4)
                row["tumor_recall"] = round(tumor_metrics["Recall"], 4)
                row["tumor_FDR"] = round(tumor_metrics["FDR"], 4)
                row["tumor_precision"] = round(tumor_metrics["Precision"], 4)

                liver_metrics_list.append(liver_metrics)
                tumor_metrics_list.append(tumor_metrics)

            rows.append(row)
            if vis_dir is not None and case_idx <= args.vis_n:
                save_case_visualization(
                    save_path=os.path.join(vis_dir, f"{case_name}.png"),
                    image=image,
                    label=label,
                    pred1=pred1_filtered,
                    tumor_full=tumor_mask.long(),
                    final_pred=final_pred,
                    case_name=case_name,
                )

            if pred_dir is not None:
                torch.save(
                    {
                        "image": image,
                        "label": label,
                        "stage1_liver_pred": pred1.unsqueeze(0).long(),
                        "stage2_tumor_pred": tumor_full.unsqueeze(0).long(),
                        "final_pred": final_pred.unsqueeze(0).long(),
                        "meta": row,
                    },
                    os.path.join(pred_dir, f"{case_name}_twostage.pt"),
                )

            msg = f"[{case_idx}/{len(pt_paths)}] {case_name}"
            if "liver_dice" in row:
                msg += f" liver={row['liver_dice']:.4f} tumor={row['tumor_dice']:.4f}"
            print(msg)

    elapsed_hours = (time.time() - time_start) / 3600.0

    # -----------------------------
    # save per-case csv
    # -----------------------------
    csv_path = os.path.join(workdir, "per_case.csv")
    fieldnames = [
        "case_name",
        "source_pt",
        "pred_liver_voxels",
        "pred_tumor_voxels",
        "liver_dice",
        "tumor_dice",
        "bbox",
        "tumor_jaccard",
        "tumor_recall",
        "tumor_FDR",
        "tumor_precision",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            rr = dict(r)
            rr["bbox"] = (
                json.dumps(rr["bbox"], ensure_ascii=False)
                if rr["bbox"] is not None
                else ""
            )
            writer.writerow({k: rr.get(k, "") for k in fieldnames})

    # -----------------------------
    # metrics
    # -----------------------------
    metrics = {
        "split": args.split,
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "n_cases": len(rows),
        "device": device,
        "elapsed_hours": round(elapsed_hours, 3),
        "liver": summarize_metrics_list(liver_metrics_list, ["Dice"]),
        "tumor": summarize_metrics_list(
            tumor_metrics_list, ["Dice", "Jaccard", "Recall", "FDR", "FNR", "Precision"]
        ),
    }

    with open(os.path.join(workdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(workdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("Two-Stage Evaluation Report\n")
        f.write("===========================\n")
        f.write(f"workdir: {workdir}\n")
        f.write(f"split: {metrics['split']}\n")
        f.write(f"seed: {metrics['seed']}\n")
        f.write(f"val_ratio: {metrics['val_ratio']}\n")
        f.write(f"test_ratio: {metrics['test_ratio']}\n")
        f.write(f"n_cases: {metrics['n_cases']}\n")
        f.write(f"device: {metrics['device']}\n")
        f.write(f"elapsed_hours: {metrics['elapsed_hours']}\n\n")
        for organ, organ_key in [("Liver", "liver"), ("Tumor", "tumor")]:
            f.write(f"{organ}\n")
            for metric_name, summary in metrics[organ_key].items():
                f.write(f"{metric_name}\n")
                f.write(f"  mean: {summary['mean']}\n")
                f.write(f"   std: {summary['std']}\n")
                f.write(f"   min: {summary['min']}\n")
                f.write(f"   max: {summary['max']}\n")
            f.write("\n")

    print("\n===== Final Metrics =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {workdir}")


if __name__ == "__main__":
    main()
