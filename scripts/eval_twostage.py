# eval_twostage_simple.py
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import scipy.ndimage as ndi

import sys
import time
from typing import Dict, List

import torch
from monai.inferers.utils import sliding_window_inference
from medseg.models.build_model import build_model
from medseg.utils.ckpt import load_ckpt

from medseg.data.dataset_offline import split_three_ways

# twostage_medseg/metrics/filter.py
from metrics.filter import filter_largest_component
from metrics.metrics_utils import compute_metrics, summarize_metrics_list
from twostage.vis_utils import save_case_visualization


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
    p.add_argument("--stage2_ckpt_b", type=str, default=None,
                   help="fine-tune 模型路径，有则与 stage2_ckpt 做 ensemble")
    p.add_argument("--ensemble_weight_b", type=float, default=0.5,
                   help="Model B 的融合权重，Model A = 1 - weight_b")

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
    """
    类似pt_path="/home/.../case_1.pt", 返回"case_1"
    """
    name = os.path.basename(pt_path)
    if name.endswith(".pt"):
        name = name[:-3]
    return name


def load_pt_paths(preprocessed_root: str) -> List[str]:
    pt_paths = sorted(glob.glob(os.path.join(preprocessed_root, "*.pt")))
    if len(pt_paths) == 0:
        raise FileNotFoundError(f"no .pt found in {preprocessed_root}")
    return pt_paths


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

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = time.strftime("%m-%d-%H-%M-%S")
    workdir = os.path.join(args.save_dir, f"{timestamp}")
    os.makedirs(workdir, exist_ok=True)

    with open(os.path.join(workdir, "command.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")

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
    # 这里的决定的是在什么测试这个指标,如果不传args.split,就默认是test
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
        print(
            f"[eval] --n={args.n}: 截断为 {len(pt_paths)} -> {min(args.n, len(pt_paths))} cases"
        )
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

    stage2_b = None
    if args.stage2_ckpt_b is not None:
        stage2_b = build_model(
            args.stage2_model,
            in_channels=1,
            out_channels=2,
            img_size=tuple(args.stage2_patch),
        ).to(device)
        load_ckpt(args.stage2_ckpt_b, stage2_b, optimizer=None, map_location=device)
        stage2_b.eval()
        print(f"[eval] ensemble: weight_A={1-args.ensemble_weight_b:.2f}  weight_B={args.ensemble_weight_b:.2f}")

    stage1.eval()
    stage2.eval()

    print(f"[eval] split={args.split}")
    print(
        f"[eval] val_ratio={args.val_ratio} test_ratio={args.test_ratio} seed={args.seed}"
    )
    print(f"[eval] n_cases={len(pt_paths)}")
    print(f"[eval] device={device}")

    rows: List[Dict] = []

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
                # sliding_window_inference是MONAI官方提供的滑动窗口推理函数,专门用于处理医学图像中无法整个放入GPU的情况
                logits1 = sliding_window_inference(
                    inputs=x,  # x:[B,C,D,H,W]
                    roi_size=tuple(
                        args.stage1_patch
                    ),  # 每个滑动窗口(patch)的大小,比如能放入GPU
                    sw_batch_size=args.stage1_sw_batch_size,  # 每次并行处理几个batch,越大越占用显存
                    predictor=stage1,  # 模型本身
                    overlap=args.overlap,  # 重叠越大精度越高,但是越慢
                )
                # 输入是[B,C,D,H,W],输出是[B,C,D,H,W]

            if isinstance(logits1, (tuple, list)):
                logits1 = logits1[0]

            logits1 = torch.as_tensor(logits1)#防御性代码,确保logits1是一个tensor
            #logits1.float():[B,C,D,H,W]
            pred1 = torch.argmax(logits1.float(), dim=1)[0].cpu()  # [D,H,W]
            liver_mask = pred1 == 1
            #pred1:[B,D,H,W]

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
                    if stage2_b is not None:
                        logits2_b = sliding_window_inference(
                            inputs=x_roi,
                            roi_size=tuple(args.stage2_patch),
                            sw_batch_size=args.stage2_sw_batch_size,
                            predictor=stage2_b,
                            overlap=args.overlap,
                        )

                if isinstance(logits2, (tuple, list)):
                    logits2 = logits2[0]
                logits2 = torch.as_tensor(logits2).float()

                if stage2_b is not None:
                    if isinstance(logits2_b, (tuple, list)):
                        logits2_b = logits2_b[0]
                    logits2_b = torch.as_tensor(logits2_b).float()
                    w = args.ensemble_weight_b
                    logits2 = (1 - w) * logits2 + w * logits2_b

                pred2 = torch.argmax(logits2, dim=1)[0].cpu()  # [d,h,w]

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
                # pred_dir由args.pred_dir指定,如果args.pred_dir为空,则pred_dir为None
                # 这个是保存预测结果的,每个case一个.pt文件,这个是给程序保存的,后续可以加载进行分析,调试
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
