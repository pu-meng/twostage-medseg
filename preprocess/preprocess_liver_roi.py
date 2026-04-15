"""
preprocess_liver_roi.py
=======================
一次性预处理：把每个完整 CT .pt 文件裁成肝脏 ROI,保存为小文件。

原文件: Task03_Liver_pt/{case}.pt   540MB ~ 1.9GB  全身 CT
输出:   Task03_Liver_roi/{case}.pt  ~100-200MB     肝脏 ROI crop

裁剪策略:
  - 有 pred_bbox_cache: 用 Stage1 预测 bbox（与推理对齐）
  - 否则: 用 GT liver mask 计算 tight bbox
  - 额外保留 margin_extra voxels 上下左右（推荐用 margin_max，这里默认 30）
  - 同时保存 crop 坐标，供训练时 random_margin 在更小范围内抖动

用法:
  python scripts/preprocess_liver_roi.py \
    --input_dir /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_pt \
    --output_dir /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_roi \
    --pred_bbox_cache /home/PuMengYu/MSD_LiverTumorSeg/Task03_Liver_json/pred_bbox_stage1.json \
    --margin_extra 30
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
from tqdm import tqdm


def compute_gt_bbox(label: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    """从 GT label 计算肝脏 tight bbox（忽略 margin）。"""
    mask = label[0] > 0  # [D, H, W]
    nz = mask.nonzero()
    if len(nz) == 0:
        D, H, W = label.shape[1], label.shape[2], label.shape[3]
        return (0, D, 0, H, 0, W)
    z0, z1 = nz[:, 0].min().item(), nz[:, 0].max().item() + 1
    y0, y1 = nz[:, 1].min().item(), nz[:, 1].max().item() + 1
    x0, x1 = nz[:, 2].min().item(), nz[:, 2].max().item() + 1
    return (int(z0), int(z1), int(y0), int(y1), int(x0), int(x1))


def expand_bbox(
    bbox: tuple[int, int, int, int, int, int],
    margin: int,
    shape: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox
    D, H, W = shape
    z0 = max(0, z0 - margin)
    z1 = min(D, z1 + margin)
    y0 = max(0, y0 - margin)
    y1 = min(H, y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(W, x1 + margin)
    return (z0, z1, y0, y1, x0, x1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pred_bbox_cache", default=None, help="Stage1 pred bbox json")
    p.add_argument(
        "--margin_extra",
        type=int,
        default=30,
        help="tight bbox 外扩 voxels(应 >= 训练时 margin_max)",
    )
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_bboxes = {}
    if args.pred_bbox_cache and Path(args.pred_bbox_cache).exists():
        with open(args.pred_bbox_cache) as f:
            raw = json.load(f)
        pred_bboxes = {k: tuple(v) for k, v in raw.items()}
        print(f"[pred_bbox] 加载 {len(pred_bboxes)} 条缓存: {args.pred_bbox_cache}")

    pt_files = sorted(input_dir.glob("*.pt"))
    print(f"共 {len(pt_files)} 个文件，margin_extra={args.margin_extra}")

    size_before, size_after = 0, 0

    for pt_path in tqdm(pt_files, desc="裁剪 ROI"):
        out_path = output_dir / pt_path.name
        if out_path.exists():
            continue  # 已处理过，跳过

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(
                pt_path, map_location="cpu", weights_only=False, mmap=True
            )

        image = data["image"]  # [1, D, H, W]
        label = data["label"]  # [1, D, H, W]
        D, H, W = image.shape[1], image.shape[2], image.shape[3]

        case_name = pt_path.stem
        if case_name in pred_bboxes:
            tight_bbox = pred_bboxes[case_name]
        else:
            tight_bbox = compute_gt_bbox(label)

        bbox = expand_bbox(tight_bbox, args.margin_extra, (D, H, W))
        z0, z1, y0, y1, x0, x1 = bbox

        image_roi = image[:, z0:z1, y0:y1, x0:x1].clone()
        label_roi = label[:, z0:z1, y0:y1, x0:x1].clone()

        out_data = {
            "image": image_roi,
            "label": label_roi,
            "crop_bbox": list(bbox),  # (z0,z1,y0,y1,x0,x1) 在原 volume 中的坐标
            "tight_bbox": list(tight_bbox),  # 无 margin 的 tight bbox
            "orig_shape": [D, H, W],
        }
        torch.save(out_data, out_path)

        size_before += pt_path.stat().st_size
        size_after += out_path.stat().st_size

    print("\n完成")
    print(f"  原始: {size_before / 1e9:.1f} GB")
    print(f"  裁后: {size_after / 1e9:.1f} GB")
    print(f"  压缩比: {size_before / max(size_after, 1):.1f}x")
    print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
    main()
