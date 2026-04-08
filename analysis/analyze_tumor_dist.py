"""
分析训练集每个 case 的肿瘤体积分布，输出统计信息，帮助确定差异化 oversampling 的阈值。
用法:
    python scripts/analyze_tumor_dist.py \
        --medseg_root /home/pumengyu/medseg_project \
        --preprocessed_root /home/pumengyu/Task03_Liver_pt \
        --val_ratio 0.2 --test_ratio 0.1 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import torch


def add_medseg_to_syspath(medseg_root: str) -> None:
    medseg_root = os.path.abspath(medseg_root)
    if medseg_root not in sys.path:
        sys.path.insert(0, medseg_root)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--medseg_root", default="/home/pumengyu/medseg_project")
    p.add_argument("--preprocessed_root", default="/home/pumengyu/Task03_Liver_pt")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    # 小肿瘤阈值（voxels），低于此视为小肿瘤
    p.add_argument("--small_thresh", type=int, default=500)
    return p.parse_args()


def load_label(pt_path: str) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        data = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
    return data["label"]  # [1, D, H, W]


def main():
    args = parse_args()
    add_medseg_to_syspath(args.medseg_root)

    from medseg.data.dataset_offline import load_pt_paths, split_three_ways

    all_paths = load_pt_paths(args.preprocessed_root)
    train_paths, val_paths, test_paths = split_three_ways(
        all_paths,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}\n")
    print(f"小肿瘤阈值: < {args.small_thresh} voxels\n")

    results = []
    for pt_path in sorted(train_paths):
        label = load_label(pt_path)
        tumor_voxels = int((label == 2).sum().item())
        results.append((os.path.basename(pt_path), tumor_voxels))

    results.sort(key=lambda x: x[1])

    no_tumor = [(n, v) for n, v in results if v == 0]
    small_tumor = [(n, v) for n, v in results if 0 < v < args.small_thresh]
    normal_tumor = [(n, v) for n, v in results if v >= args.small_thresh]

    print(f"{'='*60}")
    print(f"无肿瘤 case ({len(no_tumor)} 个):")
    for name, vox in no_tumor:
        print(f"  {name:30s}  tumor_voxels = {vox}")

    print(f"\n小肿瘤 case ({len(small_tumor)} 个, 0 < voxels < {args.small_thresh}):")
    for name, vox in small_tumor:
        print(f"  {name:30s}  tumor_voxels = {vox}")

    print(f"\n正常肿瘤 case ({len(normal_tumor)} 个, voxels >= {args.small_thresh}):")
    for name, vox in normal_tumor:
        print(f"  {name:30s}  tumor_voxels = {vox}")

    print(f"\n{'='*60}")
    print("统计摘要:")
    print(f"  无肿瘤:   {len(no_tumor):3d} / {len(results)} ({100*len(no_tumor)/len(results):.1f}%)")
    print(f"  小肿瘤:   {len(small_tumor):3d} / {len(results)} ({100*len(small_tumor)/len(results):.1f}%)")
    print(f"  正常肿瘤: {len(normal_tumor):3d} / {len(results)} ({100*len(normal_tumor)/len(results):.1f}%)")

    if normal_tumor:
        vox_list = [v for _, v in normal_tumor]
        print(f"\n  正常肿瘤体积 (voxels): min={min(vox_list)}, max={max(vox_list)}, "
              f"median={sorted(vox_list)[len(vox_list)//2]}")

    # 打印分位数帮助确定阈值
    all_nonzero = sorted([v for _, v in results if v > 0])
    if all_nonzero:
        n = len(all_nonzero)
        print("\n  有肿瘤 case 体积分位数:")
        for pct in [10, 25, 50, 75, 90]:
            idx = max(0, int(n * pct / 100) - 1)
            print(f"    P{pct:2d}: {all_nonzero[idx]:>8d} voxels")


if __name__ == "__main__":
    main()
