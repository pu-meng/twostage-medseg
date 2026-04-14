"""
plan_patch_size.py
==================
仿照 nnUNet 的 ExperimentPlanner 逻辑，根据数据集统计信息自动规划 patch size。

核心逻辑：
  1. 读取所有 .pt 文件，统计每个 case 的空间尺寸 (D, H, W)
  2. 取 median shape 作为基准，按轴比例缩放候选 patch
  3. 每轴对齐到 align 的倍数（保证能被 2^depth 整除，默认 32）
  4. 实测：对每个候选 patch 跑一次真实 forward+backward，测峰值显存
  5. 二分搜索：找到在显存预算内的最大 patch

估算 vs 实测：
  原版用公式估算显存，误差大（没算 deep supervision/cuDNN workspace/优化器状态）。
  改为直接实测，结果准确。

使用方式：
  python scripts/plan_patch_size.py
  python scripts/plan_patch_size.py --batch_size 2 --amp --model dynunet_deep
  python scripts/plan_patch_size.py --medseg_root /path/to/medseg_project --batch_size 1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# 数据集统计
# ──────────────────────────────────────────────────────────────────────────────

def collect_shapes(preprocessed_root: str) -> np.ndarray:
    pts = sorted(glob.glob(f"{preprocessed_root}/*.pt"))
    if not pts:
        raise FileNotFoundError(f"no .pt found in {preprocessed_root}")
    shapes = []
    print(f"scanning {len(pts)} cases ...")
    for p in pts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        shapes.append(tuple(d["image"].shape[1:]))  # (D, H, W)
    return np.array(shapes, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 实测显存
# ──────────────────────────────────────────────────────────────────────────────

def measure_vram(
    patch: tuple[int, int, int],
    batch_size: int,
    model_name: str,
    amp: bool,
    in_channels: int,
    medseg_root: str,
) -> tuple[float | None, str]:
    """
    真实跑一次 forward + backward，返回峰值显存(GB)。
    OOM 时返回 (None, 'OOM')。
    """
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        from medseg.models.build_model import build_model
        model = build_model(model_name, in_channels=in_channels, out_channels=2).to(device)

        x = torch.randn(batch_size, in_channels, *patch, device=device,
                        dtype=torch.float16 if amp else torch.float32)

        with torch.cuda.amp.autocast(enabled=amp):
            y = model(x)
            # deep supervision: y 可能是 list/tuple
            if isinstance(y, (list, tuple)):
                loss = sum(yi.mean() for yi in y)
            else:
                loss = y.mean()

        loss.backward()
        mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        del model, x, y, loss
        torch.cuda.empty_cache()
        return mem_gb, "OK"

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None, "OOM"
        raise


# ──────────────────────────────────────────────────────────────────────────────
# patch size 候选生成
# ──────────────────────────────────────────────────────────────────────────────

def round_to_multiple(x: float, base: int) -> int:
    return max(base, int(x // base) * base)


def make_candidate(scale: float, ratio: np.ndarray, align: int, max_per_axis: int) -> tuple[int, int, int]:
    return tuple(
        min(round_to_multiple(scale * r, align), max_per_axis)
        for r in ratio
    )  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# 主规划逻辑：实测二分搜索
# ──────────────────────────────────────────────────────────────────────────────

def plan_patch(
    median_shape: np.ndarray,
    batch_size: int,
    model_name: str,
    amp: bool,
    in_channels: int,
    medseg_root: str,
    safety_factor: float,
    align: int,
    min_scale: int,
    max_scale: int,
) -> tuple[tuple[int, int, int], float]:
    """
    按 median shape 的轴比例，从小到大逐步实测，找到不 OOM 的最大 patch。
    返回 (best_patch, peak_vram_gb)。
    """
    ratio = median_shape / median_shape.max()

    # 获取 GPU 总显存
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    budget_gb = total_gb * safety_factor
    print(f"\nGPU total: {total_gb:.1f} GB  budget({safety_factor:.0%}): {budget_gb:.1f} GB\n")

    # 生成候选 scale 列表（等差，步长 align）
    scales = list(range(min_scale, max_scale + 1, align))

    best_patch = make_candidate(scales[0], ratio, align, max_scale)
    best_mem = 0.0

    print(f"{'patch':<25} {'VRAM(GB)':<12} status")
    print("-" * 45)

    for s in scales:
        candidate = make_candidate(s, ratio, align, max_scale)
        mem_gb, status = measure_vram(candidate, batch_size, model_name, amp, in_channels, medseg_root)

        if mem_gb is not None:
            mem_str = f"{mem_gb:.2f}"
            fits = mem_gb <= budget_gb
            flag = "" if fits else " (over budget)"
            print(f"{str(list(candidate)):<25} {mem_str:<12} {status}{flag}")
            if fits:
                best_patch = candidate
                best_mem = mem_gb
        else:
            print(f"{str(list(candidate)):<25} {'- ':<12} {status}")
            # OOM → 不再往大测
            break

    return best_patch, best_mem


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Auto plan patch size (real VRAM benchmark)")
    p.add_argument("--preprocessed_root", type=str,
                   default="/home/PuMengYu/Task03_Liver_roi")
    p.add_argument("--medseg_root", type=str,
                   default="/home/PuMengYu/medseg_project")
    p.add_argument("--model", type=str, default="dynunet_deep")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--safety_factor", type=float, default=0.85,
                   help="可用显存比例，建议 0.80~0.90，留出 dataloader/推理缓存开销")
    p.add_argument("--align", type=int, default=32,
                   help="patch 各轴对齐到此数的倍数（保证被 2^depth 整除）")
    p.add_argument("--min_scale", type=int, default=64,
                   help="搜索起点（等效边长）")
    p.add_argument("--max_scale", type=int, default=320,
                   help="搜索终点（等效边长）")
    p.add_argument("--isotropic", action="store_true", default=False,
                   help="强制使用立方体 patch（各轴相同），忽略轴比例")
    return p.parse_args()


def main():
    args = parse_args()

    # 把 medseg_root 加入 sys.path
    medseg_root = os.path.abspath(args.medseg_root)
    if medseg_root not in sys.path:
        sys.path.insert(0, medseg_root)

    # 屏蔽无关 warning
    warnings.filterwarnings("ignore")

    # ── 数据集统计 ──────────────────────────────────────────────
    shapes = collect_shapes(args.preprocessed_root)
    median_shape = np.median(shapes, axis=0)

    print(f"\n{'='*55}")
    print(f"  Dataset shape statistics (D, H, W)")
    print(f"{'='*55}")
    print(f"  n        : {len(shapes)}")
    print(f"  median   : {median_shape.astype(int)}")
    print(f"  mean     : {np.mean(shapes,axis=0).astype(int)}")
    print(f"  min      : {shapes.min(axis=0).astype(int)}")
    print(f"  max      : {shapes.max(axis=0).astype(int)}")
    print(f"  25th pct : {np.percentile(shapes,25,axis=0).astype(int)}")
    print(f"  75th pct : {np.percentile(shapes,75,axis=0).astype(int)}")
    print(f"\n  model      : {args.model}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  amp        : {args.amp}")
    print(f"  in_channels: {args.in_channels}")

    # ── 实测搜索 ────────────────────────────────────────────────
    # isotropic 模式：强制立方体，ratio 全为 1
    if args.isotropic:
        search_shape = np.array([1.0, 1.0, 1.0])
        print(f"\n  mode: isotropic (cubic patch)")
    else:
        search_shape = median_shape
        print(f"\n  mode: anisotropic (scaled by median shape ratio)")

    best_patch, best_mem = plan_patch(
        median_shape=search_shape,
        batch_size=args.batch_size,
        model_name=args.model,
        amp=args.amp,
        in_channels=args.in_channels,
        medseg_root=medseg_root,
        safety_factor=args.safety_factor,
        align=args.align,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )

    # ── 输出结论 ────────────────────────────────────────────────
    coverage = [best_patch[i] / median_shape[i] for i in range(3)]
    min_shape = shapes.min(axis=0).astype(int)
    fit_all = all(best_patch[i] <= min_shape[i] for i in range(3))

    print(f"\n{'='*55}")
    print(f"  Recommended patch size")
    print(f"{'='*55}")
    print(f"  patch      : {list(best_patch)}")
    print(f"  peak VRAM  : {best_mem:.2f} GB")
    print(f"  coverage   : D={coverage[0]:.0%}  H={coverage[1]:.0%}  W={coverage[2]:.0%}  (vs median)")
    print(f"  fits min   : {'YES' if fit_all else 'NO (some cases smaller, monai pads automatically)'}")
    print(f"\n  Add to config.yaml:")
    print(f"    patch: {list(best_patch)}")
    print(f"    val_patch: {list(best_patch)}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
