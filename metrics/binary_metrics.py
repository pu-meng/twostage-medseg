from __future__ import annotations

from typing import Dict, List

import torch


def binary_stats(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, int]:
    """
    pred, gt: 同shape, 二值张量/可转bool
    返回 TP/FP/FN/TN
    """
    pred = pred.bool()
    gt = gt.bool()

    tp = int((pred & gt).sum().item())
    fp = int((pred & (~gt)).sum().item())
    fn = int(((~pred) & gt).sum().item())
    tn = int(((~pred) & (~gt)).sum().item())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def dice_binary(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.bool()
    gt = gt.bool()

    inter = (pred & gt).sum().item()
    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    return 2.0 * inter / (pred_sum + gt_sum + 1e-8)


def precision_binary(pred: torch.Tensor, gt: torch.Tensor) -> float:
    s = binary_stats(pred, gt)
    tp, fp = s["tp"], s["fp"]

    if tp + fp == 0:
        # 没预测任何前景:
        # 若 GT 也没有前景, 可视作 precision=1
        # 若 GT 有前景, precision=0
        return 1.0 if gt.bool().sum().item() == 0 else 0.0

    return tp / (tp + fp + 1e-8)


def recall_binary(pred: torch.Tensor, gt: torch.Tensor) -> float:
    s = binary_stats(pred, gt)
    tp, fn = s["tp"], s["fn"]

    if tp + fn == 0:
        # GT 没有前景
        return 1.0

    return tp / (tp + fn + 1e-8)


def iou_binary(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.bool()
    gt = gt.bool()

    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()

    if union == 0:
        return 1.0

    return inter / (union + 1e-8)


def detected_binary(pred: torch.Tensor, gt: torch.Tensor) -> int:
    """
    病例级检出:
    - GT有前景 且 预测也有前景 => 1
    - 其他 => 0
    """
    pred_has = bool(pred.bool().sum().item() > 0)
    gt_has = bool(gt.bool().sum().item() > 0)
    return int(pred_has and gt_has)


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
