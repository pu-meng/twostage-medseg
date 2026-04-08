import torch
from typing import Dict, List


def dice_binary(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    pred.bool()也是类似len(x),都是根据pred的类型调用它的.bool()方法
    len(x)==x.__len__(),这里的__len__()是x的类型的__len__()方法
    
    """
    pred = pred.bool()
    gt = gt.bool()

    inter = (pred & gt).sum().item()
    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()

    # gt 和 pred 都为空：不返回 1.0，返回 nan，由调用方决定是否纳入统计
    # 对应 nnUNet 做法：无肿瘤 case 不参与 tumor dice 均值计算
    if pred_sum == 0 and gt_sum == 0:
        return float("nan")
    return 2.0 * inter / (pred_sum + gt_sum + 1e-8)


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
    """
    nan!=nan是true,nan不等于任何值包括自身
    """
    # 过滤 nan（无肿瘤 case 的 tumor dice 不参与统计，对齐 nnUNet）
    xs = [v for v in xs if not (v != v)]  # 这里的xs只统计不是nan的值
    if len(xs) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }

    x = torch.tensor(xs, dtype=torch.float32)
    return {
        "mean": round(float(x.mean().item()), 4),
        "std": round(float(x.std(unbiased=False).item()), 4),
        "min": round(float(x.min().item()), 4),
        "max": round(float(x.max().item()), 4),
        "n": len(xs),  # 实际参与统计的 case 数，方便核对
    }


def summarize_metrics_list(metrics_list: List[Dict], keys: List[str]) -> Dict:
    return {k: summarize_metric([m[k] for m in metrics_list]) for k in keys}
