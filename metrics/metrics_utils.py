import torch
from typing import Dict, List


def dice_binary(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    pred.bool()也是类似len(x),都是根据pred的类型调用它的.bool()方法
    len(x)==x.__len__(),这里的__len__()是x的类型的__len__()方法
    不统计无肿瘤,如果pred和gt都没有肿瘤,则返回nan,调用方可以选择是否纳入统计
    这样做的原因是nnUNet在计算tumor dice时,如果某个case的gt没有肿瘤,则不参与tumor dice的均值计算
    这样可以避免无肿瘤case的tumor dice为1.0,拉高均值,导致对比不公平
    
    """
    pred = pred.bool()
    gt = gt.bool()

    inter = (pred & gt).sum().item()
    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()

    # gt 和 pred 都为空：返回 nan，不参与均值统计（双空 case 不应拉高均值）
    # gt 无肿瘤但 pred 有肿瘤（假阳性）：inter=0，返回 0.0，参与均值统计，惩罚误报
    # gt 有肿瘤：正常计算，无论 pred 是否为空
    if pred_sum == 0 and gt_sum == 0:
        return float("nan")
    return 2.0 * inter / (pred_sum + gt_sum + 1e-8)


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """
    TP:预测为正,实际为正
    TN:y预测为负,实际为负
    FP:预测为正,实际为负
    FN:预测为负,实际为正
    precision:TP/(TP+FP),预测为正的样本中实际为正的比例,也叫查准率
    recall:TP/(TP+FN),实际为正的样本中被正确预测为正的比例,也叫查全率
    FPR:FP/(FP+TN),实际为负的样本中被错误预测为正的比例
    FDR:FP/(FP+TP),预测为正的样本中被错误预测为正的比例
    NPV:TN/(TN+FN),预测为负的样本中实际为负的比例
    ACC:(TP+TN)/(TP+FP+FN+TN),预测正确的样本占总样本的比例
    FNR:FN/(TP+FN),实际为正的样本中被错误预测为负的比例
    """
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
