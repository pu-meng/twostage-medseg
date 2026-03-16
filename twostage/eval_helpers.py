import torch
from pathlib import Path
import os
import sys
def dice_binary(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    """
    pred, gt: [D,H,W]，会被转成 bool mask 后计算 Dice
    Dice = 2|A∩B| / (|A| + |B|)
    """
    pred = pred.bool()
    gt = gt.bool()
    inter = (pred & gt).sum().double()
    denom = pred.sum().double() + gt.sum().double()
    return float((2.0 * inter + eps) / (denom + eps))


def safe_case_name(pt_path: str) -> str:
    """
    输入:
        /path/to/Case0001_Liver.pt
    返回:
        Case0001_Liver
    """
    name = Path(pt_path).name
    if name.endswith(".pt"):
        name = name[:-3]
    return name


def add_project_to_syspath(project_root: str) -> None:
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
