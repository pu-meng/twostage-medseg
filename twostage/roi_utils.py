from __future__ import annotations

from typing import Dict, Tuple
import torch


def compute_bbox_from_mask(mask: torch.Tensor, margin: int = 0) -> Tuple[int, int, int, int, int, int]:
    """
    mask: [D, H, W] bool/0-1 tensor
    返回: (z0, z1, y0, y1, x0, x1), 右边界为开区间
    
    coords = torch.nonzero(mask > 0, as_tuple=False)
    找出mask中所有非零位置的坐标
    as_tuple=False: 返回一个Tensor, 而不是元组
    比如as_tuple=False
    tensor([
  [z1, y1, x1],
  [z2, y2, x2],
  [z3, y3, x3],
])
    如果as_tuple=True
    (tensor([z1, z2, z3]), tensor([y1, y2, y3]), tensor([x1, x2, x3]))
    
   
   
   """
    if mask.ndim != 3:
        raise ValueError(f"mask ndim must be 3, got {tuple(mask.shape)}")

    coords = torch.nonzero(mask > 0, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("empty mask: cannot compute bbox")

    z0, y0, x0 = coords.min(dim=0).values.tolist()
    z1, y1, x1 = coords.max(dim=0).values.tolist()

    D, H, W = mask.shape
    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)

    z1 = min(D, z1 + 1 + margin)
    y1 = min(H, y1 + 1 + margin)
    x1 = min(W, x1 + 1 + margin)
    return z0, z1, y0, y1, x0, x1


def crop_3d(t: torch.Tensor, bbox: Tuple[int, int, int, int, int, int]) -> torch.Tensor:
    """
    支持 [D,H,W] 或 [C,D,H,W]
    """
    z0, z1, y0, y1, x0, x1 = bbox
    if t.ndim == 3:
        return t[z0:z1, y0:y1, x0:x1]
    if t.ndim == 4:
        return t[:, z0:z1, y0:y1, x0:x1]
    raise ValueError(f"unsupported tensor ndim={t.ndim}, shape={tuple(t.shape)}")


def paste_3d(full: torch.Tensor, patch: torch.Tensor, bbox: Tuple[int, int, int, int, int, int]) -> torch.Tensor:
    """
    full/patch: [D,H,W]
    """
    z0, z1, y0, y1, x0, x1 = bbox
    out = full.clone()
    out[z0:z1, y0:y1, x0:x1] = patch
    return out


def bbox_to_dict(bbox: Tuple[int, int, int, int, int, int]) -> Dict[str, int]:
    z0, z1, y0, y1, x0, x1 = bbox
    return {"z0": z0, "z1": z1, "y0": y0, "y1": y1, "x0": x0, "x1": x1}
