import torch
from typing import List, Dict

def build_final_pred_from_liver_tumor(
    liver_mask: torch.Tensor,
    tumor_mask: torch.Tensor,
) -> torch.Tensor:
    """
    输入:
        liver_mask: [D,H,W] bool
        tumor_mask: [D,H,W] bool

    输出:
        final_pred: [D,H,W] long
            0 background
            1 liver
            2 tumor

    逻辑:
        先把 liver 区域设为 1
        再把 tumor 区域设为 2
        所以 tumor 会覆盖 liver
    """
    final_pred = torch.zeros_like(liver_mask, dtype=torch.long)
    final_pred[liver_mask] = 1
    final_pred[tumor_mask] = 2
    return final_pred


def summarize_metric(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    x = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": round(float(x.mean().item()), 4),
        "std": round(float(x.std(unbiased=False).item()), 4),
        "min": round(float(x.min().item()), 4),
        "max": round(float(x.max().item()), 4),
    }