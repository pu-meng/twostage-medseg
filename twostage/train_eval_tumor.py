from __future__ import annotations

from typing import Dict


def tumor_metrics_from_val_result(val_result: Dict) -> Dict[str, float]:
    """
    兼容 medseg.engine.train_eval.validate_sliding_window 的返回格式。
    stage2 是二分类：
        class_ids = [1]
        per_class = [tumor_dice]
        mean_fg = tumor_dice
    """
    mean_fg = float(val_result["mean_fg"])
    per_class = val_result.get("per_class", [])
    tumor_dice = float(per_class[0]) if len(per_class) > 0 else mean_fg
    return {
        "tumor_dice": tumor_dice,
        "mean_fg": mean_fg,
    }
