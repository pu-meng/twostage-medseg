from __future__ import annotations

import warnings
from pathlib import Path

from typing import Any
from typing import Sequence
import torch
from torch.utils.data import Dataset

from .roi_utils import compute_bbox_from_mask, crop_3d, bbox_to_dict
"""
.roi_utils表示从当前包目录下导入,也就是相对导入;
data = torch.load(
    pt_path,
    map_location="cpu",
    weights_only=False,
    mmap=True,
)
map_location="cpu"表示将数据加载到cpu上;
因为Dataset/DataLoader阶段通常先在CPU读数据,后面训练时送到GPU
mmap=True,mmsp(memory-mapped file)内存映射读取
意思是:让操作系统按需映射文件内容,什么时候用到,什么时候再读那部分




"""

class TumorROIDataset(Dataset):
    """
    直接读取你已有的离线 .pt 样本：
        {
            "image": [1, D, H, W] float32,
            "label": [1, D, H, W] int64,
        }

    训练/验证时在线处理：
    1) 用 GT liver (label > 0) 得 liver mask
    2) 取 liver bbox
    3) 裁 image / label
    4) 把 label 映射成 tumor 二分类：
          0 = non-tumor
          1 = tumor (label == 2)
    5) 再交给 transform 做 patch crop / augment
    """

    def __init__(
        self,
        pt_paths: Sequence[str],
        transform=None,
        repeats: int = 1,
        margin: int = 12,
        keep_meta: bool = True,
    ) -> None:
        self.pt_paths = [str(p) for p in pt_paths]
        self.transform = transform
        self.repeats = int(repeats)
        self.margin = int(margin)
        self.keep_meta = bool(keep_meta)

        if len(self.pt_paths) == 0:
            raise ValueError("pt_paths is empty")
        if self.repeats <= 0:
            raise ValueError("repeats must be >= 1")

    def __len__(self) -> int:
        return len(self.pt_paths) * self.repeats

    def __getitem__(self, idx: int):
        case_idx = idx % len(self.pt_paths)
        pt_path = self.pt_paths[case_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(
                pt_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )

        if not isinstance(data, dict):
            raise TypeError(f"expect dict from torch.load, got {type(data)} @ {pt_path}")
        if "image" not in data or "label" not in data:
            raise KeyError(f"pt sample missing image/label keys @ {pt_path}")

        image = data["image"]
        label = data["label"]

        if image.ndim != 4 or label.ndim != 4:
            raise ValueError(
                f"expect image/label shape [C,D,H,W], got image={tuple(image.shape)} label={tuple(label.shape)}"
            )
        if image.shape[0] != 1 or label.shape[0] != 1:
            raise ValueError(
                f"expect single-channel image/label, got image={tuple(image.shape)} label={tuple(label.shape)}"
            )

        liver_mask = (label[0] > 0)
        bbox = compute_bbox_from_mask(liver_mask, margin=self.margin)

        image_roi = crop_3d(image, bbox)
        label_roi = crop_3d(label, bbox)
        tumor_roi = (label_roi == 2).long()

        out:dict[str,Any] = {
            "image": image_roi.float(),
            "label": tumor_roi.long(),
        }

        if self.keep_meta:
            out["case_name"] = Path(pt_path).stem
            out["bbox"] = bbox_to_dict(bbox)
            out["source_pt"] = pt_path
            out["roi_shape"] = list(image_roi.shape)

        if self.transform is not None:
            out = self.transform(out)

        if isinstance(out, list):
            if len(out) == 1:
                out = out[0]
            else:
                raise RuntimeError(
                    f"transform returned list with len={len(out)}; please set num_samples=1"
                )
        return out
