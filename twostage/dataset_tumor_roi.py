from __future__ import annotations

import warnings
from pathlib import Path
from metrics.filter import filter_largest_component
from typing import Any
from typing import Sequence
import torch
from torch.utils.data import Dataset

from .roi_utils import compute_bbox_from_mask, crop_3d, bbox_to_dict
import random

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

    差异化 oversampling（hard case mining）:
    - 无肿瘤 case（tumor_voxels == 0）: repeat × no_tumor_repeat_scale
    - 小肿瘤 case（0 < tumor_voxels < small_tumor_thresh）: repeat × small_tumor_repeat_scale
    - 正常 case: repeat × 1
    通过在 __init__ 时预扫描 label 统计每个 case 的肿瘤体素数来分类。
    """

    def __init__(
        self,
        pt_paths: Sequence[str],
        transform=None,
        repeats: int = 1,
        margin: int = 12,
        keep_meta: bool = True,
        bbox_jitter: bool = False,
        bbox_max_shift: int = 8,
        random_margin: bool = False,
        margin_min: int = 8,
        margin_max: int = 20,
        small_tumor_thresh: int = 0,
        small_tumor_repeat_scale: int = 1,
        no_tumor_repeat_scale: int = 1,
    ) -> None:
        self.pt_paths = [str(p) for p in pt_paths]
        self.transform = transform
        self.repeats = int(repeats)
        self.margin = int(margin)
        self.keep_meta = bool(keep_meta)
        self.bbox_jitter = bool(bbox_jitter)
        self.bbox_max_shift = int(bbox_max_shift)
        self.random_margin = bool(random_margin)
        self.margin_min = int(margin_min)
        self.margin_max = int(margin_max)
        self.small_tumor_thresh = int(small_tumor_thresh)
        self.small_tumor_repeat_scale = int(small_tumor_repeat_scale)
        self.no_tumor_repeat_scale = int(no_tumor_repeat_scale)

        if len(self.pt_paths) == 0:
            raise ValueError("pt_paths 是空的")
        if self.repeats <= 0:
            raise ValueError("repeats 必须 >= 1")
        if self.bbox_max_shift < 0:
            raise ValueError("bbox_max_shift 必须>= 0")
        if self.margin_min < 0 or self.margin_max < 0:
            raise ValueError("margin_min/margin_max 必须 >= 0")
        if self.margin_min > self.margin_max:
            raise ValueError("margin_min 必须 <= margin_max")

        # 构建差异化 index 列表
        self._indices = self._build_indices()

    def _count_tumor_voxels(self, pt_path: str) -> int:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
        return int((data["label"] == 2).sum().item())

    def _build_indices(self) -> list[int]:
        """
        根据每个 case 的肿瘤体素数分配 repeat 次数，返回展开后的 case index 列表。
        只有当 small_tumor_thresh > 0 或 no/small_tumor_repeat_scale > 1 时才扫描。
        """
        use_hard_mining = (
            self.small_tumor_thresh > 0
            and (self.small_tumor_repeat_scale > 1 or self.no_tumor_repeat_scale > 1)
        )
        if not use_hard_mining:
            return list(range(len(self.pt_paths))) * self.repeats

        indices = []
        n_no_tumor = 0
        n_small_tumor = 0
        n_normal = 0
        for i, pt_path in enumerate(self.pt_paths):
            voxels = self._count_tumor_voxels(pt_path)
            if voxels == 0:
                r = self.repeats * self.no_tumor_repeat_scale
                n_no_tumor += 1
            elif voxels < self.small_tumor_thresh:
                r = self.repeats * self.small_tumor_repeat_scale
                n_small_tumor += 1
            else:
                r = self.repeats
                n_normal += 1
            indices.extend([i] * r)
        print(
            f"[TumorROIDataset] hard mining: "
            f"no_tumor={n_no_tumor}×{self.no_tumor_repeat_scale}, "
            f"small_tumor(<{self.small_tumor_thresh})={n_small_tumor}×{self.small_tumor_repeat_scale}, "
            f"normal={n_normal}×1 | "
            f"total indices={len(indices)}"
        )
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def _sample_margin(self) -> int:
        """
        如果 random_margin=True,则随机生成margin_min和margin_max之间的整数
        否则,返回固定的self.margin
        """
        if self.random_margin:
            return random.randint(self.margin_min, self.margin_max)
        # random.randint(a,b)返回[a,b]之间的随机整数
        return self.margin

    def _jitter_bbox(
        self,
        bbox: tuple[int, int, int, int, int, int],
        spatial_shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int, int, int]:
        """
        对 bbox 六个边界做独立随机扰动，然后裁回合法范围。
        spatial_shape: (D, H, W)
        """
        if not self.bbox_jitter or self.bbox_max_shift <= 0:
            return bbox

        z0, z1, y0, y1, x0, x1 = bbox
        D, H, W = spatial_shape
        s = self.bbox_max_shift

        z0 += random.randint(-s, s)
        z1 += random.randint(-s, s)
        y0 += random.randint(-s, s)
        y1 += random.randint(-s, s)
        x0 += random.randint(-s, s)
        x1 += random.randint(-s, s)

        # 裁回合法范围
        # z0,yo,x0等这些都要比大于等于0,小于等于D-1,H-1,W-1
        # z1,y1,x1等这些都要比大于等于z0+1,y0+1,x0+1,小于等于D,H,W
        z0 = max(0, min(z0, D - 1))
        z1 = max(z0 + 1, min(z1, D))

        y0 = max(0, min(y0, H - 1))
        y1 = max(y0 + 1, min(y1, H))

        x0 = max(0, min(x0, W - 1))
        x1 = max(x0 + 1, min(x1, W))

        return (z0, z1, y0, y1, x0, x1)

    def __getitem__(self, idx: int):
        case_idx = self._indices[idx]
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
            raise TypeError(
                f"expect dict from torch.load, got {type(data)} @ {pt_path}"
            )
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

        liver_mask = label[0] > 0

        liver_mask = filter_largest_component(liver_mask)

        # 训练时可随机 margin，模拟 stage1 预测框尺度波动
        cur_margin = self._sample_margin()
#这里的margin是上下左右前后 都加的margin,大小一样
        bbox = compute_bbox_from_mask(liver_mask, margin=cur_margin)

        # 训练时可对 bbox 做扰动，模拟 stage1 预测框位置误差
        bbox = self._jitter_bbox(bbox, spatial_shape=tuple(label.shape[1:]))
#这个的变动是对z,y,x三个维度都做了变动,但是z,y,x的变动是独立的,即z的变动和y,x的变动是独立的
#这个变动必须小于self.bbox_max_shift,否则会报错
        image_roi = crop_3d(image, bbox)
        label_roi = crop_3d(label, bbox)

        tumor_roi = (label_roi == 2).long()

        out: dict[str, Any] = {
            "image": image_roi.float(),
            "label": tumor_roi.long(),
        }

        if self.keep_meta:
            out["case_name"] = Path(pt_path).stem
            out["bbox"] = bbox_to_dict(bbox)
            out["source_pt"] = pt_path
            out["roi_shape"] = list(image_roi.shape)
            out["margin"] = int(cur_margin)

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
