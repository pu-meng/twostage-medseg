from __future__ import annotations

import warnings
from pathlib import Path
from metrics.filter import filter_largest_component
from typing import Any
from typing import Sequence
import torch
from torch.utils.data import Dataset

# 相对导入：从当前包（twostage/）下引入工具函数
#   compute_bbox_from_mask : 根据 liver mask 计算紧致包围框并加 margin
#   crop_3d               : 按 bbox 裁剪 [C,D,H,W] 张量
#   bbox_to_dict          : 将 bbox tuple 转为可序列化的 dict（供 meta 记录）
from .roi_utils import compute_bbox_from_mask, crop_3d, bbox_to_dict
import random


class TumorROIDataset(Dataset):
    """
    Stage 2 肿瘤分割数据集：在线从完整 CT volume 裁出肝脏 ROI，再做肿瘤二分类。

    输入格式（离线预处理好的 .pt 文件）：
        {
            "image": Tensor[1, D, H, W]  float32   # CT 强度
            "label": Tensor[1, D, H, W]  int64     # 0=bg, 1=liver, 2=tumor
        }

    __getitem__ 在线处理流程：
        1. 用 GT label > 0 构造 liver mask，过滤最大连通域去噪
        2. 在 liver mask 上计算 bbox（加随机或固定 margin）
        3. 对 bbox 做随机扰动（bbox_jitter），模拟 Stage 1 预测误差
        4. 按 bbox 裁出 image_roi / label_roi
        5. 将 label 重映射为肿瘤二分类：tumor(2)→1，其余→0
        6. 经 transform 做 patch crop + 数据增强后返回

    差异化 oversampling（hard case mining）：
        通过 __init__ 时预扫描每个 case 的肿瘤体素数，
        让困难样本在 _indices 中出现更多次，DataLoader 每 epoch
        更频繁地采到它们，从而改善小肿瘤/无肿瘤的分割效果：
        - 无肿瘤 case  (voxels == 0)              : repeat × no_tumor_repeat_scale
        - 小肿瘤 case  (0 < voxels < thresh)       : repeat × small_tumor_repeat_scale
        - 正常   case  (voxels >= thresh)           : repeat × 1
    """

    def __init__(
        self,
        pt_paths: Sequence[str],       # 所有 .pt 文件路径列表
        transform=None,                # MONAI transform，包含 patch crop + 增强
        repeats: int = 1,              # 每个 case 每 epoch 被采样的基础次数
        margin: int = 12,              # liver bbox 向外扩展的固定体素数
        keep_meta: bool = True,        # 是否在输出 dict 中附带调试元信息
        bbox_jitter: bool = False,     # 是否对 bbox 做随机扰动
        bbox_max_shift: int = 8,       # bbox 每个边界的最大扰动量（体素）
        random_margin: bool = False,   # 是否随机采样 margin（模拟 ROI 尺度波动）
        margin_min: int = 8,           # random_margin 时的最小 margin
        margin_max: int = 20,          # random_margin 时的最大 margin
        small_tumor_thresh: int = 0,   # 小肿瘤判定阈值（体素数），0 表示关闭 hard mining
        small_tumor_repeat_scale: int = 1,  # 小肿瘤 case 的 repeat 倍率
        no_tumor_repeat_scale: int = 1,     # 无肿瘤 case 的 repeat 倍率
        pred_bboxes: dict | None = None,    # Stage1 预测 tight bbox 字典 {case_name: (z0,z1,y0,y1,x0,x1)}
                                            # 传入时用预测 bbox 代替 GT bbox，消除训练/推理 domain gap
                                            # None 则保持原有 GT bbox + jitter 行为
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
        self.pred_bboxes = pred_bboxes  # {case_name: (z0,z1,y0,y1,x0,x1)} tight bbox，无 margin

        # 基本合法性校验
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

        # 构建展开后的索引列表（hard mining 时困难 case 会出现多次）
        self._indices = self._build_indices()

    def _count_tumor_voxels(self, pt_path: str) -> int:
        """读取单个 .pt 文件，返回 label==2（肿瘤）的体素数，用于 hard mining 分类。"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(
                pt_path, map_location="cpu", weights_only=False, mmap=True
            )
        return int((data["label"] == 2).sum().item())

    def _build_indices(self) -> list[int]:
        """
        构建展开后的 case 索引列表，决定每个 case 每 epoch 被采样几次。

        - 未开启 hard mining（small_tumor_thresh==0 或倍率均为 1）：
          直接将 [0..N-1] 重复 repeats 次，所有 case 等频采样。
        - 开启 hard mining:
          预扫描所有 case 的肿瘤体素数，按大小分三类分别设置 repeat 倍率，
          困难 case(小肿瘤/无肿瘤）在列表中出现更多次，从而被更频繁采到。
        """
        use_hard_mining = self.small_tumor_thresh > 0 and (
            self.small_tumor_repeat_scale > 1 or self.no_tumor_repeat_scale > 1
        )
        if not use_hard_mining:
            # 所有 case 等频：[0,1,...,N-1, 0,1,...,N-1, ...] 共 repeats 轮
            return list(range(len(self.pt_paths))) * self.repeats

        indices = []
        n_no_tumor = 0
        n_small_tumor = 0
        n_normal = 0

        for i, pt_path in enumerate(self.pt_paths):
            voxels = self._count_tumor_voxels(pt_path)
            if voxels == 0:
                # 无肿瘤 case：用更高倍率增加采样频率
                r = self.repeats * self.no_tumor_repeat_scale
                n_no_tumor += 1
            elif voxels < self.small_tumor_thresh:
                # 小肿瘤 case：肿瘤体素数在 (0, thresh) 范围内
                r = self.repeats * self.small_tumor_repeat_scale
                n_small_tumor += 1
            else:
                # 正常 case：肿瘤体素数 >= thresh，不额外加权
                r = self.repeats
                n_normal += 1
            # 将 case i 的下标放入列表 r 次
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
        # 返回展开后的总样本数（DataLoader 据此决定每 epoch 迭代次数）
        return len(self._indices)

    def _sample_margin(self) -> int:
        """
        采样本次裁剪使用的 margin 值。
        - random_margin=True：在 [margin_min, margin_max] 均匀随机采样，
          模拟 Stage 1 预测框尺度的随机波动，提升 Stage 2 鲁棒性。
        - random_margin=False：返回固定的 self.margin。
        """
        if self.random_margin:
            return random.randint(self.margin_min, self.margin_max)
        return self.margin

    def _jitter_bbox(
        self,
        bbox: tuple[int, int, int, int, int, int],
        spatial_shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int, int, int]:
        """
        对 bbox 六个边界做独立随机扰动，模拟 Stage 1 预测框的位置误差。

        Args:
            bbox          : (z0, z1, y0, y1, x0, x1)，由 GT liver mask 算出的精确框
            spatial_shape : (D, H, W)，完整 volume 的空间尺寸，用于边界裁剪

        Returns:
            扰动并裁回合法范围后的新 bbox。
            若 bbox_jitter=False 或 bbox_max_shift<=0，直接原样返回。

        实现细节：
            - 每个边界独立在 [-bbox_max_shift, +bbox_max_shift] 内随机偏移
            - 偏移后需裁回合法范围，保证：
                z0 ∈ [0, D-1]，z1 ∈ [z0+1, D]（框至少 1 个体素厚，不越出 volume）
              y/x 轴同理
        """
        if not self.bbox_jitter or self.bbox_max_shift <= 0:
            return bbox

        z0, z1, y0, y1, x0, x1 = bbox
        D, H, W = spatial_shape
        s = self.bbox_max_shift

        # 六个边界各自独立扰动
        z0 += random.randint(-s, s)
        z1 += random.randint(-s, s)
        y0 += random.randint(-s, s)
        y1 += random.randint(-s, s)
        x0 += random.randint(-s, s)
        x1 += random.randint(-s, s)

        # 裁回合法范围：起点 >= 0，终点 <= 轴长，且终点 > 起点（框不退化为零厚度）
        z0 = max(0, min(z0, D - 1))
        z1 = max(z0 + 1, min(z1, D))

        y0 = max(0, min(y0, H - 1))
        y1 = max(y0 + 1, min(y1, H))

        x0 = max(0, min(x0, W - 1))
        x1 = max(x0 + 1, min(x1, W))

        return (z0, z1, y0, y1, x0, x1)

    def __getitem__(self, idx: int):
        # ── Step 1: 索引映射 ──────────────────────────────────────────────────
        case_idx = self._indices[idx]
        pt_path = self.pt_paths[case_idx]

        # ── Step 2: 加载 .pt 文件 ─────────────────────────────────────────────
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

        image = data["image"]   # [1, D, H, W] float32
        label = data["label"]   # [1, D, H, W] int64，值域 {0,1,2}

        if image.ndim != 4 or label.ndim != 4:
            raise ValueError(
                f"expect image/label shape [C,D,H,W], got image={tuple(image.shape)} label={tuple(label.shape)}"
            )
        if image.shape[0] != 1 or label.shape[0] != 1:
            raise ValueError(
                f"expect single-channel image/label, got image={tuple(image.shape)} label={tuple(label.shape)}"
            )

        case_name = Path(pt_path).stem
        cur_margin = self._sample_margin()

        if "crop_bbox" in data:
            crop_bbox = data["crop_bbox"]
            tight_bbox = data["tight_bbox"]
            cz0, _, cy0, _, cx0, _ = crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3], crop_bbox[4], crop_bbox[5]
            tz0, tz1, ty0, ty1, tx0, tx1 = tight_bbox
            D, H, W = image.shape[1], image.shape[2], image.shape[3]
            rz0, rz1 = tz0 - cz0, tz1 - cz0
            ry0, ry1 = ty0 - cy0, ty1 - cy0
            rx0, rx1 = tx0 - cx0, tx1 - cx0
            bbox = (
                max(0, rz0 - cur_margin), min(D, rz1 + cur_margin),
                max(0, ry0 - cur_margin), min(H, ry1 + cur_margin),
                max(0, rx0 - cur_margin), min(W, rx1 + cur_margin),
            )
            image_roi = crop_3d(image, bbox)
            label_roi = crop_3d(label, bbox)
        elif self.pred_bboxes is not None and case_name in self.pred_bboxes:
            z0, z1, y0, y1, x0, x1 = self.pred_bboxes[case_name]
            D, H, W = image.shape[1], image.shape[2], image.shape[3]
            bbox = (
                max(0, z0 - cur_margin), min(D, z1 + cur_margin),
                max(0, y0 - cur_margin), min(H, y1 + cur_margin),
                max(0, x0 - cur_margin), min(W, x1 + cur_margin),
            )
            image_roi = crop_3d(image, bbox)
            label_roi = crop_3d(label, bbox)
        else:
            liver_mask = label[0] > 0
            liver_mask = filter_largest_component(liver_mask)
            bbox = compute_bbox_from_mask(liver_mask, margin=cur_margin)
            bbox = self._jitter_bbox(bbox, spatial_shape=tuple(label.shape[1:]))
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

        # ── transform（patch crop + 数据增强）────────────────────────────────
        if self.transform is not None:
            out = self.transform(out)

        # MONAI 某些 transform（如 RandCropByPosNegLabel）会返回 list
        # len==1 时拆包为单个 dict；len>1 时保留 list，
        # 由 DataLoader 的 list_data_collate 展平为 batch_size * num_samples
        if isinstance(out, list) and len(out) == 1:
            out = out[0]
        return out
