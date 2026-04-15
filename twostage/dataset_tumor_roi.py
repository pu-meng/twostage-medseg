from __future__ import annotations

import warnings
from pathlib import Path
from twostage_medseg.metrics.filter import filter_largest_component
from typing import Any
from typing import Sequence
import torch
from torch.utils.data import Dataset

# 相对导入：从当前包(twostage/)下引入工具函数
#   compute_bbox_from_mask : 根据 liver mask 计算紧致包围框并加 margin
#   crop_3d               : 按 bbox 裁剪 [C,D,H,W] 张量
#   bbox_to_dict          : 将 bbox tuple 转为可序列化的 dict(供 meta 记录)
from .roi_utils import compute_bbox_from_mask, crop_3d, bbox_to_dict
import random


class TumorROIDataset(Dataset):
    """
    Stage 2 肿瘤分割数据集：在线从完整 CT volume 裁出肝脏 ROI，再做肿瘤二分类。

    输入格式(离线预处理好的 .pt 文件)：
        {
            "image": Tensor[1, D, H, W]  float32   # CT 强度
            "label": Tensor[1, D, H, W]  int64     # 0=bg, 1=liver, 2=tumor
        }

    __getitem__ 在线处理流程：
        1. 用 GT label > 0 构造 liver mask，过滤最大连通域去噪
        2. 在 liver mask 上计算 bbox(加随机或固定 margin)
        3. 对 bbox 做随机扰动(bbox_jitter)，模拟 Stage 1 预测误差
        4. 按 bbox 裁出 image_roi / label_roi
        5. 将 label 重映射为肿瘤二分类：tumor(2)→1，其余→0
        6. 经 transform 做 patch crop + 数据增强后返回

    差异化 oversampling(hard case mining)：
        通过 __init__ 时预扫描每个 case 的肿瘤体素数，
        让目标样本在 _indices 中出现更多次，DataLoader 每 epoch
        更频繁地采到它们：
        - 无肿瘤 case  (voxels == 0)                          : repeat × no_tumor_repeat_scale
        - 小肿瘤 case  (0 < voxels < small_tumor_thresh)      : repeat × small_tumor_repeat_scale
        - 大肿瘤 case  (voxels >= large_tumor_thresh > 0)     : repeat × large_tumor_repeat_scale
        - 正常   case  (其余)                                 : repeat × 1
    """

    def __init__(
        self,
        pt_paths: Sequence[str],  # 所有 .pt 文件路径列表
        transform=None,  # MONAI transform，包含 patch crop + 增强
        repeats: int = 1,  # 每个 case 每 epoch 被采样的基础次数
        margin: int = 12,  # liver bbox 向外扩展的固定体素数
        keep_meta: bool = True,  # 是否在输出 dict 中附带调试元信息
        bbox_jitter: bool = False,  # 是否对 bbox 做随机扰动
        bbox_max_shift: int = 8,  # bbox 每个边界的最大扰动量(体素)
        random_margin: bool = False,  # 是否随机采样 margin(模拟 ROI 尺度波动)
        margin_min: int = 8,  # random_margin 时的最小 margin
        margin_max: int = 20,  # random_margin 时的最大 margin
        small_tumor_thresh: int = 0,  # 小肿瘤判定阈值(体素数)，0 表示关闭小肿瘤 hard mining
        small_tumor_repeat_scale: int = 1,  # 小肿瘤 case 的 repeat 倍率
        no_tumor_repeat_scale: int = 1,  # 无肿瘤 case 的 repeat 倍率
        large_tumor_thresh: int = 0,  # 大肿瘤判定阈值(体素数)，0 表示关闭大肿瘤过采样
        large_tumor_repeat_scale: int = 1,  # 大肿瘤 case 的 repeat 倍率
        pred_bboxes: dict | None = None,  # Stage1 预测 tight bbox 字典
        # 旧格式: {case_name: (z0,z1,y0,y1,x0,x1)}
        # 新格式: {case_name: {"liver": (...), "tumor": (...) or None}}
        # 传入时用预测 bbox 代替 GT bbox，消除训练/推理 domain gap
        # None 则保持原有 GT bbox + jitter 行为
        two_channel: bool = False,  # 是否启用两通道输入(Ch1=CT, Ch2=GT liver mask)
        # 训练时用 GT liver mask 作第二通道，推理时换成 Stage1 预测
        use_coarse_tumor: bool = False,  # 是否启用粗糙肿瘤通道(Ch1=CT, Ch2=Stage1粗糙肿瘤mask)
        # 需要 pred_bboxes 为新格式(含 tumor bbox)
        coarse_tumor_cache: dict | None = None,  # Stage1 软概率 cache，格式: {case_name: Tensor[1,D,H,W] float}
        # 传入时用 Stage1 软概率作为 Ch2（验证集），不传则用 GT 0/1 mask（训练集 teacher forcing）
        small_tumor_zoom_thresh: int = 0,   # 小肿瘤 zoom-in 阈值(体素数)，0=关闭
        small_tumor_zoom_factor: float = 2.0,  # zoom-in 倍率，2.0=放大2倍
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
        self.large_tumor_thresh = int(large_tumor_thresh)
        self.large_tumor_repeat_scale = int(large_tumor_repeat_scale)
        self.pred_bboxes = pred_bboxes
        self.two_channel = bool(two_channel)
        self.use_coarse_tumor = bool(use_coarse_tumor)
        self.coarse_tumor_cache = coarse_tumor_cache  # dict or None
        self.small_tumor_zoom_thresh = int(small_tumor_zoom_thresh)
        self.small_tumor_zoom_factor = float(small_tumor_zoom_factor)

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

        # 构建展开后的索引列表(hard mining 时困难 case 会出现多次)
        self._indices = self._build_indices()

    def _count_tumor_voxels(self, pt_path: str) -> int:
        """读取单个 .pt 文件，返回 label==2(肿瘤)的体素数，用于 hard mining 分类。"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(
                pt_path, map_location="cpu", weights_only=False, mmap=True
            )
        return int((data["label"] == 2).sum().item())

    def _build_indices(self) -> list[int]:
        """
        构建展开后的 case 索引列表，决定每个 case 每 epoch 被采样几次。

        - 未开启 hard mining(small_tumor_thresh==0 或倍率均为 1)：
          直接将 [0..N-1] 重复 repeats 次，所有 case 等频采样。
        - 开启 hard mining:
          预扫描所有 case 的肿瘤体素数，按大小分三类分别设置 repeat 倍率，
          困难 case(小肿瘤/无肿瘤)在列表中出现更多次，从而被更频繁采到。
        """
        use_hard_mining = (
            self.small_tumor_thresh > 0
            and (self.small_tumor_repeat_scale > 1 or self.no_tumor_repeat_scale > 1)
        ) or (self.large_tumor_thresh > 0 and self.large_tumor_repeat_scale > 1)
        if not use_hard_mining:
            # 所有 case 等频：[0,1,...,N-1, 0,1,...,N-1, ...] 共 repeats 轮
            return list(range(len(self.pt_paths))) * self.repeats

        indices = []
        n_no_tumor = 0
        n_small_tumor = 0
        n_large_tumor = 0
        n_normal = 0

        N = len(self.pt_paths)
        print(f"[TumorROIDataset] scanning {N} cases for hard mining ...", flush=True)
        for i, pt_path in enumerate(self.pt_paths):
            if i % 20 == 0:
                print(f"  [{i}/{N}]", flush=True)
            voxels = self._count_tumor_voxels(pt_path)
            if voxels == 0:
                r = self.repeats * self.no_tumor_repeat_scale
                n_no_tumor += 1
            elif self.small_tumor_thresh > 0 and voxels < self.small_tumor_thresh:
                r = self.repeats * self.small_tumor_repeat_scale
                n_small_tumor += 1
            elif self.large_tumor_thresh > 0 and voxels >= self.large_tumor_thresh:
                r = self.repeats * self.large_tumor_repeat_scale
                n_large_tumor += 1
            else:
                r = self.repeats
                n_normal += 1
            indices.extend([i] * r)

        print(
            f"[TumorROIDataset] hard mining: "
            f"no_tumor={n_no_tumor}×{self.no_tumor_repeat_scale}, "
            f"small_tumor(<{self.small_tumor_thresh})={n_small_tumor}×{self.small_tumor_repeat_scale}, "
            f"large_tumor(>={self.large_tumor_thresh})={n_large_tumor}×{self.large_tumor_repeat_scale}, "
            f"normal={n_normal}×1 | "
            f"total indices={len(indices)}"
        )
        return indices

    def __len__(self) -> int:
        # 返回展开后的总样本数(DataLoader 据此决定每 epoch 迭代次数)
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
                z0 ∈ [0, D-1]，z1 ∈ [z0+1, D](框至少 1 个体素厚，不越出 volume)
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

        # 裁回合法范围：起点 >= 0，终点 <= 轴长，且终点 > 起点(框不退化为零厚度)
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
            raise TypeError(
                f"expect dict from torch.load, got {type(data)} @ {pt_path}"
            )
        if "image" not in data or "label" not in data:
            raise KeyError(f"pt sample missing image/label keys @ {pt_path}")

        image = data["image"]  # [1, D, H, W] float32
        label = data["label"]  # [1, D, H, W] int64，值域 {0,1,2}

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
            cz0, _, cy0, _, cx0, _ = (
                crop_bbox[0],
                crop_bbox[1],
                crop_bbox[2],
                crop_bbox[3],
                crop_bbox[4],
                crop_bbox[5],
            )
            tz0, tz1, ty0, ty1, tx0, tx1 = tight_bbox
            D, H, W = image.shape[1], image.shape[2], image.shape[3]
            rz0, rz1 = tz0 - cz0, tz1 - cz0
            ry0, ry1 = ty0 - cy0, ty1 - cy0
            rx0, rx1 = tx0 - cx0, tx1 - cx0
            bbox = (
                max(0, rz0 - cur_margin),
                min(D, rz1 + cur_margin),
                max(0, ry0 - cur_margin),
                min(H, ry1 + cur_margin),
                max(0, rx0 - cur_margin),
                min(W, rx1 + cur_margin),
            )
            image_roi = crop_3d(image, bbox)
            label_roi = crop_3d(label, bbox)
        elif self.pred_bboxes is not None and case_name in self.pred_bboxes:
            entry = self.pred_bboxes[case_name]
            # 兼容新旧格式：旧格式为 tuple，新格式为 dict
            if isinstance(entry, dict):
                z0, z1, y0, y1, x0, x1 = entry["liver"]
            else:
                z0, z1, y0, y1, x0, x1 = entry
            D, H, W = image.shape[1], image.shape[2], image.shape[3]
            bbox = (
                max(0, z0 - cur_margin),
                min(D, z1 + cur_margin),
                max(0, y0 - cur_margin),
                min(H, y1 + cur_margin),
                max(0, x0 - cur_margin),
                min(W, x1 + cur_margin),
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

        # ── Zoom-in：对小肿瘤 case 缩小 ROI 物理范围，等价于放大肿瘤 ──────────
        # 仅在训练时生效（transform不为None），推理时不做
        # 判断是否为小肿瘤：GT肿瘤体素数 < small_tumor_zoom_thresh
        # 做法：以肿瘤中心为中心，取原ROI的 1/zoom_factor 大小的子区域，
        #       再用插值resize回原ROI尺寸，等价于放大zoom_factor倍
        if (
            self.transform is not None
            and self.small_tumor_zoom_thresh > 0
            and self.small_tumor_zoom_factor > 1.0
        ):
            n_tumor = int((label_roi == 2).sum().item())
            if 0 < n_tumor < self.small_tumor_zoom_thresh:
                # 肿瘤中心坐标（在ROI坐标系内）
                coords = torch.nonzero(label_roi[0] == 2, as_tuple=False).float()
                center = coords.mean(dim=0)  # [D, H, W] 方向
                d, h, w = image_roi.shape[1], image_roi.shape[2], image_roi.shape[3]
                # 缩小后的子区域大小
                sd = max(1, int(round(d / self.small_tumor_zoom_factor)))
                sh = max(1, int(round(h / self.small_tumor_zoom_factor)))
                sw = max(1, int(round(w / self.small_tumor_zoom_factor)))
                # 子区域起止，以肿瘤中心为中心，clamp保证不越界
                z0 = int(torch.clamp(center[0] - sd // 2, 0, d - sd).item())
                y0 = int(torch.clamp(center[1] - sh // 2, 0, h - sh).item())
                x0 = int(torch.clamp(center[2] - sw // 2, 0, w - sw).item())
                # 裁出子区域
                image_roi = image_roi[:, z0:z0+sd, y0:y0+sh, x0:x0+sw]
                label_roi = label_roi[:, z0:z0+sd, y0:y0+sh, x0:x0+sw]
                # 插值回原始ROI尺寸
                image_roi = torch.nn.functional.interpolate(
                    image_roi.unsqueeze(0).float(), size=(d, h, w),
                    mode="trilinear", align_corners=False
                ).squeeze(0)
                label_roi = torch.nn.functional.interpolate(
                    label_roi.unsqueeze(0).float(), size=(d, h, w),
                    mode="nearest"
                ).squeeze(0).long()
        # ─────────────────────────────────────────────────────────────────────

        tumor_roi = (label_roi == 2).long()

        if self.use_coarse_tumor:
            if self.coarse_tumor_cache is not None and case_name in self.coarse_tumor_cache:
                # 验证时：用 Stage1 软概率（已裁到全图坐标系），裁到当前 ROI bbox
                full_prob = self.coarse_tumor_cache[case_name]  # [1, D, H, W] float
                coarse_tumor = crop_3d(full_prob, bbox).float()  # [1, d, h, w]
            else:
                # 训练时：用 GT tumor mask 做 teacher forcing（0/1）
                coarse_tumor = (label_roi == 2).float()  # [1, d, h, w]
            image_2ch = torch.cat(
                [image_roi.float(), coarse_tumor], dim=0
            )  # [2, d, h, w]
        elif self.two_channel:
            # Ch1: CT，Ch2: GT liver mask(训练时用 GT，推理时由 eval 脚本替换为 Stage1 预测)
            liver_roi = (label_roi > 0).float()  # [1, d, h, w]  0/1
            image_2ch = torch.cat([image_roi.float(), liver_roi], dim=0)  # [2, d, h, w]
        else:
            image_2ch = image_roi.float()

        out: dict[str, Any] = {
            "image": image_2ch,
            "label": tumor_roi.long(),
        }
        if self.keep_meta:
            out["case_name"] = Path(pt_path).stem
            out["bbox"] = bbox_to_dict(bbox)
            out["source_pt"] = pt_path
            out["roi_shape"] = list(image_roi.shape)
            out["margin"] = int(cur_margin)

        # ── transform(patch crop + 数据增强)────────────────────────────────
        if self.transform is not None:
            out = self.transform(out)

        # MONAI 某些 transform(如 RandCropByPosNegLabel)会返回 list
        # len==1 时拆包为单个 dict；len>1 时保留 list，
        # 由 DataLoader 的 list_data_collate 展平为 batch_size * num_samples
        if isinstance(out, list) and len(out) == 1:
            out = out[0]
        return out
