# eval_twostage_simple.py
#

"""
验证 实验10 best.pt(04-02-16-03-10):

CUDA_VISIBLE_DEVICES=1 python scripts/eval_twostage.py \
  --medseg_root /home/pumengyu/medseg_project \
  --preprocessed_root /home/pumengyu/Task03_Liver_pt \
  --stage1_ckpt /home/pumengyu/experiments/dynunet_liver_only/train/03-14-01-11-56/best.pt \
  --stage2_ckpt  \
  --stage1_model dynunet --stage2_model dynunet_ca \
  --stage1_patch 144 144 144 --stage2_patch 128 128 128 \
  --stage2_sw_batch_size 2 \
  --val_ratio 0.2 --test_ratio 0.1 --seed 0 \
  --overlap 0.5 --split test --tta --min_tumor_size 100

"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import numpy as np
import scipy.ndimage as ndi

import sys
import time
from typing import Dict, List

import torch
from monai.inferers.utils import sliding_window_inference
from medseg.models.build_model import build_model
from medseg.utils.ckpt import load_ckpt

from medseg.data.dataset_offline import split_two_with_monitor

# twostage_medseg/metrics/filter.py
from metrics.filter import filter_largest_component
from metrics.metrics_utils import compute_metrics, summarize_metrics_list
from twostage.vis_utils import save_case_visualization
from 展示.vis_prob import vis_worst_cases


def add_medseg_to_syspath(medseg_root: str) -> None:
    medseg_root = os.path.abspath(medseg_root)
    if medseg_root not in sys.path:
        sys.path.insert(0, medseg_root)


def parse_args():
    _home = os.path.expanduser("~")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--medseg_root", type=str, default=os.path.join(_home, "medseg_project")
    )
    p.add_argument(
        "--preprocessed_root", type=str, default=os.path.join(_home, "Task03_Liver_pt")
    )
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, default=None)
    p.add_argument(
        "--stage1_only",
        action="store_true",
        help="跳过 Stage2,直接用 Stage1 三分类输出(类别2=肿瘤)作为最终预测",
    )
    p.add_argument(
        "--stage2_ckpt_b",
        type=str,
        default=None,
        help="fine-tune 模型路径,有则与 stage2_ckpt 做 ensemble",
    )
    p.add_argument(
        "--ensemble_weight_b",
        type=float,
        default=0.5,
        help="Model B 的融合权重,Model A = 1 - weight_b",
    )

    p.add_argument("--stage1_model", type=str, default="dynunet")
    p.add_argument("--stage2_model", type=str, default="dynunet")

    p.add_argument("--stage1_patch", type=int, nargs=3, default=[144, 144, 144])
    p.add_argument("--stage2_patch", type=int, nargs=3, default=[96, 96, 96])

    p.add_argument("--stage1_sw_batch_size", type=int, default=1)
    p.add_argument("--stage2_sw_batch_size", type=int, default=1)

    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--margin", type=int, default=12)

    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="训练集:模型看到数据,更新参数;验证集:训练过程中监控loss/dice,选择best.pt;测试集:训练完全结束,最终评估性能,报告论文指标",
    )
    p.add_argument("--n", type=int, default=0, help="只跑前N个case, 0=全部")
    p.add_argument("--save_pred_pt", action="store_true", help="是否保存预测结果")
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="结果保存目录,默认从 stage2_ckpt 自动推导为 <exp_root>/eval",
    )
    p.add_argument("--vis_n", type=int, default=13, help="最多保存前N个case的可视化")
    p.add_argument(
        "--vis_all",
        action="store_true",
        help="保存所有case的可视化(忽略--vis_n限制)",
    )
    p.add_argument(
        "--eval_liver_filled",
        action="store_true",
        help="开启后额外计算用孔洞填充后肝脏(liver_filled)的liver dice,并在可视化中新增一列展示",
    )
    p.add_argument("--min_tumor_size", type=int, default=100)
    p.add_argument(
        "--comp_prob_thresh",
        type=float,
        default=0.5,
        help="连通域自适应阈值：体积 < min_tumor_size 但平均概率 >= 此值时也保留（默认0.5）。"
        "降低此值可以救回极小高置信度肿瘤，建议0.4~0.6。",
    )
    p.add_argument(
        "--small_tumor_low_thresh",
        type=float,
        default=0.0,
        help="若初始 pred_tumor 体积在 [200,1000] voxel，用此更低阈值重新生成预测(默认0=关闭)。"
        "建议值 0.05~0.15，激进覆盖极小肿瘤。",
    )
    p.add_argument(
        "--no_tta",
        action="store_true",
        help="关闭测试时增强(默认开启:对D/H/W三轴所有翻转组合推理后取平均,8次推理)",
    )
    p.add_argument(
        "--tta_stage1",
        action="store_true",
        help="仅对Stage1也做TTA(默认TTA只加在Stage2上)",
    )
    p.add_argument(
        "--two_channel",
        action="store_true",
        help="Stage2 使用两通道输入(Ch1=CT, Ch2=Stage1预测liver_mask),需与训练时一致",
    )
    p.add_argument(
        "--use_coarse_tumor",
        action="store_true",
        help="Stage2 使用粗糙肿瘤通道(Ch1=CT, Ch2=Stage1预测粗糙肿瘤mask),需与训练时一致",
    )
    p.add_argument(
        "--stage1_out_channels",
        type=int,
        default=2,
        help="Stage1 输出通道数:2=仅肝脏(旧版),3=肝脏+肿瘤(新版)",
    )
    return p.parse_args()


def safe_case_name(pt_path: str) -> str:
    """
    类似pt_path="/home/.../case_1.pt", 返回"case_1"
    """
    name = os.path.basename(pt_path)
    if name.endswith(".pt"):
        name = name[:-3]
    return name


def load_pt_paths(preprocessed_root: str) -> List[str]:
    pt_paths = sorted(glob.glob(os.path.join(preprocessed_root, "*.pt")))
    if len(pt_paths) == 0:
        raise FileNotFoundError(f"no .pt found in {preprocessed_root}")
    return pt_paths


def compute_bbox_from_mask(mask: torch.Tensor, margin: int = 12):
    """
    mask: [D,H,W] bool
    return: ((z0,z1),(y0,y1),(x0,x1)) 右边界开区间

    """
    if mask.sum().item() == 0:
        return None

    coords = torch.nonzero(mask, as_tuple=False)
    # coords:[N,3],N=mask中非零点的数量,每一行是一个点的坐标[z,y,x]
    zmin, ymin, xmin = coords.min(dim=0).values.tolist()
    zmax, ymax, xmax = coords.max(dim=0).values.tolist()

    D, H, W = mask.shape

    z0 = max(0, zmin - margin)
    y0 = max(0, ymin - margin)
    x0 = max(0, xmin - margin)

    z1 = min(D, zmax + 1 + margin)
    y1 = min(H, ymax + 1 + margin)
    x1 = min(W, xmax + 1 + margin)

    return ((z0, z1), (y0, y1), (x0, x1))


def crop_3d(x: torch.Tensor, bbox):
    """
    x: [1,D,H,W] or [D,H,W]
    """
    (z0, z1), (y0, y1), (x0, x1) = bbox
    if x.ndim == 4:
        return x[:, z0:z1, y0:y1, x0:x1]
    elif x.ndim == 3:
        return x[z0:z1, y0:y1, x0:x1]
    else:
        raise ValueError(f"unsupported ndim={x.ndim}")


def paste_3d(dst: torch.Tensor, src: torch.Tensor, bbox):
    """
    dst: [D,H,W]全零,全局坐标系;
    dst=全局大图,尺寸是完整的CT的[D,H,W]
    src: [d,h,w]是ROI裁剪区域的预测
    src=Stage2在肝脏ROI上预测出来的小肿瘤mask,尺寸是裁剪后的[d,h,w]

    作用=把小图的预测结果"贴回"到大图对应的bbox位置
    """
    (z0, z1), (y0, y1), (x0, x1) = bbox
    out = dst.clone()
    out[z0:z1, y0:y1, x0:x1] = src
    return out


def bbox_to_dict(bbox):
    if bbox is None:
        return None
    (z0, z1), (y0, y1), (x0, x1) = bbox
    return {
        "z0": int(z0),
        "z1": int(z1),
        "y0": int(y0),
        "y1": int(y1),
        "x0": int(x0),
        "x1": int(x1),
    }


def build_final_pred_from_liver_tumor(
    liver_mask: torch.Tensor,
    tumor_mask: torch.Tensor,
    use_filled_liver: bool = True,
    liver_filled: torch.Tensor = None,
) -> torch.Tensor:
    """
    output:
      0 bg
      1 liver
      2 tumor

    Args:
        liver_mask: 原始肝脏mask(可能包含空洞）
        tumor_mask: 肿瘤mask
        use_filled_liver: 是否使用填充后的肝脏(liver_filled），默认为True
        liver_filled: 填充后的肝脏mask(当use_filled_liver=True时必须提供）
    """
    if use_filled_liver:
        if liver_filled is None:
            raise ValueError("当use_filled_liver=True时,必须提供liver_filled参数")
        final_liver = liver_filled
    else:
        final_liver = liver_mask

    final_pred = torch.zeros_like(liver_mask, dtype=torch.long)
    final_pred[final_liver] = 1
    final_pred[tumor_mask] = 2
    return final_pred


def tta_sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap):
    """
    对 D/H/W 三轴的所有翻转组合(2^3=8种)分别做滑动窗口推理,
    将 8 次预测的 logits 平均后返回。
    inputs: [1,1,D,H,W],在 GPU 上
    """
    # 三个空间轴在5D tensor中的索引:D=2, H=3, W=4
    spatial_axes = [2, 3, 4]

    # 枚举所有子集:[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]
    flip_combinations = [[]]
    for ax in spatial_axes:
        flip_combinations += [combo + [ax] for combo in flip_combinations]

    logits_sum = None
    for flip_axes in flip_combinations:
        x = torch.flip(inputs, dims=flip_axes) if flip_axes else inputs
        logits = sliding_window_inference(
            x, roi_size, sw_batch_size, predictor, overlap
        )
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = torch.as_tensor(logits).float()
        # 翻转回原始方向
        if flip_axes:
            logits = torch.flip(logits, dims=flip_axes)
        logits_sum = logits if logits_sum is None else logits_sum + logits

    return logits_sum / len(flip_combinations)


def main():
    args = parse_args()
    add_medseg_to_syspath(args.medseg_root)

    if args.stage1_only:
        if args.stage1_out_channels < 3:
            raise ValueError("--stage1_only 需要 --stage1_out_channels 3")
        ref_ckpt = args.stage1_ckpt
    else:
        if args.stage2_ckpt is None:
            raise ValueError("非 --stage1_only 模式下必须传 --stage2_ckpt")
        ref_ckpt = args.stage2_ckpt

    ckpt_abs = os.path.abspath(ref_ckpt)
    run_name = os.path.basename(os.path.dirname(ckpt_abs))  # 时间戳目录名
    if args.save_dir is None:
        # .../exp_name/train/timestamp/best.pt → .../exp_name/eval
        exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt_abs)))
        args.save_dir = os.path.join(exp_dir, "eval")
    os.makedirs(args.save_dir, exist_ok=True)
    workdir = os.path.join(args.save_dir, run_name)
    os.makedirs(workdir, exist_ok=True)

    with open(os.path.join(workdir, "command.txt"), "w", encoding="utf-8") as f:
        argv = sys.argv[:]
        # 读取环境变量 CUDA_VISIBLE_DEVICES(如 "0" 或 "0,1"),未设置则为 None
        cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        # 若设置了 GPU 环境变量,则拼上前缀；否则前缀为空字符串
        prefix = f"CUDA_VISIBLE_DEVICES={cuda_dev} " if cuda_dev is not None else ""
        # sys.executable 是当前 Python 解释器路径,argv[0] 是脚本路径
        # 拼成完整的命令开头,如:CUDA_VISIBLE_DEVICES=0 /path/to/python scripts/eval_twostage.py
        first_line = f"{prefix}{sys.executable} {argv[0]}"
        lines = [first_line]
        i = 1
        while i < len(argv):
            tok = argv[i]
            if (
                tok.startswith("--")
                and i + 1 < len(argv)
                and not argv[i + 1].startswith("--")
            ):
                # 收集同一个 key 后面可能跟的多个值(如 --patch 96 96 96)
                vals = []
                j = i + 1
                while j < len(argv) and not argv[j].startswith("--"):
                    vals.append(argv[j])
                    j += 1
                lines.append(f"  {tok} {' '.join(vals)}")
                i = j
            else:
                lines.append(f"  {tok}")
                i += 1
        f.write(" \\\n".join(lines) + "\n")

    prob_dir = os.path.join(workdir, "prob_pt")  # 写磁盘前先攒内存，eval结束后只存最差N个
    vis_dir = os.path.join(workdir, "vis_png")
    prob_buffer: dict = {}  # case_name -> prob2_full tensor，先全攒内存
    os.makedirs(vis_dir, exist_ok=True)

    all_pt = load_pt_paths(args.preprocessed_root)
    # 与训练保持一致：split_two_with_monitor
    # train=112(全部非nnUNet fold0), monitor=12(train子集), test=19(nnUNet fold0)
    tr, va, te = split_two_with_monitor(all_pt)
    # 这里的决定的是在什么测试这个指标,如果不传args.split,就默认是test
    if args.split == "train":
        pt_paths = tr
    elif args.split == "val":
        pt_paths = va
    elif args.split == "test":
        pt_paths = te
    elif args.split == "all":
        pt_paths = all_pt
    else:
        raise ValueError(f"unsupported split: {args.split}")
    liver_metrics_list: List[float] = []
    liver_filled_metrics_list: List[float] = []  # eval_liver_filled 开启时使用
    tumor_metrics_list: List[float] = []  # 只含 gt 有肿瘤的 case
    tumor_metrics_list_neg: List[float] = []  # gt 无肿瘤的 case(用于统计误报率)

    if args.n > 0:
        print(
            f"[eval] --n={args.n}: 截断为 {len(pt_paths)} -> {min(args.n, len(pt_paths))} cases"
        )
        pt_paths = pt_paths[: args.n]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # stage1 ckpt 是 deep_supervision=False 训练的,推理时也用 False
    if args.stage1_model in ["dynunet", "nnunet"]:
        from medseg.models.dynunet import build_dynunet

        stage1 = build_dynunet(
            in_channels=1, out_channels=args.stage1_out_channels, deep_supervision=False
        ).to(device)
    else:
        stage1 = build_model(
            args.stage1_model,
            in_channels=1,
            out_channels=args.stage1_out_channels,
            img_size=tuple(args.stage1_patch),
        ).to(device)

    load_ckpt(args.stage1_ckpt, stage1, optimizer=None, map_location=device)
    stage1.eval()

    stage2 = None
    stage2_b = None
    if not args.stage1_only:
        stage2_in_channels = 2 if (args.two_channel or args.use_coarse_tumor) else 1
        stage2 = build_model(
            args.stage2_model,
            in_channels=stage2_in_channels,
            out_channels=2,
            img_size=tuple(args.stage2_patch),
        ).to(device)
        load_ckpt(args.stage2_ckpt, stage2, optimizer=None, map_location=device)
        stage2.eval()

        if args.stage2_ckpt_b is not None:
            stage2_b = build_model(
                args.stage2_model,
                in_channels=stage2_in_channels,
                out_channels=2,
                img_size=tuple(args.stage2_patch),
            ).to(device)
            load_ckpt(args.stage2_ckpt_b, stage2_b, optimizer=None, map_location=device)
            stage2_b.eval()
            print(
                f"[eval] ensemble: weight_A={1 - args.ensemble_weight_b:.2f}  weight_B={args.ensemble_weight_b:.2f}"
            )

    print(f"[eval] split={args.split}")
    print(
        f"[eval] val_ratio={args.val_ratio} test_ratio={args.test_ratio} seed={args.seed}"
    )
    print(f"[eval] n_cases={len(pt_paths)}")
    print(f"[eval] device={device}")

    rows: List[Dict] = []
    stage2_prob_diag: List[Dict] = []  # 收集每个 case 的 Stage2 概率图诊断信息

    time_start = time.time()

    with torch.no_grad():  # 评估阶段不需要计算梯度,节省显存和时间
        for case_idx, pt_path in enumerate(pt_paths, start=1):
            case_name = safe_case_name(pt_path)
            # mmap=True: 内存映射加载,大文件不会一次性全读入内存
            data = torch.load(
                pt_path, map_location="cpu", weights_only=False, mmap=True
            )

            image = data["image"].float()  # [1,D,H,W],CT 灰度图,已预处理
            label = data.get(
                "label", None
            )  # [1,D,H,W],0=背景 1=肝脏 2=肿瘤,推理时可为 None

            # ================================================================
            # STAGE 1:肝脏分割(在完整 CT 体积上滑动窗口推理)
            # 输入:完整 CT x,形状 [1,1,D,H,W](Batch=1, Channel=1)
            # 输出:logits1,形状 [1,2,D,H,W](2 个类别:背景/肝脏)
            # ================================================================
            x = image.unsqueeze(0).to(device)  # [1,D,H,W] → [1,1,D,H,W],加 batch 维

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,  # 使用 FP16 混合精度,减少显存占用、加速推理
                enabled=(device == "cuda"),
            ):
                # CT 体积太大无法一次性放入 GPU,sliding_window_inference 将其切成
                # stage1_patch 大小的小块逐一推理,再拼回原始大小(重叠区域取平均)
                if not args.no_tta and args.tta_stage1:
                    logits1 = tta_sliding_window_inference(
                        inputs=x,
                        roi_size=tuple(args.stage1_patch),
                        sw_batch_size=args.stage1_sw_batch_size,
                        predictor=stage1,
                        overlap=args.overlap,
                    )
                else:
                    logits1 = sliding_window_inference(
                        inputs=x,  # 完整 CT [1,1,D,H,W]
                        roi_size=tuple(
                            args.stage1_patch
                        ),  # 每个 patch 大小,如 [144,144,144]
                        sw_batch_size=args.stage1_sw_batch_size,  # 同时推理的 patch 数,越大越占显存
                        predictor=stage1,  # Stage1 肝脏分割模型
                        overlap=args.overlap,  # patch 间重叠比例,越大越精确但越慢
                    )
                # logits1: [1,2,D,H,W],2个通道分别是"背景"和"肝脏"的未归一化分数

            if isinstance(logits1, (tuple, list)):
                logits1 = logits1[0]  # 部分模型返回 tuple,取第一个元素

            logits1 = torch.as_tensor(
                logits1
            )  # 防御性:确保是 tensor(autocast 可能返回其他类型)
            # argmax 在 channel 维取最大值 → 每个 voxel 得到类别 id (0 或 1)
            pred1 = torch.argmax(logits1.float(), dim=1)[
                0
            ].cpu()  # [1,C,D,H,W] → [D,H,W]
            liver_mask = pred1 == 1  # [D,H,W] bool,True 的位置就是预测的肝脏
            coarse_tumor_mask = (pred1 == 2) if args.stage1_out_channels == 3 else None

            # 后处理:只保留最大连通域,去掉散落的假阳性小块
            liver_mask = filter_largest_component(liver_mask)
            pred1_filtered = liver_mask.long()  # 用于后续可视化保存

            # ================================================================
            # stage1_only 模式:直接用 Stage1 三分类的类别2作为肿瘤预测,跳过 Stage2
            # ================================================================
            if args.stage1_only:
                bbox = None
                tumor_full = (pred1 == 2).long()  # Stage1 直接输出的粗糙肿瘤
            # ================================================================
            # STAGE 2:肿瘤分割(在裁剪出的肝脏 ROI 上滑动窗口推理)
            # 依赖 Stage1 结果:必须先有 liver_mask 才能知道裁哪里
            # ================================================================
            elif liver_mask.sum().item() == 0:
                # Stage1 没有预测出任何肝脏(极少数情况),则肿瘤直接置全零
                bbox = None
                tumor_full = torch.zeros_like(pred1, dtype=torch.long)
            else:
                # 根据肝脏 mask 计算 3D bounding box,并向外扩 margin 个 voxel
                # bbox 格式:((z0,z1),(y0,y1),(x0,x1))
                bbox = compute_bbox_from_mask(liver_mask, margin=args.margin)

                # 把 CT 裁剪到肝脏 ROI 区域,大幅缩小 Stage2 的输入尺寸
                image_roi = crop_3d(image, bbox)  # [1,D,H,W] → [1,d,h,w](d≤D, h≤H, w≤W)

                if args.use_coarse_tumor:
                    # 第二通道:Stage1 预测粗糙肿瘤 mask 裁到同一 bbox(无肿瘤预测时全零)
                    if (
                        coarse_tumor_mask is not None
                        and coarse_tumor_mask.sum().item() > 0
                    ):
                        tumor_roi = crop_3d(
                            coarse_tumor_mask.float().unsqueeze(0), bbox
                        )
                    else:
                        tumor_roi = torch.zeros_like(image_roi)
                    x_roi = (
                        torch.cat([image_roi, tumor_roi], dim=0).unsqueeze(0).to(device)
                    )
                elif args.two_channel:
                    # 第二通道:Stage1 预测 liver_mask 裁到同一 bbox
                    liver_roi = crop_3d(
                        liver_mask.float().unsqueeze(0), bbox
                    )  # [1,d,h,w]
                    image_roi_2ch = torch.cat(
                        [image_roi, liver_roi], dim=0
                    )  # [2,d,h,w]
                    x_roi = image_roi_2ch.unsqueeze(0).to(device)  # [1,2,d,h,w]
                else:
                    x_roi = image_roi.unsqueeze(0).to(device)  # [1,1,d,h,w],加 batch 维

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(device == "cuda"),
                ):
                    # Stage2 在裁剪后的肝脏 ROI 上做滑动窗口推理(输入比 Stage1 小得多)
                    # 注意:这里用的是 stage2_patch(如 96³),而非 stage1_patch(如 144³)
                    if not args.no_tta:
                        logits2 = tta_sliding_window_inference(
                            inputs=x_roi,
                            roi_size=tuple(args.stage2_patch),
                            sw_batch_size=args.stage2_sw_batch_size,
                            predictor=stage2,
                            overlap=args.overlap,
                        )
                    else:
                        logits2 = sliding_window_inference(
                            inputs=x_roi,  # 肝脏 ROI [1,1,d,h,w]
                            roi_size=tuple(args.stage2_patch),  # 如 [96,96,96]
                            sw_batch_size=args.stage2_sw_batch_size,
                            predictor=stage2,  # Stage2 肿瘤分割模型 A
                            overlap=args.overlap,
                        )
                    # logits2: [1,2,d,h,w],2个通道分别是"背景"和"肿瘤"的分数

                    if stage2_b is not None:
                        # 如果提供了第二个 Stage2 模型(模型 B),用相同输入再推理一次
                        # 后续会将 A 和 B 的 logits 加权平均(ensemble),提升稳定性
                        if not args.no_tta:
                            logits2_b = tta_sliding_window_inference(
                                inputs=x_roi,
                                roi_size=tuple(args.stage2_patch),
                                sw_batch_size=args.stage2_sw_batch_size,
                                predictor=stage2_b,
                                overlap=args.overlap,
                            )
                        else:
                            logits2_b = sliding_window_inference(
                                inputs=x_roi,
                                roi_size=tuple(args.stage2_patch),
                                sw_batch_size=args.stage2_sw_batch_size,
                                predictor=stage2_b,  # Stage2 肿瘤分割模型 B
                                overlap=args.overlap,
                            )

                if isinstance(logits2, (tuple, list)):
                    logits2 = logits2[0]
                logits2 = torch.as_tensor(logits2).float()

                if stage2_b is not None:
                    if isinstance(logits2_b, (tuple, list)):
                        logits2_b = logits2_b[0]
                    logits2_b = torch.as_tensor(logits2_b).float()
                    w = args.ensemble_weight_b
                    # 加权平均:logits2 = (1-w)*模型A + w*模型B
                    # 例如 w=0.5 时两个模型等权融合
                    logits2 = (1 - w) * logits2 + w * logits2_b

                # [1,2,d,h,w] → [d,h,w],sigmoid后取前景通道,阈值0.3提高小肿瘤召回
                prob2 = torch.sigmoid(logits2[0, 1]).cpu()  # [d,h,w] float
                # ── Stage2 概率图诊断 ────────────────────────────────────────
                _p_max = float(prob2.max())
                _p_mean = float(prob2.mean())
                _p01 = int((prob2 > 0.1).sum())
                _p03 = int((prob2 > 0.3).sum())
                _p05 = int((prob2 > 0.5).sum())
                stage2_prob_diag.append(
                    {
                        "case_name": case_name,
                        "prob_max": _p_max,
                        "prob_mean": _p_mean,
                        "vox_01": _p01,
                        "vox_03": _p03,
                        "vox_05": _p05,
                    }
                )
                # ─────────────────────────────────────────────────────────────
                # 保存 prob map（paste回全图坐标系）
                prob2_full = torch.zeros(pred1.shape, dtype=torch.float32)
                prob2_full = paste_3d(prob2_full, prob2, bbox)
                prob_buffer[case_name] = prob2_full.clone()  # 先攒内存，eval结束后按dice过滤再写盘

                pred2 = (prob2 > 0.3).long()  # [d,h,w]

                # 小肿瘤激进后处理:若初始预测体积极小(200~1000 voxel),认为模型过于保守,
                # 用更低阈值重新生成预测,允许多倍体积扩张以暴力覆盖住小肿瘤
                _pred_vol = int(pred2.sum().item())
                if args.small_tumor_low_thresh > 0 and 200 <= _pred_vol <= 1000:
                    pred2 = (prob2 > args.small_tumor_low_thresh).long()

                # 将 ROI 内的肿瘤预测"贴回"到完整 CT 坐标系中
                # tumor_full 初始全零(和完整 CT 同大小),paste_3d 把 pred2 写入 bbox 对应位置
                tumor_full = torch.zeros_like(pred1, dtype=torch.long)  # [D,H,W] 全零
                tumor_full = paste_3d(tumor_full, pred2.long(), bbox)  # 贴回原始坐标

            # ================================================================
            # 后处理:约束肿瘤在肝脏内,去除小假阳性连通域
            # ================================================================
            tumor_mask = tumor_full == 1  # [D,H,W] bool,Stage2 预测的肿瘤位置
            # 肝脏是实心器官，内部不应有空洞(高密度肿瘤会导致 Stage1 在肿瘤处预测出空洞）
            # 填充 liver_mask 的内部空洞，确保肝脏轮廓内部全部为实心区域
            # binary_fill_holes 3D 对开口空洞无效，改为逐 slice 2D 填充取三轴并集
            liver_np = liver_mask.cpu().numpy()
            filled_ax0 = np.stack([ndi.binary_fill_holes(liver_np[i]) for i in range(liver_np.shape[0])])
            filled_ax1 = np.stack([ndi.binary_fill_holes(liver_np[:, i, :]) for i in range(liver_np.shape[1])]).transpose(1, 0, 2)
            filled_ax2 = np.stack([ndi.binary_fill_holes(liver_np[:, :, i]) for i in range(liver_np.shape[2])]).transpose(1, 2, 0)
            liver_filled = torch.from_numpy(
                filled_ax0 | filled_ax1 | filled_ax2
            ).to(liver_mask.device)
            tumor_mask = (
                tumor_mask & liver_filled
            )  # 肿瘤只能在肝脏实心区域内(交集),排除肝外假阳性

            # 连通域分析：自适应阈值
            # 规则：体积 > min_tumor_size 直接保留；
            #       体积 <= min_tumor_size 但连通域内平均概率 >= comp_prob_thresh 也保留
            #       （救回极小高置信度肿瘤，同时过滤低概率FP碎片）
            tumor_mask_np = tumor_mask.cpu().numpy()
            prob2_full_np = prob2_full.numpy()
            labeled, num = ndi.label(tumor_mask_np)

            sizes = ndi.sum(tumor_mask_np, labeled, range(1, num + 1))

            clean = torch.zeros_like(tumor_mask)
            for comp_idx, s in enumerate(sizes):
                comp_id = comp_idx + 1
                if s > args.min_tumor_size:
                    clean[labeled == comp_id] = 1
                else:
                    # 体积小：看平均概率决定是否保留
                    mean_prob = float(prob2_full_np[labeled == comp_id].mean())
                    if mean_prob >= args.comp_prob_thresh:
                        clean[labeled == comp_id] = 1

            tumor_mask = clean.bool()

            # ================================================================
            # 合并最终预测:0=背景,1=肝脏,2=肿瘤
            # 使用填充后的肝脏(liver_filled）作为最终肝脏预测
            # ================================================================
            final_pred = build_final_pred_from_liver_tumor(
                liver_mask=liver_mask,
                tumor_mask=tumor_mask,
                use_filled_liver=True,
                liver_filled=liver_filled,
            )
            pt_path = os.path.basename(pt_path)
            row: Dict = {
                "case_name": case_name,
                "source_pt": pt_path,
                "pred_liver_voxels": int(liver_mask.sum().item()),
                "pred_tumor_voxels": int(tumor_mask.sum().item()),
                "bbox": bbox_to_dict(bbox),
            }

            if label is not None:
                gt = label[0].long()  # [D,H,W], 0/1/2
                gt_liver = gt > 0
                gt_tumor = gt == 2

                liver_metrics = compute_metrics(final_pred > 0, gt_liver)
                tumor_metrics = compute_metrics(final_pred == 2, gt_tumor)

                row["liver_dice"] = round(liver_metrics["Dice"], 4)
                if args.eval_liver_filled:
                    liver_filled_metrics = compute_metrics(liver_filled, gt_liver)
                    row["liver_filled_dice"] = round(liver_filled_metrics["Dice"], 4)
                row["tumor_dice"] = round(tumor_metrics["Dice"], 4)
                row["tumor_jaccard"] = round(tumor_metrics["Jaccard"], 4)
                row["tumor_recall"] = round(tumor_metrics["Recall"], 4)
                row["tumor_FDR"] = round(tumor_metrics["FDR"], 4)
                row["tumor_precision"] = round(tumor_metrics["Precision"], 4)
                row["has_tumor_gt"] = bool(gt_tumor.any().item())

                gt_tv = int(gt_tumor.sum().item())
                gt_lv = int(gt_liver.sum().item())
                row["gt_tumor_voxels"] = gt_tv
                row["gt_liver_voxels"] = gt_lv
                # 肿瘤大小分级：无肿瘤/极小/小/中等/大
                if gt_tv == 0:
                    row["tumor_size_cat"] = "无肿瘤"
                elif gt_tv < 5000:
                    row["tumor_size_cat"] = "极小(<5k)"
                elif gt_tv < 50000:
                    row["tumor_size_cat"] = "小(5k-50k)"
                elif gt_tv < 300000:
                    row["tumor_size_cat"] = "中等(50k-300k)"
                else:
                    row["tumor_size_cat"] = "大(>=300k)"

                liver_metrics_list.append(liver_metrics)
                if args.eval_liver_filled:
                    liver_filled_metrics_list.append(liver_filled_metrics)
                if gt_tumor.any().item():
                    tumor_metrics_list.append(tumor_metrics)  # 有肿瘤 case
                else:
                    tumor_metrics_list_neg.append(tumor_metrics)  # 无肿瘤 case

            rows.append(row)
            if args.vis_all or case_idx <= args.vis_n:
                save_case_visualization(
                    save_path=os.path.join(vis_dir, f"{case_name}.png"),
                    image=image,
                    label=label,
                    pred1=pred1_filtered,
                    tumor_full=tumor_mask.long(),
                    final_pred=final_pred,
                    case_name=case_name,
                    liver_filled=liver_filled if args.eval_liver_filled else None,
                )


            msg = f"[{case_idx}/{len(pt_paths)}] {case_name}"
            if "liver_dice" in row:
                msg += f" liver={row['liver_dice']:.4f} tumor={row['tumor_dice']:.4f}"
            print(msg)

    elapsed_hours = (time.time() - time_start) / 3600.0

    # -----------------------------
    # save per-case csv
    # -----------------------------
    csv_path = os.path.join(workdir, "per_case.csv")
    fieldnames = [
        "case_name",
        "source_pt",
        "pred_liver_voxels",
        "pred_tumor_voxels",
        "liver_dice",
        "liver_filled_dice",
        "tumor_dice",
        "bbox",
        "tumor_jaccard",
        "tumor_recall",
        "tumor_FDR",
        "tumor_precision",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            rr = dict(r)
            rr["bbox"] = (
                json.dumps(rr["bbox"], ensure_ascii=False)
                if rr["bbox"] is not None
                else ""
            )
            writer.writerow({k: rr.get(k, "") for k in fieldnames})

    # -----------------------------
    # metrics
    # -----------------------------
    metrics = {
        "split": args.split,
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "n_cases": len(rows),
        "device": device,
        "elapsed_hours": round(elapsed_hours, 3),
        "liver": summarize_metrics_list(liver_metrics_list, ["Dice"]),
        "liver_filled": summarize_metrics_list(liver_filled_metrics_list, ["Dice"])
        if liver_filled_metrics_list
        else None,
        # 有肿瘤 case：主指标，对齐 nnUNet 报告口径
        "tumor_pos": summarize_metrics_list(
            tumor_metrics_list, ["Dice", "Jaccard", "Recall", "FDR", "FNR", "Precision"]
        ),
        # 无肿瘤 case：统计误报率(FP rate)，模型把无肿瘤预测成有肿瘤的比例
        "tumor_neg_false_positive_rate": round(
            sum(1 for m in tumor_metrics_list_neg if m["FP"] > 0)
            / max(1, len(tumor_metrics_list_neg)),
            4,
        )
        if tumor_metrics_list_neg
        else None,
        "tumor_neg_n": len(tumor_metrics_list_neg),
    }

    with open(os.path.join(workdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(workdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("Two-Stage Evaluation Report\n")
        f.write("===========================\n")
        f.write(f"workdir: {workdir}\n")
        f.write(f"split: {metrics['split']}\n")
        f.write(f"seed: {metrics['seed']}\n")
        f.write(f"val_ratio: {metrics['val_ratio']}\n")
        f.write(f"test_ratio: {metrics['test_ratio']}\n")
        f.write(f"n_cases: {metrics['n_cases']}\n")
        f.write(f"device: {metrics['device']}\n")
        f.write(f"elapsed_hours: {metrics['elapsed_hours']}\n\n")
        f.write("Liver (stage1_liver)\n")
        for metric_name, summary in metrics["liver"].items():
            f.write(f"  {metric_name}: mean={summary['mean']} std={summary['std']}\n")
        if metrics.get("liver_filled"):
            f.write("Liver (liver_filled, 孔洞填充后)\n")
            for metric_name, summary in metrics["liver_filled"].items():
                f.write(f"  {metric_name}: mean={summary['mean']} std={summary['std']}\n")
        f.write("\n")

        f.write(
            f"Tumor (有肿瘤 case, n={metrics['tumor_pos'].get('Dice', {}).get('n', '?')})\n"
        )
        for metric_name, summary in metrics["tumor_pos"].items():
            f.write(f"  {metric_name}: mean={summary['mean']} std={summary['std']}\n")
        f.write("\n")

        neg_n = metrics.get("tumor_neg_n", 0)
        fpr = metrics.get("tumor_neg_false_positive_rate")
        f.write(f"Tumor (无肿瘤 case, n={neg_n})\n")
        f.write(f"  误报率(预测出肿瘤但GT无肿瘤): {fpr}\n")
        f.write("  说明: 误报率=0表示模型对所有无肿瘤case都正确预测为阴性\n")
        f.write("\n")

        # per-case 分级表，按 tumor_dice 从低到高排列
        pos_rows = [
            r for r in rows if r.get("has_tumor", True) and r["case_name"] != "liver_87"
        ]

        # 兼容没有 has_tumor 字段的情况：用 gt_tumor_voxels 或直接看 tumor_dice==1.0 判无肿瘤
        # 这里用 tumor_dice==1.0 且 pred_tumor_voxels==0 排除无肿瘤 case
        def is_no_tumor(r):
            return (
                float(r.get("tumor_dice", 0)) == 1.0
                and int(r.get("pred_tumor_voxels", 0)) == 0
            )

        pos_rows = [r for r in rows if not is_no_tumor(r)]
        pos_rows.sort(key=lambda r: float(r.get("tumor_dice", 0)))

        critical = [r for r in pos_rows if float(r["tumor_dice"]) < 0.3]
        needs_work = [r for r in pos_rows if 0.3 <= float(r["tumor_dice"]) < 0.7]
        good = [r for r in pos_rows if float(r["tumor_dice"]) >= 0.7]

        sep = "-" * 80
        header = f"  {'case':<12} {'tumor_dice':>10} {'recall':>8} {'precision':>10} {'FDR':>8} {'pred_tumor':>12} {'gt_tumor':>10} {'gt_liver':>10} {'size_cat':<14}"

        f.write("=" * 80 + "\n")
        f.write("Per-Case 分级(按 tumor_dice 从低到高)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"[严重失败] tumor_dice < 0.3  (n={len(critical)})\n")
        f.write(sep + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for r in critical:
            f.write(
                f"  {r['case_name']:<12} {float(r['tumor_dice']):>10.4f} "
                f"{float(r['tumor_recall']):>8.4f} {float(r['tumor_precision']):>10.4f} "
                f"{float(r['tumor_FDR']):>8.4f} {int(r['pred_tumor_voxels']):>12,} "
                f"{int(r.get('gt_tumor_voxels', 0)):>10,} {int(r.get('gt_liver_voxels', 0)):>10,} "
                f"{r.get('tumor_size_cat', ''):14}\n"
            )
        if not critical:
            f.write("  (无)\n")
        f.write("\n")

        f.write(f"[需要改进] 0.3 <= tumor_dice < 0.7  (n={len(needs_work)})\n")
        f.write(sep + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for r in needs_work:
            f.write(
                f"  {r['case_name']:<12} {float(r['tumor_dice']):>10.4f} "
                f"{float(r['tumor_recall']):>8.4f} {float(r['tumor_precision']):>10.4f} "
                f"{float(r['tumor_FDR']):>8.4f} {int(r['pred_tumor_voxels']):>12,} "
                f"{int(r.get('gt_tumor_voxels', 0)):>10,} {int(r.get('gt_liver_voxels', 0)):>10,} "
                f"{r.get('tumor_size_cat', ''):14}\n"
            )
        if not needs_work:
            f.write("  (无)\n")
        f.write("\n")

        f.write(f"[没问题]   tumor_dice >= 0.7  (n={len(good)})\n")
        f.write(sep + "\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for r in good:
            f.write(
                f"  {r['case_name']:<12} {float(r['tumor_dice']):>10.4f} "
                f"{float(r['tumor_recall']):>8.4f} {float(r['tumor_precision']):>10.4f} "
                f"{float(r['tumor_FDR']):>8.4f} {int(r['pred_tumor_voxels']):>12,} "
                f"{int(r.get('gt_tumor_voxels', 0)):>10,} {int(r.get('gt_liver_voxels', 0)):>10,} "
                f"{r.get('tumor_size_cat', ''):14}\n"
            )
        if not good:
            f.write("  (无)\n")
        f.write("\n")

        # Stage2 概率图诊断表
        if stage2_prob_diag:
            f.write("=" * 80 + "\n")
            f.write("Stage2 概率图诊断 (prob_max 越低说明模型在该 case 上越没有信号)\n")
            f.write("=" * 80 + "\n")
            diag_header = f"  {'case':<12} {'prob_max':>9} {'prob_mean':>10} {'>0.1vox':>9} {'>0.3vox':>9} {'>0.5vox':>9}\n"
            f.write(diag_header)
            f.write("-" * 65 + "\n")
            for d in sorted(stage2_prob_diag, key=lambda x: x["prob_max"]):
                f.write(
                    f"  {d['case_name']:<12} {d['prob_max']:>9.3f} {d['prob_mean']:>10.5f} "
                    f"{d['vox_01']:>9,} {d['vox_03']:>9,} {d['vox_05']:>9,}\n"
                )
            f.write("\n")

    # prob_pt：只保存 tumor_dice 最低的 N 个 case
    N_SAVE_PROB = 5
    sorted_rows = sorted(rows, key=lambda r: float(r.get("tumor_dice", 1.0)))
    os.makedirs(prob_dir, exist_ok=True)
    for r in sorted_rows[:N_SAVE_PROB]:
        cname = r["case_name"]
        if cname in prob_buffer:
            torch.save(prob_buffer[cname], os.path.join(prob_dir, f"{cname}_prob.pt"))
    prob_buffer.clear()
    print(f"[prob_pt] saved worst {min(N_SAVE_PROB, len(sorted_rows))} cases to {prob_dir}")

    # visualize worst 3 cases by tumor_dice
    vis_worst_cases(workdir, rows, n_worst=3, preprocessed_root=args.preprocessed_root)

    print("\n===== Final Metrics =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {workdir}")


if __name__ == "__main__":
    main()
