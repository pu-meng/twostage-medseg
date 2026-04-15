from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
import warnings as _w
from pathlib import Path as _Path

# Add medseg_project to path so `medseg` package can be found
_medseg_project = str(_Path(__file__).resolve().parents[2] / "medseg_project")
if _medseg_project not in sys.path:
    sys.path.insert(0, _medseg_project)

from monai.inferers.utils import sliding_window_inference
from twostage_medseg.metrics.filter import filter_largest_component
from twostage_medseg.twostage.roi_utils import compute_bbox_from_mask

import torch
from torch.utils.data import DataLoader
from monai.data.utils import list_data_collate
from medseg.data.dataset_offline import load_pt_paths, split_two_with_monitor
from medseg.data.transforms_offline import (
    build_train_transforms,
    build_val_transforms,
)
from medseg.engine.train_eval import (
    train_one_epoch_sigmoid_binary,
    validate_sliding_window,
)
from medseg.models.build_model import build_model

from medseg.engine.adaptive_loss import (
    train_one_epoch_binary_learnable,
    LearnableWeightedLoss,
)

from twostage_medseg.metrics.DiagLogger import DiagLogger
from medseg.utils.ckpt import (
    load_ckpt_full,
    save_ckpt_full,
)

from medseg.utils.io_utils import ensure_dir, save_cmd, save_json, save_report

from medseg.utils.train_utils import set_seed
from twostage_medseg.twostage.train_logger import TrainLoggerTwoStage
from twostage_medseg.twostage.dataset_tumor_roi import TumorROIDataset
from twostage_medseg.twostage.train_eval_tumor import tumor_metrics_from_val_result

"""
os.path.abspath(medseg_root)
把相对路径变为绝对路径; 比如
输入: medseg
输出: /home/pumengyu/medseg

sys.path是Python的模块搜索路径列表, import 语句会按照顺序在这些目录里面查找模块

sys.path.insert(0, medseg_root)是为了把medseg_root插入到列表的第0位

margin是肿瘤裁剪向外扩展的像素数
"""


def add_medseg_to_syspath(medseg_root: str) -> None:
    medseg_root = os.path.abspath(medseg_root)
    if medseg_root not in sys.path:
        sys.path.insert(0, medseg_root)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--medseg_root", type=str, required=True, help="medseg的根目录")
    p.add_argument(
        "--preprocessed_root", type=str, required=True, help="预处理后的数据集路径"
    )
    p.add_argument(
        "--exp_root",
        type=str,
        default="/home/PuMengYu/MSD_LiverTumorSeg/experiments",
        help="输出结果的保存路径",
    )
    p.add_argument("--exp_name", type=str, default="tumor_roi_dynunet")
    p.add_argument("--model", type=str, default="dynunet")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="指定总梯度步数。传入后自动覆盖 --epochs, "
        "epochs = ceil(total_steps / steps_per_epoch), "
        "steps_per_epoch = len(train_dataset) // batch_size。"
        "用于跨实验对齐训练量 (不同 repeats/scale 时步数/epoch 不同)。",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument(
        "--patch",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="训练用的 patch 大小，默认 96 96 96",
    )
    p.add_argument(
        "--val_patch",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="验证用的 patch 大小，默认 96 96 96",
    )

    p.add_argument(
        "--sw_batch_size", type=int, default=1, help="只在验证时候用,训练不用"
    )
    p.add_argument(
        "--val_overlap",
        type=float,
        default=0.0,
        help="验证时 sliding window 的 overlap, 0=无重叠最快, 0.25/0.5 更精确但慢",
    )
    p.add_argument(
        "--bbox_overlap",
        type=float,
        default=0.25,
        help="build_pred_bboxes 时 Stage1 推理的 overlap, 仅 --use_pred_bbox 时生效",
    )
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="表示每个worker提前预加载的batch数量,只在num_workers>0时有效;会占用CPU内存,不影响GPU的显存",
    )

    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="训练用的数据重复次数,默认3,可以通过增加这个值来增强数据量较少时的训练效果",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="store_true表示这个参数出现就是True,不出现为False",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="dicece",
        choices=["dicece", "dicefocal", "tversky", "focaltversky"],
    )
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_n", type=int, default=0)
    p.add_argument("--val_n", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="初始化模型权重路径 (热启动)。"
        "不传则从随机初始化开始训练; "
        "第一次训练可传 Stage1 肝脏模型的 best.pt 做迁移学习; "
        "后续迭代传上一次肿瘤模型的 best.pt 继续积累。"
        "注意: 训练时 ROI 裁剪始终用 GT 真实肝脏标签, 不需要 Stage1 模型参与; "
        "Stage1 肝脏模型只在 eval/infer 阶段才真正用到 (推理时无 GT, 靠 Stage1 预测肝脏位置)。"
        "只加载模型权重, optimizer/scheduler/epoch 全部重置, 可自由修改 LR 和训练策略。",
    )
    p.add_argument("--tumor_ratios", type=float, nargs=2, default=[0.2, 0.8])
    p.add_argument("--margin", type=int, default=12)
    # parse_args() 里新增一个参数
    p.add_argument(
        "--learnable_loss",
        action="store_true",
        help="启用可学习权重损失(alpha自动学习liver/tumor平衡)",
    )
    p.add_argument(
        "--bbox_jitter",
        action="store_true",
        help="对肝脏 bbox 施加随机扰动, 模拟 Stage1 预测框的偏差, 增强鲁棒性",
    )
    p.add_argument(
        "--bbox_max_shift",
        type=int,
        default=8,
        help="bbox 扰动的最大偏移量 (体素), 仅在 --bbox_jitter 开启时生效, 默认 8",
    )
    p.add_argument(
        "--random_margin",
        action="store_true",
        help="对 ROI margin 进行随机采样, 模拟不同裁剪尺度, 增强泛化能力",
    )
    p.add_argument(
        "--margin_min",
        type=int,
        default=8,
        help="随机 margin 的最小值 (体素), 仅在 --random_margin 开启时生效, 默认 8",
    )
    p.add_argument(
        "--margin_max",
        type=int,
        default=20,
        help="随机 margin 的最大值 (体素), 仅在 --random_margin 开启时生效, 默认 20",
    )
    # 差异化 oversampling: 小肿瘤/无肿瘤 hard case mining
    p.add_argument(
        "--small_tumor_thresh",
        type=int,
        default=0,
        help="小肿瘤阈值(voxels),0=关闭 hard mining",
    )
    p.add_argument(
        "--small_tumor_repeat_scale",
        type=int,
        default=3,
        help="小肿瘤 case 的 repeat 倍率",
    )
    p.add_argument(
        "--no_tumor_repeat_scale",
        type=int,
        default=4,
        help="无肿瘤 case 的 repeat 倍率",
    )
    p.add_argument(
        "--large_tumor_thresh",
        type=int,
        default=0,
        help="大肿瘤阈值(voxels),0=关闭大肿瘤过采样",
    )
    p.add_argument(
        "--large_tumor_repeat_scale",
        type=int,
        default=3,
        help="大肿瘤 case 的 repeat 倍率",
    )
    p.add_argument(
        "--small_tumor_zoom_thresh",
        type=int,
        default=0,
        help="小肿瘤 zoom-in 阈值(体素数)：肿瘤体素数 < 此值时做 zoom-in 放大，0=关闭。建议值 5000。",
    )
    p.add_argument(
        "--small_tumor_zoom_factor",
        type=float,
        default=2.0,
        help="zoom-in 放大倍率，2.0=放大2倍（ROI物理范围缩小一半）。",
    )

    p.add_argument(
        "--two_channel",
        action="store_true",
        help="启用两通道输入: Ch1=CT, Ch2=liver mask (训练时用 GT, 推理时用 Stage1 预测)",
    )
    p.add_argument(
        "--use_coarse_tumor",
        action="store_true",
        help="启用粗糙肿瘤通道: Ch1=CT, Ch2=Stage1预测粗糙肿瘤mask, 需配合 out_channels=3 的 Stage1 ckpt",
    )
    p.add_argument(
        "--stage1_out_channels",
        type=int,
        default=2,
        help="Stage1 输出通道数: 2=仅肝脏 (旧版), 3=肝脏+肿瘤 (新版)",
    )

    # 预测 bbox 模式: 用 Stage1 推理结果代替 GT bbox, 消除训练/推理 domain gap
    p.add_argument(
        "--use_pred_bbox",
        action="store_true",
        help="启用后, 训练集 bbox 由 Stage1 推理生成, 与推理阶段完全对齐, 关闭此开关则保持原 GT bbox + jitter 行为",
    )
    p.add_argument(
        "--stage1_ckpt",
        type=str,
        default=None,
        help="Stage1 肝脏模型路径, --use_pred_bbox 时必须提供",
    )
    p.add_argument(
        "--stage1_model",
        type=str,
        default="dynunet",
        help="Stage1 模型类型, 默认 dynunet, 与 --use_pred_bbox 配合使用",
    )
    p.add_argument(
        "--stage1_patch",
        type=int,
        nargs=3,
        default=[144, 144, 144],
        help="Stage1 推理 patch 大小，默认 144 144 144",
    )
    p.add_argument(
        "--pred_bbox_cache",
        type=str,
        default=None,
        help="pred bbox JSON 缓存路径。存在则直接读取跳过推理, 不存在则推理后自动保存, 不传则每次重新推理",
    )

    return p.parse_args()


def load_init_weights(path, model, map_location="cpu"):
    """

        ckpt=checkpoint,src=source,dst=destination
        ckpt是训练过程中的保存的快照,可以包含模型参数,优化器状态,训练轮数,loss值
        model是PyTorch的nn.Module对象,比如model=UNet(),model=ResNet50()
        dst=model.state_dict()
        返回的dst类似于
        {
        "encoder.conv1.weiots":tensor([...]),
        "encoder.conv1.bias":tensor([...]),
        "encoder.conv2.weioets":tensor([...]),
        ...
        }
        如果src是字典,那么src.items()就是字典的键值对迭代器,遍历会得到
        ("encoder.conv1.weiots",tensor([...])),("encoder.conv1.bias",tensor([...])),...

            model是当前模型参数
            src是checkpoint参数
            dst.update(matched)
            这个是把匹配的参数覆盖到当前模型参数
    从 path 加载 checkpoint
        ↓
    筛选出「名字相同 且 形状相同」的参数
        ↓
    把这些参数复制到当前 model 里
        ↓
    其他不匹配的层保持原样 (随机初始化或已有的值)

    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # 去掉_orig_mod.前缀,兼容某些checkpoint里参数名带有_orig_mod.的情况
    if any(k.startswith("_orig_mod.") for k in src):
        src = {k.removeprefix("_orig_mod."): v for k, v in src.items()}

    dst = model.state_dict()
    matched = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    # dynunet_ca 的 backbone 参数名带 backbone. 前缀，裸 ckpt 里没有，尝试加前缀重映射
    if len(matched) == 0:
        remapped = {"backbone." + k: v for k, v in src.items()}
        matched = {
            k: v for k, v in remapped.items() if k in dst and dst[k].shape == v.shape
        }
    dst.update(matched)
    model.load_state_dict(dst, strict=False)
    print(f"[init] loaded {len(matched)} matched params from {path}")


def build_pred_bboxes(
    stage1_ckpt,
    pt_paths,
    device,
    stage1_model,
    stage1_patch,
    overlap,
    cache_path=None,
    stage1_out_channels=2,
):
    """
    用 Stage1 模型对训练集推理, 返回每个 case 的预测 tight bbox 字典。

    stage1_out_channels=2 (旧版, 只输出肝脏):
        返回格式: {case_name: (z0, z1, y0, y1, x0, x1)}  -- 肝脏 bbox

    stage1_out_channels=3 (新版, 输出肝脏+粗糙肿瘤):
        返回格式: {case_name: {"liver": [...], "tumor": [...]}}
        tumor 值为 None 表示该 case Stage1 未检出肿瘤。

    cache_path: JSON 缓存路径。
        - 文件已存在 -> 直接读取, 跳过推理
        - 文件不存在 -> 推理完后自动保存
        - None       -> 纯内存, 不读写磁盘

    旧缓存格式 (list of 6 int) 自动识别并以旧格式返回, 向后兼容。
    运行完毕后立即释放 Stage1 模型, 不占用后续训练的显存。
    """

    # 命中缓存: 直接读 JSON, 跳过推理
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 自动识别新旧格式
        first = next(iter(raw.values()))
        if isinstance(first, list):
            pred_bboxes = {k: tuple(v) for k, v in raw.items()}
        else:
            pred_bboxes = {
                k: {
                    "liver": tuple(v["liver"]),
                    "tumor": tuple(v["tumor"]) if v["tumor"] is not None else None,
                }
                for k, v in raw.items()
            }
        print(f"[pred_bbox] loaded from cache ({len(pred_bboxes)} cases): {cache_path}")
        return pred_bboxes

    stage1 = build_model(
        stage1_model,
        in_channels=1,
        out_channels=stage1_out_channels,
        img_size=tuple(stage1_patch),
    ).to(device)
    load_init_weights(stage1_ckpt, stage1, map_location=device)
    stage1.eval()

    pred_bboxes = {}
    print(
        f"[pred_bbox] Stage1 (out_channels={stage1_out_channels}) generating predicted bboxes "
        f"for {len(pt_paths)} cases ..."
    )

    with torch.no_grad():
        for i, pt_path in enumerate(pt_paths):
            case_name = _Path(pt_path).stem
            with _w.catch_warnings():
                _w.simplefilter("ignore", FutureWarning)
                data = torch.load(
                    pt_path, map_location="cpu", weights_only=False, mmap=True
                )

            image = data["image"].float()
            x = image.unsqueeze(0).to(device)  # [1,1,D,H,W]

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")
            ):
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=tuple(stage1_patch),
                    sw_batch_size=1,
                    predictor=stage1,
                    overlap=overlap,
                )

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            pred = torch.argmax(logits.float(), dim=1)[0].cpu() #type:ignore
            liver_mask = pred == 1
            liver_mask = filter_largest_component(liver_mask)

            if liver_mask.sum().item() == 0:
                gt_liver = data["label"][0] > 0
                liver_bbox = compute_bbox_from_mask(gt_liver.bool(), margin=0)
                print(f"  [warn] {case_name}: Stage1 missed liver, fallback to GT bbox")
            else:
                liver_bbox = compute_bbox_from_mask(liver_mask, margin=0)

            if stage1_out_channels == 3:
                tumor_mask = pred == 2
                if tumor_mask.sum().item() == 0:
                    tumor_bbox = None
                else:
                    tumor_bbox = compute_bbox_from_mask(tumor_mask, margin=0)
                pred_bboxes[case_name] = {"liver": liver_bbox, "tumor": tumor_bbox}
            else:
                pred_bboxes[case_name] = liver_bbox

            if (i + 1) % 20 == 0 or (i + 1) == len(pt_paths):
                print(f"  [{i + 1}/{len(pt_paths)}] done")

    del stage1
    torch.cuda.empty_cache()
    print(f"[pred_bbox] done, {len(pred_bboxes)} bboxes generated")

    # 保存缓存
    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

        def _serialize(v):
            if isinstance(v, tuple):
                return list(v)
            return {
                "liver": list(v["liver"]),
                "tumor": list(v["tumor"]) if v["tumor"] is not None else None,
            }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({k: _serialize(v) for k, v in pred_bboxes.items()}, f, indent=2)
        print(f"[pred_bbox] cache saved to: {cache_path}")

    return pred_bboxes


def build_coarse_tumor_cache(
    stage1_ckpt: str,
    pt_paths: list,
    device: str,
    stage1_model: str = "dynunet",
    stage1_patch: tuple = (144, 144, 144),
    stage1_out_channels: int = 3,
    overlap: float = 0.5,
) -> dict:
    """
    用 Stage1 模型对 pt_paths 推理，返回每个 case 的 tumor softmax 软概率。
    格式: {case_name: Tensor[1, D, H, W] float}，坐标系与 .pt 文件对齐（已做 crop_bbox 偏移）。
    验证集用此 cache 作为 Ch2，替代 GT mask，消除 train/val 分布差异。
    """
    from medseg.models.build_model import build_model
    from medseg.utils.ckpt import load_init_weights

    if stage1_out_channels != 3:
        print("[coarse_tumor_cache] stage1_out_channels!=3, 无法生成软概率, 返回空 cache")
        return {}

    stage1 = build_model(
        stage1_model,
        in_channels=1,
        out_channels=stage1_out_channels,
        img_size=tuple(stage1_patch),
    ).to(device)
    load_init_weights(stage1_ckpt, stage1, map_location=device)
    stage1.eval()

    cache = {}
    print(f"[coarse_tumor_cache] 生成验证集 Stage1 软概率 cache，共 {len(pt_paths)} 个 case ...")
    with torch.no_grad():
        for i, pt_path in enumerate(pt_paths):
            case_name = _Path(pt_path).stem
            with _w.catch_warnings():
                _w.simplefilter("ignore", FutureWarning)
                data = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)

            image = data["image"].float()  # [1, D, H, W]
            x = image.unsqueeze(0).to(device)  # [1, 1, D, H, W]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=tuple(stage1_patch),
                    sw_batch_size=1,
                    predictor=stage1,
                    overlap=overlap,
                )
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            # 取 channel 2（tumor）的 softmax 软概率，[D, H, W] → [1, D, H, W]
            prob = torch.softmax(logits.float(), dim=1)[0, 2].cpu().unsqueeze(0)  # [1, D, H, W]
            cache[case_name] = prob

            if (i + 1) % 5 == 0 or (i + 1) == len(pt_paths):
                print(f"  [{i + 1}/{len(pt_paths)}] done")

    del stage1
    torch.cuda.empty_cache()
    print(f"[coarse_tumor_cache] done, {len(cache)} cases")
    return cache


def pt_has_tumor(pt_path: str) -> bool:
    """
    判断一个 .pt case 是否含有 tumor
    原始 Task03 label: 0=bg, 1=liver, 2=tumor
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
    label = data.get("label", None)
    if label is None:
        return False
    label = label.long()
    return bool((label == 2).any().item())


def filter_tumor_positive_cases(pt_paths):
    """
    筛选出含有肿瘤的 case的路径列表
    和不含肿瘤的case的路径列表

    """
    pos = []
    neg = []
    for p in pt_paths:
        if pt_has_tumor(p):
            pos.append(p)
        else:
            neg.append(p)
    return pos, neg


def main():
    args = parse_args()
    add_medseg_to_syspath(args.medseg_root)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True  # 固定patch尺寸时自动选最快卷积算法
    # 减少显存碎片化，避免因碎片导致的 OOM 或 cudaMalloc 停顿
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    all_pt = load_pt_paths(args.preprocessed_root)
    tr, va, te = split_two_with_monitor(all_pt)  # train=112, monitor=12(子集), test=19
    # tr,va,te都是路径列表

    tr_all, _, te_all = tr, va, te

    if args.train_n > 0:
        tr = tr[: args.train_n]

    # 预测 bbox 模式: 训练开始前用 Stage1 对训练集推理一次, 生成 bbox 字典
    # 训练集和验证集分开处理: 验证集始终用 GT bbox (eval 阶段不存在 domain gap 问题)
    pred_bboxes_train = None
    if args.use_pred_bbox:
        if not args.stage1_ckpt:
            raise ValueError("--use_pred_bbox 需要同时提供 --stage1_ckpt")
        _device_bbox = "cuda" if torch.cuda.is_available() else "cpu"
        pred_bboxes_train = build_pred_bboxes(
            stage1_ckpt=args.stage1_ckpt,
            pt_paths=tr,
            device=_device_bbox,
            stage1_model=args.stage1_model,
            stage1_patch=args.stage1_patch,
            overlap=args.bbox_overlap,
            cache_path=args.pred_bbox_cache,
            stage1_out_channels=args.stage1_out_channels,
        )

    # tr_pos是有肿瘤的路径列表，tr_neg是没有肿瘤的路径列表
    te_pos, te_neg = filter_tumor_positive_cases(te)
    tr_pos_cur, tr_neg_cur = filter_tumor_positive_cases(tr)
    va_pos_cur, va_neg_cur = filter_tumor_positive_cases(va)

    print(f"[原始划分] train={len(tr)} monitor={len(va)} test={len(te)}")
    print(
        f"[有肿瘤部分] train={len(tr_pos_cur)} monitor={len(va_pos_cur)} test={len(te_pos)}"
    )
    print(
        f"[无肿瘤部分] train={len(tr_neg_cur)} monitor={len(va_neg_cur)} test={len(te_neg)}"
    )

    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    workdir = os.path.join(args.exp_root, args.exp_name, "train", timestamp)
    ensure_dir(workdir)
    save_cmd(workdir)

    config = {
        "stage": "tumor_roi",
        "medseg_root": os.path.abspath(args.medseg_root),
        "preprocessed_root": os.path.abspath(args.preprocessed_root),
        "init_ckpt": args.init_ckpt,
        "exp_root": args.exp_root,
        "exp_name": args.exp_name,
        "workdir": workdir,
        "model": args.model,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "patch": list(args.patch),
        "val_patch": list(args.val_patch),
        "sw_batch_size": int(args.sw_batch_size),
        "val_overlap": float(args.val_overlap),
        "bbox_overlap": float(args.bbox_overlap),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "repeats": int(args.repeats),
        "total_steps": args.total_steps,  # None 表示未指定，按 epochs 走
        "amp": bool(args.amp),
        "loss": args.loss,
        "val_every": int(args.val_every),
        "seed": int(args.seed),
        "tumor_ratios": list(args.tumor_ratios),
        "margin": int(args.margin),
        "n_train_cases": len(tr),
        "n_monitor_cases": len(va),
        "n_test_cases": len(te),
        "num_classes": 2,
        "optimizer": "SGD_momentum0.99_wd3e-5_nesterov",
        "scheduler": "PolyLR_power0.9",
        "T_max": int(args.epochs),
        "in_channels": 2 if (args.two_channel or args.use_coarse_tumor) else 1,
        "use_coarse_tumor": bool(args.use_coarse_tumor),
        "out_channels": 2,
        "two_channel": bool(args.two_channel),
        "resume": args.resume,
        "learnable_loss": bool(args.learnable_loss),
        "bbox_jitter": bool(args.bbox_jitter),
        "bbox_max_shift": int(args.bbox_max_shift),
        "random_margin": bool(args.random_margin),
        "margin_min": int(args.margin_min),
        "margin_max": int(args.margin_max),
        "small_tumor_thresh": int(args.small_tumor_thresh),
        "small_tumor_repeat_scale": int(args.small_tumor_repeat_scale),
        "no_tumor_repeat_scale": int(args.no_tumor_repeat_scale),
        "large_tumor_thresh": int(args.large_tumor_thresh),
        "large_tumor_repeat_scale": int(args.large_tumor_repeat_scale),
        "small_tumor_zoom_thresh": int(args.small_tumor_zoom_thresh),
        "small_tumor_zoom_factor": float(args.small_tumor_zoom_factor),
        "use_pred_bbox": bool(args.use_pred_bbox),
        "stage1_ckpt": args.stage1_ckpt,
        "stage1_model": args.stage1_model,
        "stage1_patch": list(args.stage1_patch),
        "pred_bbox_cache": args.pred_bbox_cache,
    }
    save_json(config, workdir, "config")
    # 初始化诊断日志器，结果写入 workdir/diag.txt 和 diag_summary.txt
    diag = DiagLogger(workdir)

    # 记录数据集划分统计：train/val/test 各有多少 case，其中含肿瘤的有多少
    diag.log_dataset(args, tr, va, te, tr_pos_cur, va_pos_cur)

    # 数据泄漏检查：monitor是train子集，只检查train/test无重叠
    diag.check_data_leakage(tr_all, [], te_all)

    # 抽样检查标签分布：统计前 N 个 case 里 background/liver/tumor 各占多少体素
    diag.log_label_stats(tr, tag="train", max_cases=10)
    diag.log_label_stats(va, tag="monitor", max_cases=5)

    # 抽样检查肝脏 ROI 裁剪后的尺寸分布，确认 patch 能覆盖大多数 ROI
    diag.log_roi_stats(tr, tag="train", max_cases=10, tumor_label=2)

    train_tf = build_train_transforms(
        tuple(args.patch), ratios=tuple(args.tumor_ratios)
    )
    val_tf = build_val_transforms()

    train_ds = TumorROIDataset(
        tr,
        transform=train_tf,
        repeats=args.repeats,
        margin=args.margin,
        keep_meta=False,
        bbox_jitter=args.bbox_jitter,
        bbox_max_shift=args.bbox_max_shift,
        random_margin=args.random_margin,
        margin_min=args.margin_min,
        margin_max=args.margin_max,
        small_tumor_thresh=args.small_tumor_thresh,
        small_tumor_repeat_scale=args.small_tumor_repeat_scale,
        no_tumor_repeat_scale=args.no_tumor_repeat_scale,
        large_tumor_thresh=args.large_tumor_thresh,
        large_tumor_repeat_scale=args.large_tumor_repeat_scale,
        pred_bboxes=pred_bboxes_train,  # None 时行为与之前完全一致
        two_channel=args.two_channel,
        use_coarse_tumor=args.use_coarse_tumor,
        small_tumor_zoom_thresh=args.small_tumor_zoom_thresh,
        small_tumor_zoom_factor=args.small_tumor_zoom_factor,
    )

    # --total_steps：自动反推 epochs，对齐不同 repeats/scale 实验的训练量
    steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    if args.total_steps is not None:
        import math

        args.epochs = math.ceil(args.total_steps / steps_per_epoch)
        print(
            f"[total_steps] total_steps={args.total_steps}, "
            f"steps_per_epoch={steps_per_epoch}, "
            f"=> epochs={args.epochs}"
        )
    # 回写 config（epochs 可能被 total_steps 覆盖，steps_per_epoch 需要 dataset 才能算）
    config["epochs"] = int(args.epochs)
    config["steps_per_epoch"] = steps_per_epoch
    config["T_max"] = int(args.epochs)

    # use_coarse_tumor 时为验证集生成 Stage1 软概率 cache，避免 val 用 GT 作 Ch2 导致指标虚高
    val_coarse_tumor_cache = None
    if args.use_coarse_tumor and args.stage1_ckpt:
        _device_cache = "cuda" if torch.cuda.is_available() else "cpu"
        val_coarse_tumor_cache = build_coarse_tumor_cache(
            stage1_ckpt=args.stage1_ckpt,
            pt_paths=va,
            device=_device_cache,
            stage1_model=args.stage1_model,
            stage1_patch=tuple(args.stage1_patch),
            stage1_out_channels=args.stage1_out_channels,
            overlap=args.bbox_overlap,
        )

    val_ds = TumorROIDataset(
        va,
        transform=val_tf,
        repeats=1,
        margin=args.margin,
        keep_meta=True,
        bbox_jitter=False,
        bbox_max_shift=0,
        random_margin=False,
        use_coarse_tumor=args.use_coarse_tumor,
        coarse_tumor_cache=val_coarse_tumor_cache,
    )
    # train_tf是训练时的数据增强，val_tf是验证时的数据增强
    # train_ds是训练集，val_ds是验证集,是TumorROIDataset的实例
    if args.num_workers > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
            collate_fn=list_data_collate,  # 支持 num_samples=2 展平为 batch_size*2
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=min(args.num_workers, 4),  # 验证集小，不需要全部 workers
            pin_memory=True,
            persistent_workers=False,  # val_every>1 时 workers 空转浪费 CPU，按需启动
            prefetch_factor=2,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=list_data_collate,  # 支持 num_samples=2 展平为 batch_size*2
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels = 2 if (args.two_channel or args.use_coarse_tumor) else 1
    model = build_model(
        args.model, in_channels=in_channels, out_channels=2, img_size=tuple(args.patch)
    ).to(device)

    torch._dynamo.config.suppress_errors = True #type:ignore # 保险:编译失败不崩溃

    # reduce-overhead: 减少 Python 调度开销，对固定 patch 尺寸效果好，首次慢但后续更快
    model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.99, weight_decay=3e-5, nesterov=True
    )
    if args.learnable_loss:
        criterion = LearnableWeightedLoss(base_loss_type=args.loss).to(device)
        criterion_optimizer = torch.optim.Adam(criterion.parameters(), lr=1e-3)
    else:
        criterion = None
        criterion_optimizer = None

    # Poly LR decay: lr = lr_init * (1 - epoch/epochs)^0.9，对齐nnUNet
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp and device == "cuda" else None

    # 提前构建 loss_fn，避免每 epoch 重建（节省少量 Python 开销）
    from medseg.engine.train_eval import build_loss_fn_binary
    _loss_fn = build_loss_fn_binary(args.loss) if not args.learnable_loss else None

    start_epoch = 1
    best = -1.0
    best_epoch = 0

    # 新增部分：加载初始化权重,这里加载的是分割肝脏的权重,相当于迁移学习
    # 不传--init_ckpt就是完全不用stage1权重,从头训练肿瘤,
    # 传了--init_ckpt就是用stage1权重初始化,然后训练分割肿瘤
    # 但是不论传还是不传--init_ckpt,这个肝脏ROI都是基于真实标签分割的
    # 验证阶段才是需要同时传stage1和stage2的权重的
    # 如果stage1和stage2 的模型都一样,可以传,但是不一样可以不传,影响都不大
    if args.init_ckpt:
        load_init_weights(args.init_ckpt, model, map_location=device)
    # 如果需要 resume 训练
    if args.resume:
        ckpt = load_ckpt_full(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best_metric", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        print(
            f"[resume] start_epoch={start_epoch} best={best:.4f} best_epoch={best_epoch}"
        )

    logger = TrainLoggerTwoStage(workdir)

    print(f"[tumor stage2] workdir: {workdir}")
    print(f"[tumor stage2] train={len(tr)} monitor={len(va)} test={len(te)}")
    print(f"[tumor stage2] device={device} model={args.model}")

    wall_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.learnable_loss:
            train_loss = train_one_epoch_binary_learnable(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                criterion_optimizer=criterion_optimizer,
                device=device,
                scaler=scaler,
                epoch=epoch,
                epochs=args.epochs,
            )
        else:
            train_loss = train_one_epoch_sigmoid_binary(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                loss_type=args.loss,
                epoch=epoch,
                epochs=args.epochs,
                loss_fn=_loss_fn,  # 复用预建的 loss_fn，避免每 epoch 重建
            )

        val_tumor_dice = None
        if epoch % args.val_every == 0:
            val_result = validate_sliding_window(
                model=model,
                loader=val_loader,
                device=device,
                roi_size=tuple(args.val_patch),
                sw_batch_size=args.sw_batch_size,
                num_classes=2,
                overlap=args.val_overlap,
                return_per_class=True,
            )
            metrics = tumor_metrics_from_val_result(val_result)
            val_tumor_dice = metrics["tumor_dice"]
            if val_tumor_dice > best:
                best = val_tumor_dice
                best_epoch = epoch
                save_ckpt_full(
                    os.path.join(workdir, "best.pt"),
                    model,
                    optimizer,
                    epoch,
                    best,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_epoch=best_epoch,
                )
        scheduler.step()
        save_ckpt_full(
            os.path.join(workdir, "last.pt"),
            model,
            optimizer,
            epoch,
            best,
            scheduler=scheduler,
            scaler=scaler,
            best_epoch=best_epoch,
        )

        logger.log(
            epoch=epoch,
            train_loss=float(train_loss),
            val_liver=float("nan"),
            val_tumor=float("nan") if val_tumor_dice is None else float(val_tumor_dice),
            best=float(best),
            lr=float(optimizer.param_groups[0]["lr"]),
        )
        w = criterion.get_weights() if args.learnable_loss and criterion else {}
        if args.learnable_loss and criterion is not None:
            logger.log_extra(epoch=epoch, w_liver=w["w_liver"], w_tumor=w["w_tumor"])
        if val_tumor_dice is not None:
            diag.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_tumor_dice=val_tumor_dice,
                best=best,
                w_liver=w.get("w_liver"),
                w_tumor=w.get("w_tumor"),
            )

    total_sec = time.time() - wall_start
    metrics = {
        "best_tumor_dice": round(float(best), 4),
        "best_epoch": int(best_epoch),
        "epochs": int(args.epochs),
        "n_train_cases": len(tr),
        "n_monitor_cases": len(va),
        "total_train_hours": round(total_sec / 3600.0, 2),
    }
    diag.log_final(
        best_tumor_dice=round(float(best), 4),
        best_epoch=best_epoch,
        total_hours=round(total_sec / 3600.0, 2),
    )
    save_json(metrics, workdir, "metrics")

    save_report(
        text="\n".join([f"{k}: {v}" for k, v in metrics.items()]),
        out_dir=workdir,
        filename="report.txt",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
