from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from medseg.data.dataset_offline import load_pt_paths, split_three_ways
from medseg.data.transforms_offline import (
    build_train_transforms,
    build_val_transforms,
)
from medseg.engine.train_eval import train_one_epoch_binary, validate_sliding_window
from medseg.models.build_model import build_model

from medseg.engine.adaptive_loss import (
    train_one_epoch_binary_learnable,
    LearnableWeightedLoss,
)


from medseg.utils.ckpt import (
    load_ckpt_full,
    save_ckpt_full,
)

from medseg.utils.io_utils import ensure_dir, save_cmd, save_json, save_report

from medseg.utils.train_utils import set_seed
from twostage.train_logger import TrainLoggerTwoStage
from twostage.dataset_tumor_roi import TumorROIDataset
from twostage.train_eval_tumor import tumor_metrics_from_val_result

"""
os.path.abspath(medseg_root)
把相对路径变为绝对路径;比如
输入:medseg
输出:/home/pumengyu/medseg

sys.path是Python的模块搜索路径列表,import 语句会按照顺序在这些目录里面查找模块

sys.path.insert(0, medseg_root)是为了把medseg_root插入到列表的第0位

margin是肿瘤裁剪向外扩展的像素数
"""


def add_medseg_to_syspath(medseg_root: str) -> None:
    medseg_root = os.path.abspath(medseg_root)
    if medseg_root not in sys.path:
        sys.path.insert(0, medseg_root)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--medseg_root", type=str, required=True)
    p.add_argument("--preprocessed_root", type=str, required=True)
    p.add_argument(
        "--exp_root", type=str, default="/home/pumengyu/experiments_twostage"
    )
    p.add_argument("--exp_name", type=str, default="tumor_roi_dynunet")
    p.add_argument("--model", type=str, default="dynunet")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--patch", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--val_patch", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--sw_batch_size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--prefetch_factor", type=int, default=4)

    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--loss", type=str, default="dicece", choices=["dicece", "dicefocal", "tversky"]
    )
    p.add_argument("--val_every", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_n", type=int, default=0)
    p.add_argument("--val_n", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init_ckpt", type=str, default=None)  # 新增的初始化权重参数
    p.add_argument("--tumor_ratios", type=float, nargs=2, default=[0.2, 0.8])
    p.add_argument("--margin", type=int, default=12)
    # parse_args() 里新增一个参数
    p.add_argument(
        "--learnable_loss",
        action="store_true",
        help="启用可学习权重损失（alpha自动学习liver/tumor平衡）",
    )
    return p.parse_args()


def load_init_weights(path, model, map_location="cpu"):
    """
        从checkpoint里面挑出匹配的参数,加载到当前模型,其余全部忽略

        这里的
        src =
    {
        "encoder.conv1.weight": tensor(...)
        "encoder.conv1.bias": tensor(...)
    }
    model是当前模型参数
    src是checkpoint参数
    dst.update(matched)
    这个是把匹配的参数覆盖到当前模型参数


    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    dst = model.state_dict()
    matched = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    dst.update(matched)
    model.load_state_dict(dst, strict=False)
    print(f"[init] loaded {len(matched)} matched params from {path}")


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

    all_pt = load_pt_paths(args.preprocessed_root)
    tr, va, te = split_three_ways(
        all_pt,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    # tr,va,te都是路径列表

    tr_pos, tr_neg = filter_tumor_positive_cases(tr)
    va_pos, va_neg = filter_tumor_positive_cases(va)
    te_pos, te_neg = filter_tumor_positive_cases(te)

    print(f"[split raw] train={len(tr)} val={len(va)} test={len(te)}")
    print(f"[有肿瘤部分] train={len(tr_pos)} val={len(va_pos)} test={len(te_pos)}")
    print(f"[无肿瘤部分] train={len(tr_neg)} val={len(va_neg)} test={len(te_neg)}")

    tr = tr_pos
    va = va_pos

    if args.train_n > 0:
        tr = tr[: args.train_n]
    if args.val_n > 0:
        va = va[: args.val_n]

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
        "overlap": float(args.overlap),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "repeats": int(args.repeats),
        "amp": bool(args.amp),
        "loss": args.loss,
        "val_every": int(args.val_every),
        "seed": int(args.seed),
        "tumor_ratios": list(args.tumor_ratios),
        "margin": int(args.margin),
        "n_train_cases": len(tr),
        "n_val_cases": len(va),
        "n_test_cases": len(te),
        "num_classes": 2,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "T_max": int(args.epochs),
        "in_channels": 1,
        "out_channels": 2,
        "resume": args.resume,
        "learnable_loss": bool(args.learnable_loss),
    }
    save_json(config, workdir, "config")

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
    )
    val_ds = TumorROIDataset(
        va,
        transform=val_tf,
        repeats=1,
        margin=args.margin,
        keep_meta=True,
    )

    if args.num_workers > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        args.model, in_channels=1, out_channels=2, img_size=tuple(args.patch)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.learnable_loss:
        criterion = LearnableWeightedLoss(base_loss_type=args.loss).to(device)
        criterion_optimizer = torch.optim.Adam(criterion.parameters(), lr=1e-3)
    else:
        criterion = None
        criterion_optimizer = None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device == "cuda" else None

    start_epoch = 1
    best = -1.0
    best_epoch = 0

    # 新增部分：加载初始化权重
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
    print(f"[tumor stage2] train={len(tr)} val={len(va)} test={len(te)}")
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
            train_loss = train_one_epoch_binary(  # 原有代码完全不变
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                loss_type=args.loss,
                epoch=epoch,
                epochs=args.epochs,
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
                overlap=args.overlap,
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
        if args.learnable_loss and criterion is not None:
            w = criterion.get_weights()
            logger.log_extra(epoch=epoch, w_liver=w["w_liver"], w_tumor=w["w_tumor"])

    total_sec = time.time() - wall_start
    metrics = {
        "best_tumor_dice": round(float(best), 4),
        "best_epoch": int(best_epoch),
        "epochs": int(args.epochs),
        "n_train_cases": len(tr),
        "n_val_cases": len(va),
        "total_train_hours": round(total_sec / 3600.0, 2),
    }
    save_json(metrics, workdir, "metrics")

    save_report(
        text="\n".join([f"{k}: {v}" for k, v in metrics.items()]),
        out_dir=workdir,
        filename="report.txt",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
