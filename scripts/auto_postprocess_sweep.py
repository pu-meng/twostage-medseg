"""
auto_postprocess_sweep.py
=========================
仿照 nnUNet 的做法：在验证集上自动搜索最优后处理参数，
找到最优配置后固定下来用于测试集评估。

用法:
    python scripts/auto_postprocess_sweep.py \
        --stage2_ckpt /path/to/best.pt \
        [其他eval参数...]
"""

import argparse
import subprocess
import json
import re
import sys
import os
import itertools
from pathlib import Path


PYTHON = sys.executable
EVAL_SCRIPT = str(Path(__file__).parent / "eval_twostage.py")


def parse_dice_from_report(report_path):
    """从report.txt里解析Tumor Dice均值"""
    try:
        with open(report_path) as f:
            for line in f:
                m = re.search(r"Dice:\s*mean=([\d.]+)", line)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return -1.0


def find_latest_report(save_dir):
    """eval输出到save_dir/<timestamp>/report.txt，找最新的"""
    candidates = sorted(Path(save_dir).glob("*/report.txt"), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def run_eval(base_args, extra_args, save_dir):
    """跑一次eval，返回tumor dice"""
    cmd = [PYTHON, EVAL_SCRIPT] + base_args + [
        "--save_dir", save_dir,
        "--split", "val",
        "--no_tta",  # sweep时关TTA加速
    ] + extra_args
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = find_latest_report(save_dir)
    return parse_dice_from_report(report) if report else -1.0


def main():
    p = argparse.ArgumentParser()
    # 透传给eval的基础参数
    p.add_argument("--medseg_root", required=True)
    p.add_argument("--preprocessed_root", required=True)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--stage2_ckpt", required=True)
    p.add_argument("--stage1_model", default="dynunet")
    p.add_argument("--stage2_model", default="dynunet_deep")
    p.add_argument("--stage1_patch", nargs=3, default=["144", "144", "144"])
    p.add_argument("--stage2_patch", nargs=3, default=["160", "160", "160"])
    p.add_argument("--stage1_out_channels", default="3")
    p.add_argument("--val_ratio", default="0.2")
    p.add_argument("--test_ratio", default="0.1")
    p.add_argument("--seed", default="0")
    p.add_argument("--sweep_dir", default=None, help="sweep临时结果保存目录")
    args = p.parse_args()

    base_args = [
        "--medseg_root", args.medseg_root,
        "--preprocessed_root", args.preprocessed_root,
        "--stage1_ckpt", args.stage1_ckpt,
        "--stage2_ckpt", args.stage2_ckpt,
        "--stage1_model", args.stage1_model,
        "--stage2_model", args.stage2_model,
        "--stage1_patch", *args.stage1_patch,
        "--stage2_patch", *args.stage2_patch,
        "--stage1_out_channels", args.stage1_out_channels,
        "--val_ratio", args.val_ratio,
        "--test_ratio", args.test_ratio,
        "--seed", args.seed,
    ]

    sweep_root = args.sweep_dir or os.path.join(
        os.path.dirname(args.stage2_ckpt), "..", "..", "sweep_postprocess"
    )
    os.makedirs(sweep_root, exist_ok=True)

    # ── 搜索空间（对应nnUNet的postprocessing search）──────────────────────
    prob_thresholds   = [0.2, 0.3, 0.4, 0.5]
    min_tumor_sizes   = [0, 100, 200, 500]
    # 是否开启连通域过滤（对应nnUNet的remove_all_but_the_largest_connected_component）
    use_postprocess   = [True, False]

    best_dice = -1.0
    best_cfg  = {}
    results   = []

    combos = list(itertools.product(prob_thresholds, min_tumor_sizes, use_postprocess))
    print(f"共 {len(combos)} 组参数，在验证集上sweep...")

    for i, (pt, mts, use_pp) in enumerate(combos):
        tag = f"pt{pt}_mts{mts}_pp{int(use_pp)}"
        save_dir = os.path.join(sweep_root, tag)
        os.makedirs(save_dir, exist_ok=True)

        extra = [
            "--prob_threshold", str(pt),
            "--min_tumor_size", str(mts),
        ]
        if not use_pp:
            extra.append("--no_postprocess")

        print(f"[{i+1}/{len(combos)}] {tag} ...", end=" ", flush=True)
        dice = run_eval(base_args, extra, save_dir)
        print(f"val_dice={dice:.4f}")
        results.append({"tag": tag, "prob_threshold": pt, "min_tumor_size": mts,
                        "postprocess": use_pp, "val_dice": dice})
        if dice > best_dice:
            best_dice = dice
            best_cfg = {"prob_threshold": pt, "min_tumor_size": mts, "postprocess": use_pp}

    # 保存sweep结果
    results.sort(key=lambda x: -x["val_dice"])
    result_path = os.path.join(sweep_root, "sweep_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"最优配置 (val_dice={best_dice:.4f}):")
    print(f"  prob_threshold : {best_cfg['prob_threshold']}")
    print(f"  min_tumor_size : {best_cfg['min_tumor_size']}")
    print(f"  postprocess    : {best_cfg['postprocess']}")
    print(f"sweep结果已保存: {result_path}")

    # 用最优配置在测试集上评估
    print(f"\n用最优配置在测试集上评估...")
    test_extra = [
        "--prob_threshold", str(best_cfg["prob_threshold"]),
        "--min_tumor_size", str(best_cfg["min_tumor_size"]),
        "--split", "test",
    ]
    if not best_cfg["postprocess"]:
        test_extra.append("--no_postprocess")

    # 去掉base_args里的--split val（run_eval里写死了val，这里直接调subprocess）
    test_save_dir = os.path.join(sweep_root, "best_on_test")
    os.makedirs(test_save_dir, exist_ok=True)
    cmd = [PYTHON, EVAL_SCRIPT] + base_args + [
        "--save_dir", test_save_dir,
    ] + test_extra
    subprocess.run(cmd, check=True)

    print(f"\n测试集结果: {os.path.join(test_save_dir, 'report.txt')}")


if __name__ == "__main__":
    main()
