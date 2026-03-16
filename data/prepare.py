import os
from typing import Tuple
from datetime import datetime
import json
from typing import Optional

def prepare_workdir_and_config(args) -> Tuple[str, Optional[str]]:
    """
    创建输出目录并保存 config.json
    返回:
        workdir, pred_dir
    """
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    workdir = os.path.join(args.save_dir, timestamp)
    os.makedirs(workdir, exist_ok=True)

    pred_dir: Optional[str] = None
    if args.save_pred_pt:
        pred_dir = os.path.join(workdir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)

    config = {
        "project_root": os.path.abspath(args.project_root),
        "preprocessed_root": os.path.abspath(args.preprocessed_root),
        "stage1_ckpt": os.path.abspath(args.stage1_ckpt),
        "stage2_ckpt": os.path.abspath(args.stage2_ckpt),
        "stage1_model": args.stage1_model,
        "stage2_model": args.stage2_model,
        "stage1_patch": list(args.stage1_patch),
        "stage2_patch": list(args.stage2_patch),
        "stage1_sw_batch_size": int(args.stage1_sw_batch_size),
        "stage2_sw_batch_size": int(args.stage2_sw_batch_size),
        "overlap": float(args.overlap),
        "margin": int(args.margin),
        "n": int(args.n),
        "save_pred_pt": bool(args.save_pred_pt),
        "workdir": workdir,
    }

    with open(os.path.join(workdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return workdir, pred_dir
